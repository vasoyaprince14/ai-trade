"""
Order Flow Based F&O Strategy
================================
Core of the platform - uses OI analysis to trade options.

Integrates:
  - vendors/algo-strategies/short-straddle/ (straddle execution)
  - vendors/non-directional-strategy/ (non-directional logic)
  - Our OI Tracker for signals

Strategies:
  1. Short Straddle  - Sell ATM CE + PE when IV is high
  2. Iron Fly        - Short ATM straddle + Long OTM wings (hedged)
  3. OI Flow Trade   - Buy directional options based on OI signals

Entry conditions (from order flow analysis):
  - LONG_BUILD_UP  → Buy Call
  - SHORT_BUILD_UP → Buy Put
  - GAMMA_PINNING  → Sell Straddle (range-bound)
  - High IV Rank   → Sell Premium (straddle/iron fly)
"""
import sys
from pathlib import Path
from datetime import datetime, date
from typing import List, Optional, Dict

import pandas as pd
from loguru import logger

from core.strategies.base import BaseStrategy, TradeSignal
from core.order_flow.oi_tracker import OITracker
from config.settings import config

FNO_CFG = config.get("strategies", {}).get("order_flow_fno", {})
STRADDLE_CFG = config.get("strategies", {}).get("straddle", {})


class OrderFlowFNOStrategy(BaseStrategy):
    """
    Main F&O strategy that combines order flow signals with
    premium selling and directional options buying.

    Adapted from:
    - vendors/algo-strategies/short-straddle/0920_short_straddle/nifty50_0920_short_straddle.py
    - vendors/non-directional-strategy/
    """

    name = "order_flow_fno"
    description = "OI order flow based F&O: straddle + directional options"

    def __init__(self, broker=None, cfg: Dict = None, symbol: str = "NIFTY"):
        super().__init__(broker, cfg or FNO_CFG)
        self.symbol = symbol
        self.strategy_type = self.config.get("strategy_type", "straddle")
        self.exit_profit_pct = self.config.get("exit_profit_pct", 40)
        self.exit_loss_pct = self.config.get("exit_loss_pct", 100)
        self.dte_entry = self.config.get("days_to_expiry_entry", 14)
        self.dte_exit = self.config.get("days_to_expiry_exit", 3)
        self.delta_neutral = self.config.get("delta_neutral", True)
        self.iv_rank_min = self.config.get("sell_iv_rank_min", 50)

        self.oi_tracker = OITracker(symbol)
        self._open_positions: List[Dict] = []

        # Market config
        lot_sizes = config.get("market", {}).get("lot_sizes", {})
        self.lot_size = lot_sizes.get(symbol, 25)

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[TradeSignal]:
        """
        Generate F&O signals based on:
        1. Order flow analysis (OI changes)
        2. IV conditions (high IV → sell, low IV → buy)
        3. PCR sentiment
        """
        signals = []
        expiry = kwargs.get("expiry")

        # Get current market state from OI tracker
        try:
            state = self.oi_tracker.snapshot(expiry)
        except Exception as e:
            logger.error(f"OI snapshot failed: {e}")
            return signals

        if not state:
            return signals

        flow_signal = state.get("flow_signal", "NEUTRAL")
        pcr_signal = state.get("pcr_signal", "NEUTRAL")
        spot = state.get("spot", 0)
        atm_iv = state.get("atm_iv", 0)
        max_pain = state.get("max_pain", 0)

        if spot == 0:
            return signals

        atm_strike = round(spot / 50) * 50

        # ── Strategy 1: Sell Premium (Straddle / Iron Fly) ─────────────
        # Trigger: Gamma Pinning OR high IV → range-bound market
        if flow_signal == "GAMMA_PINNING" or self._is_high_iv(atm_iv):
            if self.strategy_type in ("straddle", "iron_fly"):
                signals += self._generate_straddle_signals(
                    spot, atm_strike, expiry, state
                )

        # ── Strategy 2: Directional Options Buy ─────────────────────────
        # Trigger: Strong OI buildup with PCR confirmation
        elif flow_signal == "LONG_BUILD_UP" and pcr_signal in ("BULLISH", "SLIGHTLY_BULLISH"):
            signals.append(self._directional_signal("CE", spot, atm_strike, expiry, state, flow_signal))

        elif flow_signal == "SHORT_BUILD_UP" and pcr_signal in ("BEARISH", "SLIGHTLY_BEARISH"):
            signals.append(self._directional_signal("PE", spot, atm_strike, expiry, state, flow_signal))

        # ── Strategy 3: Max Pain Reversion ─────────────────────────────
        # If spot is far from max pain with 3-5 DTE, expect reversion
        elif max_pain and abs(spot - max_pain) > 100:
            signals += self._max_pain_trade(spot, max_pain, atm_strike, expiry, state)

        return [s for s in signals if s is not None]

    def _generate_straddle_signals(self, spot, atm, expiry, state) -> List[TradeSignal]:
        """
        Short straddle: Sell ATM CE + Sell ATM PE.
        Based on vendors/algo-strategies/short-straddle/ logic.
        """
        ce_sym = f"{self.symbol}_ATM_{expiry}_CE"  # resolved by broker
        pe_sym = f"{self.symbol}_ATM_{expiry}_PE"
        atm_iv = state.get("atm_iv", 20)

        return [
            TradeSignal(
                strategy=self.name,
                symbol=ce_sym,
                direction="SELL",
                signal_type="ENTRY",
                price=spot,
                confidence=0.7,
                notes=f"Short Straddle CE | ATM={atm} | IV={atm_iv:.1f}% | Gamma Pin",
                meta={
                    "option_type": "CE",
                    "atm_strike": atm,
                    "expiry": expiry,
                    "straddle_leg": "CE",
                    "exit_profit_pct": self.exit_profit_pct,
                    "exit_loss_pct": self.exit_loss_pct,
                }
            ),
            TradeSignal(
                strategy=self.name,
                symbol=pe_sym,
                direction="SELL",
                signal_type="ENTRY",
                price=spot,
                confidence=0.7,
                notes=f"Short Straddle PE | ATM={atm} | IV={atm_iv:.1f}% | Gamma Pin",
                meta={
                    "option_type": "PE",
                    "atm_strike": atm,
                    "expiry": expiry,
                    "straddle_leg": "PE",
                    "exit_profit_pct": self.exit_profit_pct,
                    "exit_loss_pct": self.exit_loss_pct,
                }
            ),
        ]

    def _directional_signal(self, opt_type, spot, atm, expiry, state, flow_sig) -> TradeSignal:
        """Buy directional option on strong OI flow signal."""
        sym = f"{self.symbol}_ATM_{expiry}_{opt_type}"
        pcr = state.get("pcr_oi", 1.0)
        flow_score = state.get("flow_score", 0.5)
        confidence = min((flow_score + 0.5) / 1.5, 0.95)

        # Set target/SL as % of option premium
        target_pct = self.config.get("target_pct", 50) / 100
        sl_pct = self.config.get("sl_pct", 30) / 100

        return TradeSignal(
            strategy=self.name,
            symbol=sym,
            direction="BUY",
            signal_type="ENTRY",
            price=0,  # Market order
            confidence=confidence,
            notes=(
                f"OI Flow {opt_type} | Signal: {flow_sig} | "
                f"PCR: {pcr:.2f} | Score: {flow_score:.2f}"
            ),
            meta={
                "option_type": opt_type,
                "atm_strike": atm,
                "expiry": expiry,
                "flow_signal": flow_sig,
                "target_pct": target_pct,
                "sl_pct": sl_pct,
            }
        )

    def _max_pain_trade(self, spot, max_pain, atm, expiry, state) -> List[TradeSignal]:
        """Trade max pain reversion with 3-5 DTE."""
        signals = []
        diff = max_pain - spot
        if abs(diff) < 50:
            return signals

        # If spot below max pain → bullish bias (buy CE or sell PE)
        if diff > 100:
            signals.append(TradeSignal(
                strategy=self.name,
                symbol=f"{self.symbol}_ATM_{expiry}_CE",
                direction="BUY",
                signal_type="ENTRY",
                confidence=0.6,
                notes=f"Max Pain Reversion UP | Spot={spot:.0f} Pain={max_pain:.0f} Diff={diff:.0f}",
                meta={"option_type": "CE", "atm_strike": atm, "max_pain": max_pain}
            ))
        # If spot above max pain → bearish bias
        elif diff < -100:
            signals.append(TradeSignal(
                strategy=self.name,
                symbol=f"{self.symbol}_ATM_{expiry}_PE",
                direction="BUY",
                signal_type="ENTRY",
                confidence=0.6,
                notes=f"Max Pain Reversion DOWN | Spot={spot:.0f} Pain={max_pain:.0f} Diff={diff:.0f}",
                meta={"option_type": "PE", "atm_strike": atm, "max_pain": max_pain}
            ))
        return signals

    def _is_high_iv(self, current_iv: float, threshold: float = 20.0) -> bool:
        """Simple IV check - high IV favors selling premium."""
        return current_iv > threshold
