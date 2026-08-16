"""
Intraday Scalp Momentum Strategy
==================================
Timeframe: 5-min / 15-min
Logic:
  - VWAP Momentum Breakout (from ai-trader signal_generator.py)
  - EMA 9/21 crossover confirmation
  - RSI filter (not overbought/oversold)
  - Volume spike for confirmation
  - Buy ATM Call on bullish signal, Buy ATM Put on bearish

Integrates:
  vendors/ai-trader/strategy/signal_generator.py
  vendors/ai-trader/strategy/regime_detector.py
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent.parent.parent
VENDOR_DIR = ROOT_DIR / "vendors"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.strategies.base import BaseStrategy, TradeSignal
from config.settings import config

STRAT_CFG = config.get("strategies", {}).get("scalp_momentum", {})


class ScalpMomentumStrategy(BaseStrategy):
    """
    VWAP + EMA momentum strategy for intraday Nifty options.
    Buys ATM Call on bullish breakout, ATM Put on bearish breakdown.
    """

    name = "scalp_momentum"
    description = "5-min VWAP+EMA momentum with RSI & volume filter"

    def __init__(self, broker=None, cfg: Dict = None):
        super().__init__(broker, cfg or STRAT_CFG)
        self.ema_fast = self.config.get("ema_fast", 9)
        self.ema_slow = self.config.get("ema_slow", 21)
        self.rsi_ob = self.config.get("rsi_overbought", 70)
        self.rsi_os = self.config.get("rsi_oversold", 30)
        self.max_trades = self.config.get("max_trades_per_day", 5)
        self.target_pct = self.config.get("target_pct", 0.6)
        self.sl_pct = self.config.get("sl_pct", 0.3)
        self._trades_today = 0

        # Try loading ai-trader signal generators
        self._vwap_signal = None
        self._bearish_signal = None
        try:
            ai_path = str(VENDOR_DIR / "ai-trader")
            if ai_path not in sys.path:
                sys.path.append(ai_path)
            from strategy.signal_generator import vwap_momentum_breakout, bearish_momentum
            self._vwap_signal = vwap_momentum_breakout
            self._bearish_signal = bearish_momentum
            logger.info("Loaded ai-trader signal generators")
        except (ImportError, Exception):
            pass

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[TradeSignal]:
        """
        Generate signals from 5-min OHLCV data with indicators.
        Expects columns: timestamp, open, high, low, close, volume,
                         ema9, ema21, rsi, vwap, volume_ratio
        """
        signals = []
        if data.empty or len(data) < 20:
            return signals
        if self._trades_today >= self.max_trades:
            return signals

        latest = data.iloc[-1].to_dict()
        prev = data.iloc[-2].to_dict()

        # Check ATM options for symbol
        symbol = kwargs.get("symbol", "NIFTY")
        spot = latest.get("close", 0)

        # ── Use ai-trader signal generators if available ──────────────
        if self._vwap_signal:
            sig = self._vwap_signal(latest, symbol)
            if sig:
                atm = round(spot / 50) * 50
                option_sym = f"{symbol}_ATM_CE"  # Will be resolved by broker
                signals.append(TradeSignal(
                    strategy=self.name,
                    symbol=option_sym,
                    direction="BUY",
                    signal_type="ENTRY",
                    price=latest.get("close", 0),
                    confidence=sig.technical_strength,
                    notes=f"VWAP Bullish | {sig.details}",
                    meta={"option_type": "CE", "atm_strike": atm, "ai_details": sig.details}
                ))

        if self._bearish_signal:
            sig = self._bearish_signal(latest, symbol)
            if sig:
                atm = round(spot / 50) * 50
                option_sym = f"{symbol}_ATM_PE"
                signals.append(TradeSignal(
                    strategy=self.name,
                    symbol=option_sym,
                    direction="BUY",
                    signal_type="ENTRY",
                    price=latest.get("close", 0),
                    confidence=sig.technical_strength,
                    notes=f"Bearish Momentum | {sig.details}",
                    meta={"option_type": "PE", "atm_strike": atm}
                ))

        # ── Built-in fallback signals ──────────────────────────────────
        if not signals:
            signals = self._builtin_signals(data, latest, prev, symbol)

        return signals

    def _builtin_signals(self, data, latest, prev, symbol) -> List[TradeSignal]:
        """Built-in EMA + RSI + Volume momentum."""
        signals = []
        close = latest.get("close", 0)
        ema9 = latest.get("ema9", 0)
        ema21 = latest.get("ema21", 0)
        rsi = latest.get("rsi", 50)
        vwap = latest.get("vwap", close)
        vol_ratio = latest.get("volume_ratio", 1.0)

        prev_ema9 = prev.get("ema9", 0)
        prev_ema21 = prev.get("ema21", 0)

        # Bullish: EMA9 crosses above EMA21, price > VWAP, RSI not overbought, volume spike
        bullish = (
            ema9 > ema21 and prev_ema9 <= prev_ema21
            and close > vwap
            and rsi < self.rsi_ob
            and vol_ratio > 1.3
        )

        # Bearish: EMA9 crosses below EMA21, price < VWAP, RSI not oversold, volume spike
        bearish = (
            ema9 < ema21 and prev_ema9 >= prev_ema21
            and close < vwap
            and rsi > self.rsi_os
            and vol_ratio > 1.3
        )

        atm = round(close / 50) * 50

        if bullish:
            target = close * (1 + self.target_pct / 100)
            sl = close * (1 - self.sl_pct / 100)
            signals.append(TradeSignal(
                strategy=self.name,
                symbol=f"{symbol}_ATM_CE",
                direction="BUY",
                signal_type="ENTRY",
                price=close,
                target=target,
                stop_loss=sl,
                confidence=0.7,
                notes=f"EMA bullish cross | RSI={rsi:.1f} | Vol={vol_ratio:.2f}x",
                meta={"option_type": "CE", "atm_strike": atm}
            ))

        elif bearish:
            target = close * (1 + self.target_pct / 100)
            sl = close * (1 - self.sl_pct / 100)
            signals.append(TradeSignal(
                strategy=self.name,
                symbol=f"{symbol}_ATM_PE",
                direction="BUY",
                signal_type="ENTRY",
                price=close,
                target=target,
                stop_loss=sl,
                confidence=0.7,
                notes=f"EMA bearish cross | RSI={rsi:.1f} | Vol={vol_ratio:.2f}x",
                meta={"option_type": "PE", "atm_strike": atm}
            ))

        return signals
