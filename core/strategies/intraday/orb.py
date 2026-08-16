"""
Opening Range Breakout (ORB) Strategy
=======================================
Timeframe: 5-min bars
Logic:
  - First 15 minutes = Opening Range (high/low)
  - Breakout above range high → Buy Call
  - Breakdown below range low → Buy Put
  - Target: 2R, SL: range width
"""
from typing import List, Optional, Dict
from datetime import datetime, time
import pandas as pd
from loguru import logger

from core.strategies.base import BaseStrategy, TradeSignal
from config.settings import config

ORB_CFG = config.get("strategies", {}).get("orb", {})


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout for Nifty options."""

    name = "orb"
    description = "15-min opening range breakout for Nifty options"

    def __init__(self, broker=None, cfg: Dict = None):
        super().__init__(broker, cfg or ORB_CFG)
        self.range_minutes = self.config.get("range_minutes", 15)
        self.target_r = self.config.get("target_r", 2.0)
        self.sl_r = self.config.get("sl_r", 1.0)
        self.max_trades = self.config.get("max_trades_per_day", 2)
        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        self._or_established = False
        self._trades_today = 0
        self._signal_given = {"CE": False, "PE": False}

    def _reset_day(self):
        self._or_high = None
        self._or_low = None
        self._or_established = False
        self._trades_today = 0
        self._signal_given = {"CE": False, "PE": False}

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[TradeSignal]:
        if data.empty:
            return []
        symbol = kwargs.get("symbol", "NIFTY")
        signals = []

        # Check if new day - reset state
        if len(data) > 0:
            first_ts = pd.to_datetime(data.iloc[0]["timestamp"])
            if first_ts.hour == 9 and first_ts.minute == 15:
                self._reset_day()

        # Build opening range from first N bars
        market_open = time(9, 15)
        or_cutoff = time(9, 15 + self.range_minutes)

        or_bars = data[
            data["timestamp"].apply(
                lambda x: pd.to_datetime(x).time() < or_cutoff
            )
        ]

        if len(or_bars) < self.range_minutes // 5:
            return []  # OR not established yet

        self._or_high = or_bars["high"].max()
        self._or_low = or_bars["low"].min()
        self._or_established = True
        or_range = self._or_high - self._or_low

        if or_range <= 0:
            return []

        # Only trade bars after OR is established
        post_or = data[
            data["timestamp"].apply(
                lambda x: pd.to_datetime(x).time() >= or_cutoff
            )
        ]

        if post_or.empty:
            return []

        latest = post_or.iloc[-1]
        close = latest["close"]
        atm = round(close / 50) * 50

        # Bullish breakout
        if close > self._or_high and not self._signal_given["CE"] and self._trades_today < self.max_trades:
            target = close + or_range * self.target_r
            sl = self._or_high - or_range * self.sl_r
            self._signal_given["CE"] = True
            self._trades_today += 1
            signals.append(TradeSignal(
                strategy=self.name,
                symbol=f"{symbol}_ATM_CE",
                direction="BUY",
                signal_type="ENTRY",
                price=close,
                target=target,
                stop_loss=sl,
                confidence=0.75,
                notes=f"ORB Breakout UP | OR: {self._or_low:.0f}-{self._or_high:.0f} | Range: {or_range:.0f}",
                meta={"option_type": "CE", "atm_strike": atm, "or_high": self._or_high, "or_low": self._or_low}
            ))

        # Bearish breakdown
        elif close < self._or_low and not self._signal_given["PE"] and self._trades_today < self.max_trades:
            target = close - or_range * self.target_r
            sl = self._or_low + or_range * self.sl_r
            self._signal_given["PE"] = True
            self._trades_today += 1
            signals.append(TradeSignal(
                strategy=self.name,
                symbol=f"{symbol}_ATM_PE",
                direction="BUY",
                signal_type="ENTRY",
                price=close,
                target=target,
                stop_loss=sl,
                confidence=0.75,
                notes=f"ORB Breakdown DOWN | OR: {self._or_low:.0f}-{self._or_high:.0f} | Range: {or_range:.0f}",
                meta={"option_type": "PE", "atm_strike": atm}
            ))

        return signals
