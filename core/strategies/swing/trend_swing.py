"""
Swing Trend Following Strategy
================================
Timeframe: Daily (1d)
Logic:
  - EMA 21/55 trend direction
  - MACD crossover signal
  - ADX > 25 for trend strength
  - ATR-based SL and target

Entry: Long above EMA55, Short below EMA55, confirmed by MACD
Hold: 3-20 days
"""
from typing import List, Dict
import pandas as pd
from loguru import logger

from core.strategies.base import BaseStrategy, TradeSignal
from config.settings import config

SWING_CFG = config.get("strategies", {}).get("swing_trend", {})


class SwingTrendStrategy(BaseStrategy):
    """EMA+MACD swing strategy for Nifty equities and F&O."""

    name = "swing_trend"
    description = "Daily EMA 21/55 + MACD swing trend following"

    def __init__(self, broker=None, cfg: Dict = None):
        super().__init__(broker, cfg or SWING_CFG)
        self.ema_fast = self.config.get("ema_fast", 21)
        self.ema_slow = self.config.get("ema_slow", 55)
        self.sl_atr_mult = self.config.get("sl_atr_multiplier", 2.0)
        self.target_atr_mult = self.config.get("target_atr_multiplier", 4.0)
        self.adx_threshold = self.config.get("adx_threshold", 20)

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[TradeSignal]:
        signals = []
        if data.empty or len(data) < 60:
            return signals

        symbol = kwargs.get("symbol", "NIFTY")
        latest = data.iloc[-1].to_dict()
        prev = data.iloc[-2].to_dict()

        close = latest.get("close", 0)
        ema_fast = latest.get(f"ema{self.ema_fast}", 0) or latest.get("ema21", 0)
        ema_slow = latest.get(f"ema{self.ema_slow}", 0) or latest.get("ema50", 0)
        macd = latest.get("macd", 0)
        macd_sig = latest.get("macd_signal", 0)
        prev_macd = prev.get("macd", 0)
        prev_macd_sig = prev.get("macd_signal", 0)
        atr = latest.get("atr", close * 0.01)
        adx = latest.get("adx", 0)

        if ema_fast == 0 or ema_slow == 0:
            return signals

        trend_up = close > ema_slow and ema_fast > ema_slow
        trend_down = close < ema_slow and ema_fast < ema_slow
        trend_strength = adx > self.adx_threshold if adx > 0 else True

        macd_cross_up = macd > macd_sig and prev_macd <= prev_macd_sig
        macd_cross_down = macd < macd_sig and prev_macd >= prev_macd_sig

        # Bullish: Trend up + MACD cross up
        if trend_up and macd_cross_up and trend_strength:
            sl = close - atr * self.sl_atr_mult
            target = close + atr * self.target_atr_mult
            signals.append(TradeSignal(
                strategy=self.name,
                symbol=symbol,
                direction="BUY",
                signal_type="ENTRY",
                price=close,
                target=target,
                stop_loss=sl,
                confidence=0.72,
                notes=f"Swing BUY | EMA{self.ema_fast}>{self.ema_slow} + MACD cross up | ADX={adx:.1f}",
                meta={"atr": atr, "ema_fast": ema_fast, "ema_slow": ema_slow}
            ))

        # Bearish: Trend down + MACD cross down
        elif trend_down and macd_cross_down and trend_strength:
            sl = close + atr * self.sl_atr_mult
            target = close - atr * self.target_atr_mult
            signals.append(TradeSignal(
                strategy=self.name,
                symbol=symbol,
                direction="SELL",
                signal_type="ENTRY",
                price=close,
                target=target,
                stop_loss=sl,
                confidence=0.68,
                notes=f"Swing SELL | EMA{self.ema_fast}<{self.ema_slow} + MACD cross down | ADX={adx:.1f}",
                meta={"atr": atr}
            ))

        return signals
