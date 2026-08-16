"""
Long-Term Positional Strategy (Weekly)
========================================
Timeframe: Weekly bars
Logic:
  - EMA 200 trend filter
  - ADX > 25 for trend strength
  - Buy when price pulls back to EMA in uptrend
  - Target: 4x ATR, SL: 2x ATR
  - Hold: 4-52 weeks
"""
from typing import List, Dict
import pandas as pd
from loguru import logger

from core.strategies.base import BaseStrategy, TradeSignal
from config.settings import config

LT_CFG = config.get("strategies", {}).get("long_term", {})


class LongTermStrategy(BaseStrategy):
    """Weekly EMA 200 trend-following for positional Nifty trades."""

    name = "long_term"
    description = "Weekly EMA200 positional trend following"

    def __init__(self, broker=None, cfg: Dict = None):
        super().__init__(broker, cfg or LT_CFG)
        self.ema_period = self.config.get("ema", 200)
        self.adx_threshold = self.config.get("adx_threshold", 25)

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[TradeSignal]:
        signals = []
        if data.empty or len(data) < 210:
            logger.warning("Insufficient data for long-term strategy (need 200+ bars)")
            return signals

        symbol = kwargs.get("symbol", "NIFTY")
        latest = data.iloc[-1].to_dict()
        prev = data.iloc[-2].to_dict()

        close = latest.get("close", 0)
        ema200 = latest.get("ema200", 0)
        atr = latest.get("atr", close * 0.02)
        adx = latest.get("adx", 0)
        rsi = latest.get("rsi", 50)

        if ema200 == 0:
            return signals

        # Long-term uptrend conditions
        uptrend = close > ema200
        strong_trend = adx > self.adx_threshold if adx > 0 else True
        pullback = close < prev.get("close", close) * 1.01  # price not at ATH

        # Buy signal: In uptrend, price dips near EMA
        proximity_to_ema = abs(close - ema200) / ema200
        near_ema = proximity_to_ema < 0.03  # within 3% of EMA200

        if uptrend and strong_trend and (near_ema or pullback) and rsi < 65:
            sl = close - atr * 2
            target = close + atr * 4
            signals.append(TradeSignal(
                strategy=self.name,
                symbol=symbol,
                direction="BUY",
                signal_type="ENTRY",
                price=close,
                target=target,
                stop_loss=sl,
                confidence=0.65,
                notes=(
                    f"LT Buy | EMA200={ema200:.0f} | ADX={adx:.1f} | "
                    f"RSI={rsi:.1f} | Proximity={proximity_to_ema:.1%}"
                ),
                meta={"ema200": ema200, "atr": atr, "adx": adx}
            ))

        # Exit signal: Close below EMA200
        if not uptrend and prev.get("close", close) > ema200:
            signals.append(TradeSignal(
                strategy=self.name,
                symbol=symbol,
                direction="SELL",
                signal_type="EXIT",
                price=close,
                confidence=0.8,
                notes=f"LT Exit: Close below EMA200 ({ema200:.0f})",
                meta={"ema200": ema200}
            ))

        return signals
