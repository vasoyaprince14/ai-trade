"""
XAUUSD Live Data Fetcher
=========================
Gold (GC=F) OHLCV via yfinance + macro context (DXY, 10Y yield, VIX).
yfinance gives 15-20min delayed futures data — sufficient for 15m/1H signals.
"""

import yfinance as yf
import pandas as pd
from loguru import logger

GOLD   = "GC=F"
DXY    = "DX-Y.NYB"
TNX    = "^TNX"   # US 10Y yield
VIX    = "^VIX"


def get_price() -> float:
    """
    Current gold price — uses 1m bar (freshest available ~10min delay during US hours).
    GC=F is gold futures; most liquid and closest to spot XAUUSD.
    """
    try:
        df = yf.download(GOLD, period="1d", interval="1m", progress=False)
        if not df.empty:
            val = df["Close"].iloc[-1]
            return float(val.values[0]) if hasattr(val, "values") else float(val)
    except Exception:
        pass
    try:
        df = yf.download(GOLD, period="5d", interval="5m", progress=False)
        if not df.empty:
            val = df["Close"].iloc[-1]
            return float(val.values[0]) if hasattr(val, "values") else float(val)
    except Exception as e:
        logger.warning(f"get_price failed: {e}")
    return 0.0


def get_bars(interval: str = "15m", period: str = "5d") -> pd.DataFrame:
    """
    OHLCV bars for GC=F.
    interval: '1m','5m','15m','30m','1h','1d'
    period:   '1d','5d','30d','60d','90d'
    """
    df = yf.download(GOLD, interval=interval, period=period, progress=False)
    if df.empty:
        return df

    # Flatten MultiIndex columns (yfinance returns Ticker-level MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df.index = pd.to_datetime(df.index)
    df = df[["open", "close", "high", "low", "volume"]].dropna()
    return df


def get_macro() -> dict:
    """
    Return latest macro values needed for gold context:
      dxy    — US Dollar Index (inverse to gold)
      us10y  — 10-year Treasury yield (inverse to gold)
      vix    — fear gauge (positive for gold as safe-haven)
    """
    result = {"dxy": 104.0, "us10y": 4.5, "vix": 15.0}
    for key, ticker in [("dxy", DXY), ("us10y", TNX), ("vix", VIX)]:
        try:
            df = yf.download(ticker, period="5d", interval="1d", progress=False)
            if not df.empty:
                val = df["Close"].iloc[-1]
                result[key] = float(val.values[0]) if hasattr(val, "values") else float(val)
        except Exception as e:
            logger.debug(f"Macro fetch {key} failed: {e}")
    return result
