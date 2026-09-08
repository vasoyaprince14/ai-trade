"""
XAUUSD Live Data Fetcher
=========================
Data sources (best to worst delay):
  1. yahooquery GLD.regularMarketPrice  — realtime tick (~0 delay)
  2. yahooquery GLD history bars        — 1-2 min delay
  3. yfinance GLD 1m bars              — ~1 min delay (fallback)
  4. yfinance GC=F bars                — 10-15 min delay (last resort)

Daily ratio = median(GC=F_daily / GLD_daily) over last 10 days.
Stable — changes < 0.001%/day due to GLD expense ratio.
"""

import time
import pandas as pd
from loguru import logger

GOLD = "GC=F"
GLD  = "GLD"
DXY  = "DX-Y.NYB"
TNX  = "^TNX"
VIX  = "^VIX"

_ratio_cache: dict = {"ratio": 10.93, "ts": 0.0}
_RATIO_TTL = 3600  # recalibrate every hour


def _get_daily_ratio() -> float:
    """GC=F / GLD ratio from settled daily bars. Cached 1 hour."""
    if time.time() - _ratio_cache["ts"] < _RATIO_TTL:
        return _ratio_cache["ratio"]
    try:
        import yfinance as yf
        df_gld = yf.download(GLD,  period="10d", interval="1d", progress=False)
        df_gc  = yf.download(GOLD, period="10d", interval="1d", progress=False)
        if not df_gld.empty and not df_gc.empty:
            def _s(df):
                c = df["Close"]
                return c.squeeze() if hasattr(c, "squeeze") else c
            merged = pd.concat([_s(df_gld).rename("gld"), _s(df_gc).rename("gc")], axis=1).dropna()
            if len(merged) >= 3:
                ratio = float((merged["gc"] / merged["gld"]).median())
                _ratio_cache.update({"ratio": ratio, "ts": time.time()})
                logger.debug(f"[data] Daily ratio: {ratio:.4f}")
                return ratio
    except Exception as e:
        logger.debug(f"[data] Ratio refresh failed: {e}")
    return _ratio_cache["ratio"]


def get_price() -> float:
    """
    Near-realtime gold spot price (USD/troy oz).
    yahooquery GLD.regularMarketPrice × daily ratio — effectively 0 delay.
    """
    ratio = _get_daily_ratio()

    # Primary: yahooquery regularMarketPrice (live tick)
    try:
        from yahooquery import Ticker
        px = Ticker(GLD).price[GLD].get("regularMarketPrice", 0)
        if px and px > 100:
            price = round(float(px) * ratio, 2)
            logger.debug(f"[data] Live price (yahooquery): {px:.4f}×{ratio:.4f}={price:.2f}")
            return price
    except Exception:
        pass

    # Secondary: yfinance GLD 1m
    try:
        import yfinance as yf
        df = yf.download(GLD, period="1d", interval="1m", progress=False)
        if not df.empty:
            val = df["Close"].iloc[-1]
            gld_px = float(val.values[0]) if hasattr(val, "values") else float(val)
            price = round(gld_px * ratio, 2)
            logger.debug(f"[data] Live price (yf GLD): {gld_px:.4f}×{ratio:.4f}={price:.2f}")
            return price
    except Exception:
        pass

    # Last resort: GC=F (stale)
    try:
        import yfinance as yf
        df = yf.download(GOLD, period="1d", interval="1m", progress=False)
        if not df.empty:
            val = df["Close"].iloc[-1]
            return float(val.values[0]) if hasattr(val, "values") else float(val)
    except Exception as e:
        logger.warning(f"[data] get_price all failed: {e}")
    return 0.0


def get_bars(interval: str = "15m", period: str = "5d") -> pd.DataFrame:
    """
    OHLCV bars in gold price (USD/troy oz), 1-2 min delay via yahooquery GLD.
    Falls back to yfinance GC=F (10-15 min delay) if yahooquery fails.
    """
    ratio = _get_daily_ratio()
    period_map = {"1d": "1d", "5d": "5d", "30d": "1mo", "60d": "3mo", "90d": "3mo"}
    yq_period  = period_map.get(period, "1mo")

    # Primary: yahooquery GLD bars × ratio
    try:
        from yahooquery import Ticker
        raw = Ticker(GLD).history(period=yq_period, interval=interval)
        if raw is not None and not raw.empty:
            df = raw.reset_index()
            # drop symbol level from MultiIndex
            if "symbol" in df.columns:
                df = df.drop(columns=["symbol"])
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            if date_col:
                df = df.set_index(date_col)
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "timestamp"
            df.columns = [c.lower() for c in df.columns]
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col].astype(float) * ratio
            if "volume" not in df.columns:
                df["volume"] = 0.0
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            if not df.empty:
                logger.debug(f"[data] Bars: yahooquery GLD ({len(df)}) last={df.index[-1]}")
                return df
    except Exception as e:
        logger.debug(f"[data] yahooquery bars failed: {e}")

    # Fallback: yfinance GC=F
    try:
        import yfinance as yf
        df = yf.download(GOLD, interval=interval, period=period, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            df.index.name = "timestamp"
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            logger.debug(f"[data] Bars: yfinance GC=F (stale, {len(df)})")
            return df
    except Exception as e:
        logger.warning(f"[data] get_bars all failed: {e}")
    return pd.DataFrame()


def get_macro() -> dict:
    """DXY, US 10Y yield, VIX — yahooquery for live values."""
    result = {"dxy": 104.0, "us10y": 4.5, "vix": 15.0}
    keys   = {"dxy": DXY, "us10y": TNX, "vix": VIX}
    try:
        from yahooquery import Ticker
        data = Ticker(list(keys.values())).price
        for k, ticker in keys.items():
            entry = data.get(ticker, {})
            if isinstance(entry, dict):
                px = entry.get("regularMarketPrice", 0)
                if px:
                    result[k] = float(px)
    except Exception:
        import yfinance as yf
        for k, ticker in keys.items():
            try:
                df = yf.download(ticker, period="5d", interval="1d", progress=False)
                if not df.empty:
                    val = df["Close"].iloc[-1]
                    result[k] = float(val.values[0]) if hasattr(val, "values") else float(val)
            except Exception:
                pass
    return result
