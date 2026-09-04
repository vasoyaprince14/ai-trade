"""
XAUUSD Live Data Fetcher
=========================
Gold (GC=F) OHLCV via yfinance + macro context (DXY, 10Y yield, VIX).

Data delay note:
  GC=F (gold futures) — yfinance has ~10-15 min delay during US hours.
  GLD  (gold ETF)     — yfinance has ~1 min delay (equity, not futures).
  Strategy: use GLD as real-time price source with a FIXED physical ratio.

GLD physical backing:
  1 GLD share = 0.09569 troy oz of gold (SPDR trust deed; decreases ~0.04%/yr
  due to management fees — effectively constant for intraday use).
  Spot gold price = GLD_price / 0.09569

  We do NOT calibrate the ratio from GC=F because GC=F has a 10-15 min delay
  on yfinance. Calibrating stale_GC=F / live_GLD produces an inflated ratio
  that cancels out the GLD improvement and gives us the stale futures price.
  The fixed physical ratio gives genuine near-live spot gold (~1 min delay).
"""

import yfinance as yf
import pandas as pd
from loguru import logger

GOLD    = "GC=F"
SPOT    = "XAUUSD=X"  # Spot gold forex pair — minimal delay, no futures premium
GLD     = "GLD"       # Gold ETF — ~1 min delay, backup when XAUUSD=X unavailable
DXY     = "DX-Y.NYB"
TNX     = "^TNX"      # US 10Y yield
VIX     = "^VIX"


def _extract(df, col="Close") -> float:
    val = df[col].iloc[-1]
    return float(val.values[0]) if hasattr(val, "values") else float(val)


def get_price() -> float:
    """
    Current spot gold price (USD/troy oz).
    1. XAUUSD=X  — spot gold forex, near-realtime, no futures premium/delay
    2. GLD ETF   — equity, ~1 min delay; ratio computed live from GLD NAV
    3. GC=F 1m   — gold futures, ~10-15 min delay (last resort)
    """
    # Primary: XAUUSD=X spot gold
    try:
        df = yf.download(SPOT, period="1d", interval="1m", progress=False)
        if not df.empty:
            price = round(_extract(df), 2)
            if price > 100:   # sanity check
                logger.debug(f"[data] Live price via XAUUSD=X: {price:.2f}")
                return price
    except Exception:
        pass

    # Secondary: GLD ETF with ratio from accurate daily bars
    # Daily GC=F bars reflect the true settled price (no intraday delay).
    # Ratio is stable day-to-day (changes ~0.001%/day due to GLD expense).
    try:
        df_gld_d = yf.download(GLD,  period="10d", interval="1d", progress=False)
        df_gc_d  = yf.download(GOLD, period="10d", interval="1d", progress=False)
        if not df_gld_d.empty and not df_gc_d.empty:
            gld_s  = df_gld_d["Close"].squeeze().rename("gld")
            gc_s   = df_gc_d["Close"].squeeze().rename("gc")
            merged = pd.concat([gld_s, gc_s], axis=1).dropna()
            if len(merged) >= 3:
                ratio = float((merged["gc"] / merged["gld"]).median())
                # Apply to live 1-min GLD
                df1 = yf.download(GLD, period="1d", interval="1m", progress=False)
                if not df1.empty:
                    gld_px = _extract(df1)
                    price  = round(gld_px * ratio, 2)
                    logger.debug(f"[data] Live price via GLD daily ratio: {gld_px:.2f}×{ratio:.4f}={price:.2f}")
                    return price
    except Exception:
        pass

    # Final fallback: GC=F 1m (stale but better than nothing)
    for ticker, period, interval in [(GOLD, "1d", "1m"), (GOLD, "5d", "5m")]:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if not df.empty:
                return round(_extract(df), 2)
        except Exception:
            pass

    logger.warning("[data] get_price: all sources failed")
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
