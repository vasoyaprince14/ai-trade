"""
NSE Market Breadth Agent
=========================
Tracks:
  - Advances / Declines across Nifty 50 stocks
  - % stocks above EMA20 / EMA50 / EMA200
  - RSI breadth (% stocks with RSI > 50)
  - Volume breadth (% stocks with above-avg volume)
  - Sector breadth
  - Composite Breadth Score (0-100)

Cached 15 minutes — uses yfinance for price data.

Usage:
    from core.agents.breadth import get_breadth
    b = get_breadth()
    print(b["breadth_score"], b["signal"])
"""

import time
import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

# Nifty 50 constituents (all with .NS suffix)
NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "WIPRO.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
    "TATAMOTORS.NS", "SUNPHARMA.NS", "TECHM.NS", "ULTRACEMCO.NS", "ADANIENT.NS",
    "TITAN.NS", "JSWSTEEL.NS", "COALINDIA.NS", "BPCL.NS", "GRASIM.NS",
    "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TATASTEEL.NS",
    "NESTLEIND.NS", "INDUSINDBK.NS", "BRITANNIA.NS", "HINDALCO.NS", "SHREECEM.NS",
    "TATACONSUM.NS", "PIDILITIND.NS", "M&M.NS", "BAJAJ-AUTO.NS", "UPL.NS",
]

_cache: dict = {}
_cache_ts: float = 0
CACHE_TTL = 900   # 15 min


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s, n=14):
    d    = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs   = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def get_breadth(symbols: list = None) -> dict:
    """
    Compute NSE market breadth.
    symbols: list of yfinance tickers (default: Nifty 50)
    Returns comprehensive breadth dict.
    """
    global _cache, _cache_ts
    now = time.time()
    if _cache and now - _cache_ts < CACHE_TTL:
        return _cache

    syms = symbols or NIFTY50

    # Download all in one batch (efficient)
    try:
        raw = yf.download(syms, period="90d", interval="1d",
                          progress=False, group_by="ticker")
    except Exception as e:
        logger.warning(f"Breadth download error: {e}")
        return _empty_breadth()

    advances = declines = unchanged = 0
    above_ema20 = above_ema50 = above_ema200 = 0
    rsi_gt50 = vol_above_avg = 0
    hi52w = lo52w = 0
    processed = 0

    for ticker in syms:
        try:
            if ticker in raw.columns.get_level_values(0):
                df = raw[ticker].dropna()
            elif len(syms) == 1:
                df = raw.dropna()
            else:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            if "close" not in df.columns or len(df) < 50:
                continue

            close  = df["close"]
            volume = df.get("volume", pd.Series([1] * len(df)))

            price      = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            ema20_val  = float(_ema(close, 20).iloc[-1])
            ema50_val  = float(_ema(close, 50).iloc[-1])
            ema200_val = float(_ema(close, 200).iloc[-1]) if len(close) >= 200 else ema50_val
            rsi_val    = float(_rsi(close).iloc[-1])
            vol_now    = float(volume.iloc[-1])
            vol_avg    = float(volume.iloc[-20:].mean())

            if price > prev_price:     advances  += 1
            elif price < prev_price:   declines  += 1
            else:                      unchanged += 1

            if price > ema20_val:      above_ema20  += 1
            if price > ema50_val:      above_ema50  += 1
            if price > ema200_val:     above_ema200 += 1
            if rsi_val > 50:           rsi_gt50     += 1
            if vol_now > vol_avg:      vol_above_avg += 1

            high_52w = float(close.rolling(252).max().iloc[-1])
            low_52w  = float(close.rolling(252).min().iloc[-1])
            if price >= high_52w * 0.99:  hi52w += 1
            if price <= low_52w * 1.01:   lo52w += 1

            processed += 1

        except Exception:
            continue

    if processed == 0:
        return _empty_breadth()

    p = processed
    scores = {
        "adv_dec":    (advances - declines) / p * 50 + 50,
        "ema20":      above_ema20  / p * 100,
        "ema50":      above_ema50  / p * 100,
        "rsi_breadth": rsi_gt50   / p * 100,
        "vol_breadth": vol_above_avg / p * 100,
    }
    breadth_score = round(sum(scores.values()) / len(scores), 1)

    # Signal
    if breadth_score > 70:
        signal = "BULLISH"
        desc   = "Broad market strength — most stocks participating in uptrend"
    elif breadth_score < 35:
        signal = "BEARISH"
        desc   = "Broad market weakness — most stocks in downtrend"
    elif breadth_score > 55:
        signal = "MILDLY_BULLISH"
        desc   = "Moderate breadth — leaning bullish but not unanimous"
    elif breadth_score < 45:
        signal = "MILDLY_BEARISH"
        desc   = "Moderate breadth — leaning bearish"
    else:
        signal = "NEUTRAL"
        desc   = "Mixed breadth — no clear directional edge"

    # Divergence check (price vs breadth)
    result = {
        "breadth_score":   breadth_score,
        "signal":          signal,
        "description":     desc,
        "advances":        advances,
        "declines":        declines,
        "unchanged":       unchanged,
        "adv_dec_ratio":   round(advances / max(declines, 1), 2),
        "above_ema20_pct": round(above_ema20  / p * 100, 1),
        "above_ema50_pct": round(above_ema50  / p * 100, 1),
        "above_ema200_pct":round(above_ema200 / p * 100, 1),
        "rsi_breadth_pct": round(rsi_gt50 / p * 100, 1),
        "vol_breadth_pct": round(vol_above_avg / p * 100, 1),
        "new_52w_high":    hi52w,
        "new_52w_low":     lo52w,
        "stocks_scanned":  processed,
        "component_scores": {k: round(v, 1) for k, v in scores.items()},
    }

    _cache    = result
    _cache_ts = now
    logger.info(f"[Breadth] Score={breadth_score} | {signal} | A/D={advances}/{declines}")
    return result


def _empty_breadth() -> dict:
    return {
        "breadth_score": 50, "signal": "NEUTRAL", "description": "No data",
        "advances": 0, "declines": 0, "unchanged": 0, "adv_dec_ratio": 1,
        "above_ema20_pct": 50, "above_ema50_pct": 50, "above_ema200_pct": 50,
        "rsi_breadth_pct": 50, "vol_breadth_pct": 50,
        "new_52w_high": 0, "new_52w_low": 0, "stocks_scanned": 0,
        "component_scores": {},
    }
