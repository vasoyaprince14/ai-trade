"""
Sector Rotation Agent
======================
Tracks money flow across NSE sectors:
  BANK | IT | AUTO | PHARMA | FMCG | METAL | ENERGY | REALTY | PSU | INFRA

For each sector:
  - 1D / 5D / 20D returns
  - Relative strength vs Nifty 50
  - Volume trend
  - EMA momentum

Output:
  - Money IN sectors (outperforming)
  - Money OUT sectors (underperforming)
  - Regime label (Risk-On / Risk-Off / Rotation)

Usage:
    from core.agents.sector import get_sector_rotation
    sr = get_sector_rotation()
"""

import time
import pandas as pd
import yfinance as yf
from loguru import logger

# NSE Sector indices — yfinance tickers
SECTOR_INDICES = {
    "BANK":    "^NSEBANK",
    "IT":      "^CNXIT",
    "PHARMA":  "^CNXPHARMA",
    "AUTO":    "^CNXAUTO",
    "FMCG":    "^CNXFMCG",
    "METAL":   "^CNXMETAL",
    "ENERGY":  "^CNXENERGY",
    "REALTY":  "^CNXREALTY",
    "INFRA":   "^CNXINFRA",
    "MIDCAP":  "^NSEMDCP50",
}

NIFTY_BENCH = "^NSEI"

_cache: dict = {}
_cache_ts: float = 0
CACHE_TTL = 900  # 15 min


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def get_sector_rotation() -> dict:
    """
    Compute sector rotation data.
    Returns dict with money_in, money_out, rotation_regime, sectors table.
    """
    global _cache, _cache_ts
    now = time.time()
    if _cache and now - _cache_ts < CACHE_TTL:
        return _cache

    all_tickers = list(SECTOR_INDICES.values()) + [NIFTY_BENCH]

    try:
        raw = yf.download(all_tickers, period="60d", interval="1d",
                          progress=False, group_by="ticker")
    except Exception as e:
        logger.warning(f"Sector download error: {e}")
        return _empty_sector()

    # Nifty benchmark returns
    nifty_close = _extract_close(raw, NIFTY_BENCH)
    if nifty_close is None or len(nifty_close) < 22:
        return _empty_sector()

    nifty_ret_1d  = float((nifty_close.iloc[-1] - nifty_close.iloc[-2]) / nifty_close.iloc[-2] * 100)
    nifty_ret_5d  = float((nifty_close.iloc[-1] - nifty_close.iloc[-6]) / nifty_close.iloc[-6] * 100)
    nifty_ret_20d = float((nifty_close.iloc[-1] - nifty_close.iloc[-21]) / nifty_close.iloc[-21] * 100)

    sectors = []
    for name, ticker in SECTOR_INDICES.items():
        try:
            close = _extract_close(raw, ticker)
            if close is None or len(close) < 22:
                continue

            ret_1d  = float((close.iloc[-1] - close.iloc[-2])  / close.iloc[-2]  * 100)
            ret_5d  = float((close.iloc[-1] - close.iloc[-6])  / close.iloc[-6]  * 100)
            ret_20d = float((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100)

            rs_1d  = round(ret_1d  - nifty_ret_1d,  2)
            rs_5d  = round(ret_5d  - nifty_ret_5d,  2)
            rs_20d = round(ret_20d - nifty_ret_20d, 2)

            ema9   = float(_ema(close, 9).iloc[-1])
            ema21  = float(_ema(close, 21).iloc[-1])
            momentum = "UP" if ema9 > ema21 else "DOWN"

            # Composite relative strength score
            rs_score = rs_1d * 0.2 + rs_5d * 0.3 + rs_20d * 0.5

            sectors.append({
                "sector":   name,
                "ret_1d":   round(ret_1d, 2),
                "ret_5d":   round(ret_5d, 2),
                "ret_20d":  round(ret_20d, 2),
                "rs_1d":    rs_1d,
                "rs_5d":    rs_5d,
                "rs_20d":   rs_20d,
                "rs_score": round(rs_score, 2),
                "momentum": momentum,
                "price":    round(float(close.iloc[-1]), 2),
            })
        except Exception:
            continue

    if not sectors:
        return _empty_sector()

    df = pd.DataFrame(sectors).sort_values("rs_score", ascending=False)

    money_in  = df[df["rs_score"] > 0.5]["sector"].tolist()[:3]
    money_out = df[df["rs_score"] < -0.5]["sector"].tolist()[-3:]
    money_out.reverse()

    # Rotation regime
    top_sectors = df.head(3)["sector"].tolist()
    cyclicals   = {"BANK", "AUTO", "METAL", "ENERGY", "REALTY"}
    defensives  = {"PHARMA", "FMCG", "IT"}

    top_set = set(top_sectors)
    if len(top_set & cyclicals) >= 2:
        rotation_regime = "RISK_ON"
        regime_desc     = "Cyclicals leading — broad risk appetite"
    elif len(top_set & defensives) >= 2:
        rotation_regime = "RISK_OFF"
        regime_desc     = "Defensives leading — cautious market"
    else:
        rotation_regime = "MIXED"
        regime_desc     = "Mixed sector leadership — no clear rotation theme"

    result = {
        "sectors":         df.to_dict("records"),
        "money_in":        money_in,
        "money_out":       money_out,
        "rotation_regime": rotation_regime,
        "regime_desc":     regime_desc,
        "nifty_ret_1d":    round(nifty_ret_1d, 2),
        "nifty_ret_5d":    round(nifty_ret_5d, 2),
        "nifty_ret_20d":   round(nifty_ret_20d, 2),
        "top_sectors":     top_sectors,
        "timestamp":       pd.Timestamp.now().strftime("%H:%M"),
    }

    _cache    = result
    _cache_ts = now
    logger.info(f"[Sector] {rotation_regime} | IN: {money_in} | OUT: {money_out}")
    return result


def _extract_close(raw, ticker):
    """Extract close series for a ticker from multi-ticker download."""
    try:
        if ticker in raw.columns.get_level_values(0):
            s = raw[ticker]["Close"].dropna()
        else:
            return None
        return s
    except Exception:
        return None


def _empty_sector() -> dict:
    return {
        "sectors": [], "money_in": [], "money_out": [],
        "rotation_regime": "UNKNOWN", "regime_desc": "No data",
        "nifty_ret_1d": 0, "nifty_ret_5d": 0, "nifty_ret_20d": 0,
        "top_sectors": [], "timestamp": "",
    }
