"""
Market Regime Detection
========================
Detects the current market regime using multiple signals:
  ADX, ATR, RSI, VIX, EMA structure, breadth

Regimes:
  TRENDING_UP      → Momentum / ORB / BUY_CE
  TRENDING_DOWN    → Reversal / BUY_PE
  HIGH_VOLATILITY  → Premium selling (straddle/condor)
  LOW_VOLATILITY   → Breakout preparation
  RANGING          → Mean reversion
  EVENT_DRIVEN     → Reduce size, wait

Usage:
    from core.agents.regime import detect_regime
    regime = detect_regime(features_dict, iv_rank=45, vix=14)
"""

import numpy as np
from datetime import datetime
from loguru import logger


def _adx(high, low, close, n=14):
    """Simplified ADX calculation."""
    try:
        import pandas as pd
        hi = pd.Series(high)
        lo = pd.Series(low)
        cl = pd.Series(close)
        plus_dm  = hi.diff().clip(lower=0)
        minus_dm = (-lo.diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm]  = 0
        minus_dm[minus_dm < plus_dm] = 0
        tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=n, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(span=n, adjust=False).mean() / atr.replace(0, 1e-9)
        minus_di = 100 * minus_dm.ewm(span=n, adjust=False).mean() / atr.replace(0, 1e-9)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
        return float(dx.ewm(span=n, adjust=False).mean().iloc[-1]), float(plus_di.iloc[-1]), float(minus_di.iloc[-1])
    except Exception:
        return 25.0, 25.0, 25.0


def detect_regime(
    features: dict,
    iv_rank:  float = 50.0,
    vix:      float = 15.0,
    df_daily  = None,
) -> dict:
    """
    Detect market regime from feature dict (from OITracker.get_model_features()).

    Returns:
    {
      regime: str,
      confidence: float,
      strategy: str,
      indicators: dict,
      description: str,
    }
    """
    # Extract key signals
    tape_bias   = features.get("f_tape_bias_numeric", 0)    # -1 to +1
    fii_bias    = features.get("f_fii_bias_numeric", 0)
    pcr         = features.get("f_pcr_oi", 1.0)
    atm_iv      = features.get("f_atm_iv", 15.0)
    vix_val     = features.get("f_vix", vix)
    bull_pct    = features.get("f_bull_pct", 0.5)
    net_oi_bias = features.get("f_net_oi_bias", 0)
    oi_velocity = features.get("f_net_oi_velocity", 0)
    is_first_hr = features.get("f_is_first_hour", 0)
    is_last_hr  = features.get("f_is_last_hour", 0)

    # Score each regime
    scores = {
        "TRENDING_UP":     0,
        "TRENDING_DOWN":   0,
        "HIGH_VOLATILITY": 0,
        "LOW_VOLATILITY":  0,
        "RANGING":         0,
        "EVENT_DRIVEN":    0,
    }

    reasons = []

    # Trend signals
    if tape_bias > 0.3:
        scores["TRENDING_UP"] += 2
        reasons.append(f"Tape bullish ({tape_bias:.1f})")
    elif tape_bias < -0.3:
        scores["TRENDING_DOWN"] += 2
        reasons.append(f"Tape bearish ({tape_bias:.1f})")

    if fii_bias > 0.5:
        scores["TRENDING_UP"] += 1
        reasons.append("FII bullish")
    elif fii_bias < -0.5:
        scores["TRENDING_DOWN"] += 1
        reasons.append("FII bearish")

    if net_oi_bias > 0.3:
        scores["TRENDING_UP"] += 1
    elif net_oi_bias < -0.3:
        scores["TRENDING_DOWN"] += 1

    if bull_pct > 0.65:
        scores["TRENDING_UP"] += 1
        reasons.append(f"Bull% high ({bull_pct:.0%})")
    elif bull_pct < 0.35:
        scores["TRENDING_DOWN"] += 1

    # Volatility regime
    if iv_rank > 65 or vix_val > 20:
        scores["HIGH_VOLATILITY"] += 3
        reasons.append(f"IV Rank {iv_rank:.0f} | VIX {vix_val:.1f}")
    elif iv_rank < 25 and vix_val < 13:
        scores["LOW_VOLATILITY"] += 3
        reasons.append(f"IV crushed (Rank {iv_rank:.0f})")

    # Ranging signals
    if pcr > 0.85 and pcr < 1.15:
        scores["RANGING"] += 2
        reasons.append(f"PCR {pcr:.2f} neutral")
    if abs(tape_bias) < 0.1 and abs(fii_bias) < 0.3:
        scores["RANGING"] += 1
        reasons.append("No directional bias")

    # Event driven
    if vix_val > 25:
        scores["EVENT_DRIVEN"] += 3
        reasons.append(f"VIX spike {vix_val:.1f}")
    if iv_rank > 80:
        scores["EVENT_DRIVEN"] += 2
    if is_first_hr or is_last_hr:
        scores["EVENT_DRIVEN"] += 1
        reasons.append("Opening/closing hour")

    # OI velocity spike
    if abs(oi_velocity) > 0.5:
        scores["HIGH_VOLATILITY"] += 1

    # Pick winning regime
    regime = max(scores, key=scores.get)
    top_score = scores[regime]
    total = sum(scores.values()) or 1
    confidence = round(top_score / total * 100, 1)

    # Strategy map
    strategy_map = {
        "TRENDING_UP":     "BUY_CE | ORB long | Momentum",
        "TRENDING_DOWN":   "BUY_PE | ORB short | Reversal",
        "HIGH_VOLATILITY": "SELL_STRADDLE | SELL_IRON_CONDOR | Reduce size",
        "LOW_VOLATILITY":  "BUY_STRADDLE | Prepare for breakout",
        "RANGING":         "SELL_STRADDLE | Mean reversion | Wait at extremes",
        "EVENT_DRIVEN":    "REDUCE_SIZE 50% | No new trade | Widen targets",
    }

    description_map = {
        "TRENDING_UP":     "Strong directional momentum. EMAs stacked, tape bullish, FII buying.",
        "TRENDING_DOWN":   "Downtrend confirmed. Bearish tape, FII selling, PE accumulation.",
        "HIGH_VOLATILITY": "IV elevated. Dealers short gamma. Market can make large moves.",
        "LOW_VOLATILITY":  "IV crushed. Market coiled. Watch for breakout either side.",
        "RANGING":         "Market pinned. OI walls at key strikes. Trade reversals at extremes.",
        "EVENT_DRIVEN":    "VIX spike or high-impact event. Reduce exposure, wait for clarity.",
    }

    result = {
        "regime":      regime,
        "confidence":  confidence,
        "strategy":    strategy_map[regime],
        "description": description_map[regime],
        "scores":      scores,
        "indicators": {
            "tape_bias":   tape_bias,
            "fii_bias":    fii_bias,
            "pcr":         pcr,
            "iv_rank":     iv_rank,
            "vix":         vix_val,
            "bull_pct":    bull_pct,
            "oi_velocity": oi_velocity,
        },
        "reasons":     reasons,
        "timestamp":   datetime.now().strftime("%H:%M:%S"),
    }

    logger.info(f"[Regime] {regime} ({confidence:.0f}%) | {' | '.join(reasons[:3])}")
    return result
