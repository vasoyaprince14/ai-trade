"""
GEX — Gamma Exposure Calculator
=================================
GEX = Gamma × OI × ContractSize × Spot²  (per $1 move)

For Nifty options:
  Net GEX (per strike) = CE_GEX - PE_GEX
  +GEX zone → dealers long gamma → market mean-reverts (pinned)
  -GEX zone → dealers short gamma → market accelerates (trending)

Gamma Flip = strike where Net GEX crosses zero.
Above gamma flip → +GEX (stable)
Below gamma flip → -GEX (volatile)

Usage:
    from core.options.gex import compute_gex, gex_dashboard
    result = compute_gex(option_df, spot=24334, expiry_days=1)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from loguru import logger

NIFTY_LOT = 25      # Nifty lot size
BANKNIFTY_LOT = 15
RISK_FREE = 0.065   # Indian risk-free rate


def _bsm_gamma(S, K, T, r, sigma):
    """BSM gamma for a European option (same for CE and PE)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def compute_gex(df: pd.DataFrame, spot: float, expiry_days: int = 1,
                lot_size: int = NIFTY_LOT) -> dict:
    """
    Compute Net GEX, Gamma Flip, and dealer bias from option chain DataFrame.

    df must have columns: strike, option_type, oi, iv  (standard OI tracker output)

    Returns:
    {
      by_strike: DataFrame (strike, ce_gex, pe_gex, net_gex)
      total_ce_gex, total_pe_gex, net_total_gex
      gamma_flip: float (strike where net GEX crosses 0)
      dealer_bias: "LONG" | "SHORT" | "NEUTRAL"
      largest_positive_gex_strike, largest_negative_gex_strike
      expected_volatility: "HIGH" | "LOW" | "MODERATE"
      summary: str
    }
    """
    if df is None or df.empty:
        return _empty_gex(spot)

    T = max(expiry_days, 0.5) / 365.0

    results = []
    for strike in sorted(df["strike"].unique()):
        row_ce = df[(df["strike"] == strike) & (df["option_type"] == "CE")]
        row_pe = df[(df["strike"] == strike) & (df["option_type"] == "PE")]

        ce_oi = int(row_ce["oi"].values[0]) if len(row_ce) else 0
        pe_oi = int(row_pe["oi"].values[0]) if len(row_pe) else 0
        ce_iv = float(row_ce["iv"].values[0]) if len(row_ce) else 0.0
        pe_iv = float(row_pe["iv"].values[0]) if len(row_pe) else 0.0

        # Use ATM IV as fallback for zero-IV entries
        atm_iv_fallback = 0.12
        if ce_iv <= 0:
            ce_iv = pe_iv if pe_iv > 0 else atm_iv_fallback
        if pe_iv <= 0:
            pe_iv = ce_iv if ce_iv > 0 else atm_iv_fallback

        ce_gamma = _bsm_gamma(spot, strike, T, RISK_FREE, ce_iv / 100)
        pe_gamma = _bsm_gamma(spot, strike, T, RISK_FREE, pe_iv / 100)

        # GEX = gamma × OI × lot_size × spot²
        ce_gex = ce_gamma * ce_oi * lot_size * (spot ** 2) / 1e9  # in ₹ billions
        pe_gex = pe_gamma * pe_oi * lot_size * (spot ** 2) / 1e9

        results.append({
            "strike":  int(strike),
            "ce_oi":   ce_oi, "pe_oi": pe_oi,
            "ce_iv":   round(ce_iv, 2), "pe_iv": round(pe_iv, 2),
            "ce_gex":  round(ce_gex, 4),
            "pe_gex":  round(pe_gex, 4),
            "net_gex": round(ce_gex - pe_gex, 4),
        })

    gex_df = pd.DataFrame(results).sort_values("strike")

    total_ce  = float(gex_df["ce_gex"].sum())
    total_pe  = float(gex_df["pe_gex"].sum())
    net_total = round(total_ce - total_pe, 4)

    # Gamma flip: strike closest to where net_gex changes sign
    gamma_flip = _find_gamma_flip(gex_df, spot)

    # Dealer bias
    if net_total > 0.5:
        dealer_bias = "LONG_GAMMA"   # market will mean-revert
    elif net_total < -0.5:
        dealer_bias = "SHORT_GAMMA"  # market will accelerate
    else:
        dealer_bias = "NEUTRAL"

    expected_vol = "HIGH" if dealer_bias == "SHORT_GAMMA" else "LOW" if dealer_bias == "LONG_GAMMA" else "MODERATE"

    # Key strikes
    largest_pos = int(gex_df.loc[gex_df["net_gex"].idxmax(), "strike"])
    largest_neg = int(gex_df.loc[gex_df["net_gex"].idxmin(), "strike"])

    summary = (
        f"Net GEX: {net_total:+.2f}B | Dealer: {dealer_bias} | "
        f"Gamma Flip: {gamma_flip:.0f} | Vol: {expected_vol}"
    )

    logger.info(f"[GEX] {summary}")

    return {
        "by_strike":                  gex_df,
        "total_ce_gex":               round(total_ce, 4),
        "total_pe_gex":               round(total_pe, 4),
        "net_total_gex":              net_total,
        "gamma_flip":                 gamma_flip,
        "dealer_bias":                dealer_bias,
        "largest_positive_gex_strike": largest_pos,
        "largest_negative_gex_strike": largest_neg,
        "expected_volatility":        expected_vol,
        "summary":                    summary,
        "spot":                       spot,
    }


def _find_gamma_flip(gex_df: pd.DataFrame, spot: float) -> float:
    """Find the strike closest to zero net GEX — the gamma flip level."""
    df = gex_df.copy().sort_values("strike")
    # Look for sign change
    for i in range(len(df) - 1):
        n1 = df.iloc[i]["net_gex"]
        n2 = df.iloc[i + 1]["net_gex"]
        if n1 * n2 < 0:
            # Linear interpolation
            k1, k2 = df.iloc[i]["strike"], df.iloc[i + 1]["strike"]
            flip = k1 + (k2 - k1) * abs(n1) / (abs(n1) + abs(n2))
            return round(flip, 0)
    # No sign change: return strike with minimum abs net GEX
    idx = df["net_gex"].abs().idxmin()
    return float(df.loc[idx, "strike"])


def _empty_gex(spot: float) -> dict:
    return {
        "by_strike": pd.DataFrame(),
        "total_ce_gex": 0, "total_pe_gex": 0, "net_total_gex": 0,
        "gamma_flip": spot, "dealer_bias": "NEUTRAL",
        "largest_positive_gex_strike": spot, "largest_negative_gex_strike": spot,
        "expected_volatility": "MODERATE", "summary": "No data", "spot": spot,
    }
