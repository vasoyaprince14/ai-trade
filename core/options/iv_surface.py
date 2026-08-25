"""
IV Surface Builder
===================
Builds the full Implied Volatility surface for Nifty options:
  - IV Smile     : IV vs Strike (current expiry)
  - IV Skew      : Put premium over Call at same moneyness
  - IV Rank      : (Current - 1Y Low) / (1Y High - 1Y Low) × 100
  - IV Percentile: % of days in past year IV was lower
  - Term Structure: Historical vol at different lookback windows (proxy)

Usage:
    from core.options.iv_surface import IVSurface
    ivs = IVSurface()
    smile   = ivs.build_smile(option_df, spot=24334)
    rank    = ivs.get_iv_rank()
    summary = ivs.full_summary(option_df, spot)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from datetime import datetime, timedelta

VIX_TICKER = "^INDIAVIX"   # India VIX = Nifty 30-day expected vol


class IVSurface:

    def __init__(self):
        self._vix_cache = None
        self._vix_cache_ts = 0

    # ── IV Smile ──────────────────────────────────────────────────────────────

    def build_smile(self, df: pd.DataFrame, spot: float) -> pd.DataFrame:
        """
        Build IV smile from option chain DataFrame.
        Returns DataFrame: strike, ce_iv, pe_iv, avg_iv, moneyness, skew
        """
        if df is None or df.empty:
            return pd.DataFrame()

        results = []
        for strike in sorted(df["strike"].unique()):
            ce = df[(df["strike"] == strike) & (df["option_type"] == "CE")]
            pe = df[(df["strike"] == strike) & (df["option_type"] == "PE")]

            ce_iv = float(ce["iv"].values[0]) if len(ce) and float(ce["iv"].values[0]) > 0 else np.nan
            pe_iv = float(pe["iv"].values[0]) if len(pe) and float(pe["iv"].values[0]) > 0 else np.nan

            # Fill missing with the other side
            if np.isnan(ce_iv) and not np.isnan(pe_iv):
                ce_iv = pe_iv
            elif np.isnan(pe_iv) and not np.isnan(ce_iv):
                pe_iv = ce_iv

            if np.isnan(ce_iv) and np.isnan(pe_iv):
                continue

            avg_iv    = np.nanmean([ce_iv, pe_iv])
            moneyness = round((strike - spot) / spot * 100, 2)   # % OTM/ITM
            skew      = round(pe_iv - ce_iv, 2) if not (np.isnan(ce_iv) or np.isnan(pe_iv)) else 0

            results.append({
                "strike":    int(strike),
                "moneyness": moneyness,
                "ce_iv":     round(ce_iv, 2) if not np.isnan(ce_iv) else None,
                "pe_iv":     round(pe_iv, 2) if not np.isnan(pe_iv) else None,
                "avg_iv":    round(avg_iv, 2),
                "skew":      skew,       # positive = puts richer than calls
            })

        return pd.DataFrame(results)

    # ── ATM IV and Skew Summary ────────────────────────────────────────────────

    def atm_summary(self, smile_df: pd.DataFrame, spot: float) -> dict:
        """Extract ATM IV, 5% OTM put/call IV for skew."""
        if smile_df is None or smile_df.empty:
            return {}

        # ATM = closest strike to spot
        smile_df = smile_df.copy()
        smile_df["dist"] = (smile_df["strike"] - spot).abs()
        atm = smile_df.loc[smile_df["dist"].idxmin()]

        # 2% OTM put
        otm_put_mask  = smile_df["moneyness"] <= -1.5
        otm_call_mask = smile_df["moneyness"] >= 1.5

        otm_put_iv  = float(smile_df[otm_put_mask]["pe_iv"].dropna().median()) if otm_put_mask.any() else None
        otm_call_iv = float(smile_df[otm_call_mask]["ce_iv"].dropna().median()) if otm_call_mask.any() else None

        atm_iv = float(atm["avg_iv"])
        risk_reversal = round(otm_put_iv - otm_call_iv, 2) if otm_put_iv and otm_call_iv else 0

        return {
            "atm_strike":      int(atm["strike"]),
            "atm_iv":          atm_iv,
            "atm_ce_iv":       atm.get("ce_iv"),
            "atm_pe_iv":       atm.get("pe_iv"),
            "atm_skew":        float(atm.get("skew", 0)),
            "otm_put_iv":      otm_put_iv,
            "otm_call_iv":     otm_call_iv,
            "risk_reversal":   risk_reversal,   # + = put vol > call vol (bearish fear)
            "smile_shape":     "SMIRK" if risk_reversal > 1 else "FLAT" if abs(risk_reversal) < 0.5 else "SMILE",
        }

    # ── IV Rank and Percentile ────────────────────────────────────────────────

    def get_iv_rank(self) -> dict:
        """
        India VIX-based IV Rank and IV Percentile.
        IV Rank      = (current - 52w low) / (52w high - 52w low) × 100
        IV Percentile = % of past year where VIX was LOWER than today
        """
        import time
        now = time.time()
        if self._vix_cache and now - self._vix_cache_ts < 3600:
            return self._vix_cache

        try:
            df = yf.download(VIX_TICKER, period="1y", interval="1d", progress=False)
            if df.empty:
                return self._vix_fallback()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            close = df["close"].dropna()
            current   = float(close.iloc[-1])
            high_52w  = float(close.max())
            low_52w   = float(close.min())

            iv_rank  = round((current - low_52w) / (high_52w - low_52w) * 100, 1) if high_52w != low_52w else 50
            iv_pct   = round((close < current).sum() / len(close) * 100, 1)

            # Term structure via historical volatility at different windows
            returns  = close.pct_change().dropna()
            hv7   = float(returns.tail(7).std()  * np.sqrt(252) * 100)
            hv14  = float(returns.tail(14).std() * np.sqrt(252) * 100)
            hv30  = float(returns.tail(30).std() * np.sqrt(252) * 100)
            hv90  = float(returns.tail(90).std() * np.sqrt(252) * 100)

            regime = (
                "HIGH_VOL"   if iv_rank > 70 else
                "LOW_VOL"    if iv_rank < 25 else
                "NORMAL_VOL"
            )

            result = {
                "vix_current":  round(current, 2),
                "vix_52w_high": round(high_52w, 2),
                "vix_52w_low":  round(low_52w, 2),
                "iv_rank":      iv_rank,
                "iv_percentile":iv_pct,
                "regime":       regime,
                "hv_7d":        round(hv7, 2),
                "hv_14d":       round(hv14, 2),
                "hv_30d":       round(hv30, 2),
                "hv_90d":       round(hv90, 2),
                "term_slope":   "CONTANGO" if hv7 < hv30 else "BACKWARDATION",
            }

            self._vix_cache    = result
            self._vix_cache_ts = now
            return result

        except Exception as e:
            logger.warning(f"IV rank fetch error: {e}")
            return self._vix_fallback()

    def _vix_fallback(self) -> dict:
        return {
            "vix_current": 15.0, "vix_52w_high": 22.0, "vix_52w_low": 10.0,
            "iv_rank": 50.0, "iv_percentile": 50.0, "regime": "NORMAL_VOL",
            "hv_7d": 12.0, "hv_14d": 13.0, "hv_30d": 14.0, "hv_90d": 15.0,
            "term_slope": "CONTANGO",
        }

    # ── Full Summary ──────────────────────────────────────────────────────────

    def full_summary(self, df: pd.DataFrame, spot: float) -> dict:
        """Run everything, return combined dict for dashboard."""
        smile   = self.build_smile(df, spot)
        atm     = self.atm_summary(smile, spot)
        iv_rank = self.get_iv_rank()

        # Strategy suggestion based on IV regime
        iv_r = iv_rank.get("iv_rank", 50)
        atm_iv = atm.get("atm_iv", 15)
        if iv_r > 70:
            strategy_hint = "IV HIGH → Sell premium (straddle / iron condor)"
        elif iv_r < 25:
            strategy_hint = "IV LOW → Buy options (straddle / directional)"
        else:
            strategy_hint = "IV NORMAL → Direction-based CE/PE buying"

        return {
            "smile":         smile,
            "atm":           atm,
            "iv_rank":       iv_rank,
            "strategy_hint": strategy_hint,
        }


# Singleton
_instance = None

def get_iv_surface() -> IVSurface:
    global _instance
    if _instance is None:
        _instance = IVSurface()
    return _instance
