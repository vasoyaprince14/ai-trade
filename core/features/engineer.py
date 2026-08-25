"""
Feature Engineering
====================
Builds 150+ features for ML models from:
  - OHLCV (price, volume, returns)
  - Technical indicators (EMA, RSI, MACD, BB, ATR, ADX, VWAP)
  - Options data (OI, IV, PCR, max pain, Greeks)
  - Order flow (OI change velocity, IV skew, put/call pressure)
  - Market regime (VIX, breadth)
  - News sentiment
  - Time features (hour, day, week, expiry proximity)

Inspired by Qlib's feature engineering pipeline.
Output is ML-ready feature vectors with labels.
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger


class FeatureEngineer:
    """
    Builds feature matrix for ML training and real-time prediction.
    Uses 150+ features across price, options, sentiment, time.
    """

    def __init__(self, symbol: str = "NIFTY"):
        self.symbol = symbol

    # ── Master feature builder ─────────────────────────────────────────────────

    def build_features(
        self,
        ohlcv: pd.DataFrame,
        oi_df: Optional[pd.DataFrame] = None,
        sentiment_score: float = 0.0,
        vix: float = 0.0,
        pcr: float = 1.0,
        max_pain: float = 0.0,
        institutional: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Build full feature DataFrame from all sources.
        Returns ready-to-use feature matrix.
        """
        if ohlcv.empty or len(ohlcv) < 30:
            return pd.DataFrame()

        df = ohlcv.copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # 1. Price features
        df = self._price_features(df)

        # 2. Technical indicators
        df = self._technical_features(df)

        # 3. Volume features
        df = self._volume_features(df)

        # 4. Volatility features
        df = self._volatility_features(df)

        # 5. Options features
        if oi_df is not None and not oi_df.empty:
            df = self._options_features(df, oi_df)

        # 6. Market context features
        df["vix"]             = vix
        df["pcr_oi"]          = pcr
        df["sentiment_score"] = sentiment_score
        df["max_pain"]        = max_pain
        if max_pain > 0 and "close" in df.columns:
            df["max_pain_dist"] = (df["close"] - max_pain) / max_pain

        # 7. Institutional features
        if institutional:
            df = self._institutional_features(df, institutional)

        # 8. Time features
        df = self._time_features(df)

        # 9. Target labels
        df = self._add_labels(df)

        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return df.dropna()

    # ── Price features ─────────────────────────────────────────────────────────

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        df["returns_1"]   = c.pct_change(1)
        df["returns_3"]   = c.pct_change(3)
        df["returns_5"]   = c.pct_change(5)
        df["returns_10"]  = c.pct_change(10)
        df["returns_20"]  = c.pct_change(20)
        df["log_return"]  = np.log(c / c.shift(1))
        df["hl_ratio"]    = (df["high"] - df["low"]) / c          # bar range
        df["oc_ratio"]    = (df["close"] - df["open"]) / df["open"]  # bar direction
        df["body_size"]   = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
        df["upper_wick"]  = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-9)
        df["lower_wick"]  = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-9)
        # Rolling highs/lows
        for w in [5, 10, 20, 50]:
            df[f"high_{w}d"] = df["high"].rolling(w).max()
            df[f"low_{w}d"]  = df["low"].rolling(w).min()
            df[f"pct_from_high_{w}d"] = (c - df[f"high_{w}d"]) / df[f"high_{w}d"]
            df[f"pct_from_low_{w}d"]  = (c - df[f"low_{w}d"])  / df[f"low_{w}d"]
        return df

    # ── Technical features ─────────────────────────────────────────────────────

    def _technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            import pandas_ta as ta
            c = df["close"]
            h, l = df["high"], df["low"]

            # Trend
            for p in [9, 21, 50, 100, 200]:
                df[f"ema{p}"] = ta.ema(c, length=p)
            for p in [10, 20, 50]:
                df[f"sma{p}"] = ta.sma(c, length=p)

            # EMA ratios
            df["ema9_21_ratio"]   = df["ema9"]   / df["ema21"]
            df["ema21_50_ratio"]  = df["ema21"]  / df["ema50"]
            df["ema50_200_ratio"] = df["ema50"]  / df["ema200"]
            df["price_ema21_ratio"] = c / df["ema21"]
            df["price_ema50_ratio"] = c / df["ema50"]

            # Momentum
            for p in [7, 14, 21]:
                df[f"rsi{p}"] = ta.rsi(c, length=p)
            macd = ta.macd(c)
            if macd is not None:
                df["macd"]        = macd.iloc[:, 0]
                df["macd_signal"] = macd.iloc[:, 1]
                df["macd_hist"]   = macd.iloc[:, 2]
                df["macd_cross"]  = (df["macd"] > df["macd_signal"]).astype(int)

            # Stochastic
            stoch = ta.stoch(h, l, c)
            if stoch is not None:
                df["stoch_k"] = stoch.iloc[:, 0]
                df["stoch_d"] = stoch.iloc[:, 1]

            # ROC
            for p in [5, 10, 20]:
                df[f"roc{p}"] = ta.roc(c, length=p)

            # Volatility
            for p in [7, 14, 21]:
                df[f"atr{p}"] = ta.atr(h, l, c, length=p)
            df["atr_ratio"] = df["atr14"] / c

            # Bollinger Bands
            bb = ta.bbands(c, length=20)
            if bb is not None:
                cols = bb.columns.tolist()
                bbu = [x for x in cols if x.startswith("BBU")][0]
                bbl = [x for x in cols if x.startswith("BBL")][0]
                bbm = [x for x in cols if x.startswith("BBM")][0]
                df["bb_upper"]  = bb[bbu]
                df["bb_lower"]  = bb[bbl]
                df["bb_mid"]    = bb[bbm]
                df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
                df["bb_pos"]    = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

            # ADX
            adx = ta.adx(h, l, c)
            if adx is not None:
                df["adx"] = adx.iloc[:, 0]

            # CCI
            df["cci"] = ta.cci(h, l, c, length=14)

            # Williams %R
            df["willr"] = ta.willr(h, l, c, length=14)

        except Exception as e:
            logger.warning(f"Technical features error: {e}")
        return df

    # ── Volume features ────────────────────────────────────────────────────────

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volume" not in df.columns or df["volume"].sum() == 0:
            return df
        v = df["volume"]
        df["vol_sma10"]     = v.rolling(10).mean()
        df["vol_sma20"]     = v.rolling(20).mean()
        df["vol_ratio"]     = v / df["vol_sma20"].replace(0, 1)
        df["vol_spike"]     = (df["vol_ratio"] > 2).astype(int)
        df["obv"]           = (np.sign(df["close"].diff()) * v).cumsum()
        df["obv_sma10"]     = df["obv"].rolling(10).mean()
        df["obv_trend"]     = (df["obv"] > df["obv_sma10"]).astype(int)
        # VWAP
        df["vwap"]          = (df["close"] * v).cumsum() / v.cumsum().replace(0, 1)
        df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]
        return df

    # ── Volatility features ────────────────────────────────────────────────────

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        for w in [5, 10, 20, 30]:
            df[f"hist_vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)
        df["vol_regime"] = (df["hist_vol_20d"] > df["hist_vol_20d"].rolling(60).mean()).astype(int)
        # Parkinson volatility (uses H/L)
        df["park_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) *
            (np.log(df["high"] / df["low"]) ** 2).rolling(20).mean() * 252
        )
        return df

    # ── Options features ──────────────────────────────────────────────────────

    def _options_features(self, df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
        """Add option chain derived features."""
        if oi_df.empty:
            return df

        ce = oi_df[oi_df["option_type"] == "CE"]
        pe = oi_df[oi_df["option_type"] == "PE"]

        total_ce_oi  = ce["oi"].sum()
        total_pe_oi  = pe["oi"].sum()
        total_ce_chg = ce["oi_change"].sum()
        total_pe_chg = pe["oi_change"].sum()
        atm_iv_ce    = ce.nsmallest(1, "strike")["iv"].mean() if not ce.empty else 0
        atm_iv_pe    = pe.nsmallest(1, "strike")["iv"].mean() if not pe.empty else 0

        # Assign same options values to all rows (snapshot)
        df["oi_pcr"]         = total_pe_oi / (total_ce_oi + 1e-9)
        df["ce_oi_chg"]      = total_ce_chg
        df["pe_oi_chg"]      = total_pe_chg
        df["net_oi_chg"]     = total_pe_chg - total_ce_chg
        df["iv_skew"]        = atm_iv_pe - atm_iv_ce
        df["oi_buildup"]     = (df["net_oi_chg"] > 0).astype(int)  # put writing = bullish
        df["iv_regime"]      = (atm_iv_ce > 20).astype(int)  # high IV regime

        return df

    # ── Institutional features ─────────────────────────────────────────────────

    def _institutional_features(self, df: pd.DataFrame, inst: Dict) -> pd.DataFrame:
        """
        Add FII/DII positioning features from InstitutionalPositionAnalyzer output.
        All are scalar snapshots broadcast across all rows (daily resolution).
        """
        from core.institutional.positioning import REGIME_SCORES

        df["fii_composite"]       = inst.get("composite", 0.0)
        df["fii_futures_bias"]    = inst.get("futures_bias", 0.0)
        df["fii_call_bias"]       = inst.get("call_bias", 0.0)
        df["fii_put_bias"]        = inst.get("put_bias", 0.0)
        df["fii_regime_score"]    = inst.get("regime_score", 0.0)

        # Raw positions (normalised by total to be scale-invariant)
        fut_total = inst.get("fii_fut_long", 0) + inst.get("fii_fut_short", 0) + 1
        df["fii_fut_long_pct"]    = inst.get("fii_fut_long", 0) / fut_total
        df["fii_fut_short_pct"]   = inst.get("fii_fut_short", 0) / fut_total
        df["fii_fut_net_norm"]    = inst.get("fii_fut_net", 0) / fut_total

        call_total = inst.get("fii_call_long", 0) + inst.get("fii_call_short", 0) + 1
        df["fii_call_long_pct"]   = inst.get("fii_call_long", 0) / call_total
        df["fii_call_short_pct"]  = inst.get("fii_call_short", 0) / call_total

        put_total  = inst.get("fii_put_long", 0) + inst.get("fii_put_short", 0) + 1
        df["fii_put_long_pct"]    = inst.get("fii_put_long", 0) / put_total
        df["fii_put_short_pct"]   = inst.get("fii_put_short", 0) / put_total

        # Cash flow (crores, kept as-is; ML will learn the scale)
        df["fii_cash_net"]        = inst.get("fii_cash_net", 0)
        df["dii_cash_net"]        = inst.get("dii_cash_net", 0)
        df["dii_absorption"]      = inst.get("dii_absorption", 0.0)

        # Velocity (day-over-day change)
        vel = inst.get("velocity", {})
        df["fii_composite_delta"] = vel.get("composite_delta", 0.0)
        df["fii_fut_bias_delta"]  = vel.get("futures_bias_delta", 0.0)
        df["fii_call_bias_delta"] = vel.get("call_bias_delta", 0.0)
        df["fii_put_bias_delta"]  = vel.get("put_bias_delta", 0.0)

        # Divergence flags as binary features
        divs = set(inst.get("divergences", []))
        df["div_fut_bull_opt_bear"]  = int("FUTURES_BULLISH_OPTIONS_BEARISH" in divs)
        df["div_fut_bear_opt_bull"]  = int("FUTURES_BEARISH_OPTIONS_BULLISH" in divs)
        df["div_dii_absorbing"]      = int("DII_ABSORBING_FII_SELLING" in divs)
        df["div_premium_selling"]    = int("CALL_SHORT_PUT_SHORT_PREMIUM_SELLING" in divs)

        # One-hot regime (7 classes → 7 binary columns)
        regime = inst.get("regime", "NEUTRAL")
        for r in ["AGGRESSIVE_BULLISH", "BULLISH", "MILD_BULLISH", "NEUTRAL",
                  "MILD_BEARISH", "BEARISH", "AGGRESSIVE_BEARISH"]:
            df[f"regime_{r.lower()}"] = int(regime == r)

        return df

    # ── Time features ─────────────────────────────────────────────────────────

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        df["hour"]       = idx.hour
        df["minute"]     = idx.minute
        df["day_of_week"] = idx.dayofweek   # 0=Mon
        df["day_of_month"] = idx.day
        df["week_of_year"] = idx.isocalendar().week.astype(int)
        df["month"]      = idx.month
        # Distance to key market times
        df["mins_from_open"]  = (idx.hour - 9) * 60 + (idx.minute - 15)
        df["mins_to_close"]   = (15 - idx.hour) * 60 + (30 - idx.minute)
        df["is_first_hour"]   = (df["mins_from_open"] <= 60).astype(int)
        df["is_last_hour"]    = (df["mins_to_close"] <= 60).astype(int)
        df["is_morning_session"] = ((idx.hour >= 9) & (idx.hour < 12)).astype(int)
        return df

    # ── Target labels ─────────────────────────────────────────────────────────

    def _add_labels(self, df: pd.DataFrame,
                    forward_bars: int = 6,
                    threshold: float = 0.003) -> pd.DataFrame:
        """
        Classification labels for ML:
          2 = Strong Up   (>+0.3%)
          1 = Weak Up     (>+0.1%)
          0 = Neutral
         -1 = Weak Down  (<-0.1%)
         -2 = Strong Down (<-0.3%)
        """
        fwd_return = df["close"].shift(-forward_bars) / df["close"] - 1
        df["fwd_return"]   = fwd_return
        df["label_binary"] = (fwd_return > threshold).astype(int)   # 1=up
        df["label_5class"] = pd.cut(
            fwd_return,
            bins=[-np.inf, -0.003, -0.001, 0.001, 0.003, np.inf],
            labels=[-2, -1, 0, 1, 2],
        ).astype(float)
        # Regression target
        df["target_return"] = fwd_return
        return df

    # ── Feature columns (for ML input) ────────────────────────────────────────

    @staticmethod
    def get_feature_cols(df: pd.DataFrame) -> List[str]:
        """Return list of feature columns (exclude OHLCV and labels)."""
        exclude = {"open", "high", "low", "close", "volume", "symbol",
                   "fwd_return", "label_binary", "label_5class", "target_return"}
        return [c for c in df.columns if c not in exclude and df[c].dtype in (float, int, "float64", "int64")]
