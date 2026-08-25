"""
Order Flow & OI Tracker
=======================
Integrates:
  - vendors/ai-trader/strategy/options_flow_detector.py  (FlowSignal detection)
  - vendors/nifty-flow-analysis/ (OI divergence engine)
  - Our NSE scraper for live data

Provides:
  - Real-time OI change tracking per strike
  - PCR calculation and trend
  - Max Pain strike
  - Institutional flow signals (Long Build-Up, Short Covering, etc.)
  - Support/Resistance from OI concentration
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from collections import deque

import pandas as pd
import numpy as np
from loguru import logger

# Add vendors to path
ROOT_DIR = Path(__file__).parent.parent.parent
VENDOR_DIR = ROOT_DIR / "vendors"
# Keep our root first so our config/ resolves correctly
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.data.nse_scraper import get_scraper, NSEScraper
from core.data.nse_participant import get_participant_data
from core.order_flow.tape_reader import TapeReader
from config.settings import config


OI_CFG = config.get("order_flow", {})
PCR_BULL = OI_CFG.get("pcr_bull_threshold", 0.7)
PCR_BEAR = OI_CFG.get("pcr_bear_threshold", 1.3)
STRIKES_RANGE = OI_CFG.get("strikes_range", 10)
OI_CHANGE_THRESHOLD = OI_CFG.get("oi_change_threshold", 0.05)


class OITracker:
    """
    Real-time Open Interest tracker for Nifty/BankNifty options.
    Wraps the OptionsFlowDetector from ai-trader plus our own NSE scraper.
    """

    def __init__(self, symbol: str = "NIFTY"):
        self.symbol = symbol
        self.scraper: NSEScraper = get_scraper()
        self._pcr_history: deque = deque(maxlen=50)   # last 50 snapshots
        self._oi_history: deque = deque(maxlen=100)
        self._last_snapshot: Optional[pd.DataFrame] = None
        self._last_spot: float = 0

        # Tape reader — institutional order flow
        self.tape_reader = TapeReader(symbol)
        # FII/DII participant data
        self._participant = get_participant_data()

        # Try importing the ai-trader flow detector
        self._flow_detector = None
        try:
            ai_trader_path = str(VENDOR_DIR / "ai-trader")
            if ai_trader_path not in sys.path:
                sys.path.append(ai_trader_path)
            from strategy.options_flow_detector import OptionsFlowDetector
            self._flow_detector = OptionsFlowDetector()
            logger.info("Using ai-trader OptionsFlowDetector")
        except (ImportError, Exception):
            logger.info("Using built-in flow detector")

    # ---- Core Snapshot -----------------------------------------------

    def snapshot(self, expiry: Optional[str] = None) -> Dict:
        """
        Fetch current option chain and compute all metrics.
        Returns a comprehensive market state dict.
        """
        records = self.scraper.parse_option_chain(
            self.symbol, expiry, STRIKES_RANGE
        )
        if not records:
            return {}

        df = pd.DataFrame(records)
        self._last_snapshot = df
        spot = df["underlying"].iloc[0] if "underlying" in df.columns else 0
        self._last_spot = spot

        # PCR
        pcr_data = self._calc_pcr(df)
        self._pcr_history.append(pcr_data)

        # OI concentration (support/resistance)
        sr_levels = self._calc_support_resistance(df, spot)

        # Max Pain
        max_pain = self.scraper.get_max_pain(self.symbol, expiry)

        # Flow signal
        flow = self._detect_flow(df, spot)

        # IV data
        iv_data = self._calc_iv_metrics(df)

        result = {
            "symbol": self.symbol,
            "timestamp": datetime.now(),
            "spot": spot,
            "expiry": expiry or (records[0].get("expiry") if records else ""),
            "pcr_oi": pcr_data["pcr_oi"],
            "pcr_vol": pcr_data["pcr_vol"],
            "pcr_signal": self._interpret_pcr(pcr_data["pcr_oi"]),
            "max_pain": max_pain,
            "support_strikes": sr_levels["support"],
            "resistance_strikes": sr_levels["resistance"],
            "flow_signal": flow.get("signal", "NEUTRAL"),
            "flow_score": flow.get("score", 0.0),
            "flow_details": flow.get("details", {}),
            "atm_iv": iv_data.get("atm_iv", 0),
            "iv_skew": iv_data.get("skew", 0),
            "total_call_oi": pcr_data["total_call_oi"],
            "total_put_oi": pcr_data["total_put_oi"],
            "df": df,
        }
        return result

    # ---- PCR -----------------------------------------------------------

    def _calc_pcr(self, df: pd.DataFrame) -> Dict:
        call_df = df[df["option_type"] == "CE"]
        put_df = df[df["option_type"] == "PE"]

        call_oi = call_df["oi"].sum()
        put_oi = put_df["oi"].sum()
        call_vol = call_df["volume"].sum()
        put_vol = put_df["volume"].sum()

        return {
            "pcr_oi": round(put_oi / call_oi, 3) if call_oi > 0 else 0,
            "pcr_vol": round(put_vol / call_vol, 3) if call_vol > 0 else 0,
            "total_call_oi": int(call_oi),
            "total_put_oi": int(put_oi),
            "timestamp": datetime.now(),
        }

    def _interpret_pcr(self, pcr: float) -> str:
        if pcr < PCR_BULL:
            return "BEARISH"    # Too many calls = market topped
        elif pcr > PCR_BEAR:
            return "BULLISH"    # Too many puts = market bottomed
        elif 0.8 <= pcr <= 1.1:
            return "NEUTRAL"
        elif pcr < 1.0:
            return "SLIGHTLY_BEARISH"
        else:
            return "SLIGHTLY_BULLISH"

    def get_pcr_trend(self) -> str:
        """Is PCR rising (bearish) or falling (bullish)?"""
        if len(self._pcr_history) < 3:
            return "NEUTRAL"
        recent = [p["pcr_oi"] for p in list(self._pcr_history)[-5:]]
        if recent[-1] > recent[0] * 1.05:
            return "RISING"   # bearish
        elif recent[-1] < recent[0] * 0.95:
            return "FALLING"  # bullish
        return "STABLE"

    # ---- Support / Resistance from OI ---------------------------------

    def _calc_support_resistance(self, df: pd.DataFrame, spot: float) -> Dict:
        """
        Support = strikes with highest PUT OI (put writers defend these)
        Resistance = strikes with highest CALL OI (call writers defend these)
        """
        call_df = df[df["option_type"] == "CE"].copy()
        put_df = df[df["option_type"] == "PE"].copy()

        # Below spot = support candidates; above spot = resistance
        support = put_df[put_df["strike"] <= spot].nlargest(3, "oi")["strike"].tolist()
        resistance = call_df[call_df["strike"] >= spot].nlargest(3, "oi")["strike"].tolist()

        return {
            "support": sorted(support, reverse=True),
            "resistance": sorted(resistance),
        }

    # ---- Institutional Flow Detection ---------------------------------

    def _detect_flow(self, df: pd.DataFrame, spot: float) -> Dict:
        """
        Detect institutional activity using:
        - OI buildup patterns (Long Build Up, Short Covering, etc.)
        - Volume spikes relative to average
        - ATM vs OTM OI distribution

        Delegates to ai-trader OptionsFlowDetector if available.
        """
        if self._flow_detector and not df.empty:
            try:
                result = self._flow_detector.analyze(df, spot)
                return {
                    "signal": result.signal.value,
                    "score": result.score,
                    "pcr": result.pcr,
                    "max_oi_strike": result.max_oi_strike,
                    "details": result.details,
                }
            except Exception as e:
                logger.debug(f"FlowDetector error: {e}")

        # Built-in fallback
        return self._builtin_flow_detect(df, spot)

    def _builtin_flow_detect(self, df: pd.DataFrame, spot: float) -> Dict:
        """
        Simple but effective OI flow detection:
        Long Build-Up  : price↑ + OI↑ → Bullish
        Short Covering : price↑ + OI↓ → Bullish (temporary)
        Long Unwinding : price↓ + OI↓ → Bearish
        Short Build-Up : price↓ + OI↑ → Bearish
        """
        if df.empty:
            return {"signal": "NEUTRAL", "score": 0.0, "details": {}}

        call_oi_change = df[df["option_type"] == "CE"]["oi_change"].sum()
        put_oi_change = df[df["option_type"] == "PE"]["oi_change"].sum()
        net_oi_change = call_oi_change - put_oi_change  # positive = more call writers

        # High OI buildup threshold
        total_oi = df["oi"].sum()
        significant = abs(net_oi_change) / (total_oi + 1) > OI_CHANGE_THRESHOLD

        if significant:
            if net_oi_change < 0:  # More put writing → support / bullish
                signal = "LONG_BUILD_UP"
                score = min(abs(net_oi_change) / (total_oi + 1) * 10, 1.0)
            else:  # More call writing → resistance / bearish
                signal = "SHORT_BUILD_UP"
                score = min(net_oi_change / (total_oi + 1) * 10, 1.0)
        else:
            signal = "NEUTRAL"
            score = 0.0

        # Check for gamma pinning (very high OI at one strike)
        oi_by_strike = df.groupby("strike")["oi"].sum()
        if not oi_by_strike.empty:
            max_oi_strike = oi_by_strike.idxmax()
            max_oi = oi_by_strike.max()
            if max_oi > oi_by_strike.mean() * 3:
                signal = "GAMMA_PINNING"
                score = 0.8

        return {
            "signal": signal,
            "score": round(score, 3),
            "details": {
                "call_oi_change": call_oi_change,
                "put_oi_change": put_oi_change,
                "net_oi_change": net_oi_change,
            }
        }

    # ---- IV Metrics --------------------------------------------------

    def _calc_iv_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate ATM IV and IV skew."""
        if df.empty:
            return {}
        spot = df["underlying"].iloc[0] if "underlying" in df.columns else 0
        atm = round(spot / 50) * 50

        atm_call = df[(df["strike"] == atm) & (df["option_type"] == "CE")]["iv"]
        atm_put = df[(df["strike"] == atm) & (df["option_type"] == "PE")]["iv"]

        atm_iv = 0
        if not atm_call.empty and not atm_put.empty:
            atm_iv = (atm_call.iloc[0] + atm_put.iloc[0]) / 2

        # Skew: OTM Put IV - OTM Call IV (positive = put skew = fear)
        otm_put_strike = atm - 200
        otm_call_strike = atm + 200
        otm_put_iv = df[(df["strike"] == otm_put_strike) & (df["option_type"] == "PE")]["iv"]
        otm_call_iv = df[(df["strike"] == otm_call_strike) & (df["option_type"] == "CE")]["iv"]
        skew = 0
        if not otm_put_iv.empty and not otm_call_iv.empty:
            skew = otm_put_iv.iloc[0] - otm_call_iv.iloc[0]

        return {
            "atm_iv": round(atm_iv, 2),
            "skew": round(skew, 2),
        }

    # ---- Tape + Institutional Flow ----------------------------------------

    def tick_tape(self, expiry: Optional[str] = None):
        """
        Poll option chain → detect institutional events via TapeReader.
        Call every 30-60 seconds during market hours.
        Returns list of new TapeEvents.
        """
        return self.tape_reader.tick(expiry)

    def get_tape_summary(self) -> Dict:
        """Full tape reading summary + FII/DII positioning."""
        tape = self.tape_reader.get_flow_summary()
        participant = self._participant.get_full_picture()
        return {
            "tape": tape,
            "participant": participant,
            "combined_bias": self._merge_bias(
                tape.get("bias", "NEUTRAL"),
                participant.get("smart_money_bias", "NEUTRAL"),
            ),
        }

    def get_model_features(self, expiry: Optional[str] = None) -> Dict:
        """
        Extract 100+ features for ML model decision making.
        Combines: tape reader + option chain + FII/DII data.
        """
        # Tape features (100+ features)
        features = self.tape_reader.extract_features()

        # Add VIX
        vix = self.scraper.get_vix()
        features["f_vix"] = vix or 0

        # Add FII/DII
        participant = self._participant.get_participant_summary()
        fii = participant.get("participants", {}).get("FII", {})
        dii = participant.get("participants", {}).get("DII", {})
        pro = participant.get("participants", {}).get("PRO", {})
        features["f_fii_net_futures"]  = fii.get("net_futures", 0)
        features["f_fii_net_calls"]    = fii.get("net_calls",   0)
        features["f_fii_net_puts"]     = fii.get("net_puts",    0)
        features["f_fii_bias_score"]   = fii.get("bias_score",  0)
        features["f_dii_net_futures"]  = dii.get("net_futures", 0)
        features["f_dii_net_calls"]    = dii.get("net_calls",   0)
        features["f_dii_net_puts"]     = dii.get("net_puts",    0)
        features["f_dii_bias_score"]   = dii.get("bias_score",  0)
        features["f_pro_net_futures"]  = pro.get("net_futures", 0)
        features["f_pro_net_calls"]    = pro.get("net_calls",   0)
        features["f_pro_net_puts"]     = pro.get("net_puts",    0)

        bias_map = {"STRONGLY_BULLISH": 2, "BULLISH": 1, "NEUTRAL": 0,
                    "BEARISH": -1, "STRONGLY_BEARISH": -2}
        features["f_fii_bias_numeric"] = bias_map.get(fii.get("bias", "NEUTRAL"), 0)
        features["f_dii_bias_numeric"] = bias_map.get(dii.get("bias", "NEUTRAL"), 0)
        features["f_smart_money_bias"] = bias_map.get(
            participant.get("combined_bias", "NEUTRAL"), 0
        )

        return features

    def _merge_bias(self, tape_bias: str, smart_bias: str) -> str:
        """Combine tape bias + smart money bias into single view."""
        score_map = {
            "STRONGLY_BULLISH": 2, "BULLISH": 1, "NEUTRAL": 0,
            "BEARISH": -1, "STRONGLY_BEARISH": -2,
        }
        score = score_map.get(tape_bias, 0) + score_map.get(smart_bias, 0)
        if score >= 3:
            return "STRONGLY_BULLISH"
        elif score >= 1:
            return "BULLISH"
        elif score <= -3:
            return "STRONGLY_BEARISH"
        elif score <= -1:
            return "BEARISH"
        return "NEUTRAL"

    # ---- Summary for Dashboard ----------------------------------------

    def get_market_summary(self) -> Dict:
        """Quick summary for dashboard display."""
        snap = self.snapshot()
        if not snap:
            return {}
        vix = self.scraper.get_vix()
        tape = self.tape_reader.get_flow_summary()
        return {
            **snap,
            "vix": vix,
            "pcr_trend": self.get_pcr_trend(),
            "market_open": self.scraper.is_market_open(),
            # Institutional tape data
            "tape_bias":     tape.get("bias", "NEUTRAL"),
            "tape_events":   tape.get("total_events", 0),
            "hot_strikes":   tape.get("hot_strikes", []),
            "big_players":   tape.get("big_players", []),
            "recent_tape":   tape.get("recent_tape", []),
        }
