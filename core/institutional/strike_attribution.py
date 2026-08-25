"""
Strike-Level Institutional Flow Attribution
=============================================
Combines option chain data with FII positioning to infer where
institutional activity is concentrated.

Note: This is an INFERENCE engine, not direct attribution.
We cannot see which individual institution sold a specific strike.
We combine:
  - CE/PE OI change (heavy buildup = institutional activity)
  - Volume relative to OI (absorption vs fresh positioning)
  - IV change (selling pressure suppresses IV)
  - Bid/ask direction
  - FII aggregate bias (confirms or contradicts strike-level signals)

Output: call_walls, put_walls, institutional zones, and a
combined bearish/bullish zone score per strike.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


class StrikeAttributionEngine:
    """
    Infers institutional positioning zones from option chain data.
    """

    def __init__(self, top_n: int = 3):
        """
        Args:
            top_n: Number of top call/put walls to return.
        """
        self.top_n = top_n

    # ── Main entry point ───────────────────────────────────────────────────────

    def analyze(
        self,
        option_chain: List[dict],
        spot: float,
        fii_call_bias: float = 0.0,
        fii_put_bias:  float = 0.0,
        fii_composite: float = 0.0,
    ) -> Dict:
        """
        Full strike-level analysis.

        Args:
            option_chain : List of dicts from NSEScraper.parse_option_chain()
                           Each dict has: strike, option_type, oi, oi_change,
                           volume, ltp, iv, delta, underlying.
            spot         : Current underlying price.
            fii_call_bias: From InstitutionalPositionAnalyzer (-1..+1).
            fii_put_bias : From InstitutionalPositionAnalyzer (-1..+1).
            fii_composite: Overall FII composite bias (-1..+1).

        Returns dict with:
            call_walls, put_walls, institutional_zones,
            gamma_pin_zone, max_pain_zone, strike_scores
        """
        if not option_chain:
            return self._empty_result()

        df = pd.DataFrame(option_chain)
        ce = df[df["option_type"] == "CE"].copy()
        pe = df[df["option_type"] == "PE"].copy()

        if ce.empty or pe.empty:
            return self._empty_result()

        call_walls = self._find_walls(ce, spot, side="CALL")
        put_walls  = self._find_walls(pe, spot, side="PUT")

        # Strike-level institutional score
        strike_scores = self._score_strikes(ce, pe, spot, fii_composite)

        # Gamma pin zone: strike where total OI is highest
        gamma_pin = self._gamma_pin(ce, pe)

        # Institutional zones: call resistance and put support levels
        zones = self._institutional_zones(
            call_walls, put_walls, fii_call_bias, fii_put_bias, spot
        )

        # Overall direction inference
        direction, confidence = self._infer_direction(
            call_walls, put_walls, strike_scores, fii_composite, spot
        )

        return {
            "spot":                  spot,
            "call_walls":            call_walls,
            "put_walls":             put_walls,
            "gamma_pin_strike":      gamma_pin,
            "institutional_zones":   zones,
            "strike_scores":         strike_scores,
            "inferred_direction":    direction,
            "direction_confidence":  round(confidence, 3),
            "fii_composite":         fii_composite,
        }

    # ── Call / Put Walls ──────────────────────────────────────────────────────

    def _find_walls(self, df: pd.DataFrame, spot: float, side: str) -> List[Dict]:
        """
        Identify top N OTM strikes with heavy OI buildup.
        For calls: OTM = strike > spot.
        For puts:  OTM = strike < spot.
        """
        df = df.copy()
        df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0)
        df["oi_change"] = pd.to_numeric(df["oi_change"], errors="coerce").fillna(0)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

        if side == "CALL":
            otm = df[df["strike"] >= spot].copy()
        else:
            otm = df[df["strike"] <= spot].copy()

        if otm.empty:
            return []

        # Wall score = OI + 0.5 * OI change (fresh buildup weighted)
        otm["wall_score"] = otm["oi"] + 0.5 * otm["oi_change"].clip(lower=0)
        otm = otm.sort_values("wall_score", ascending=False).head(self.top_n)

        walls = []
        for _, row in otm.iterrows():
            walls.append({
                "strike":     int(row["strike"]),
                "oi":         int(row["oi"]),
                "oi_change":  int(row.get("oi_change", 0)),
                "volume":     int(row.get("volume", 0)),
                "iv":         round(float(row.get("iv", 0)), 2),
                "ltp":        round(float(row.get("ltp", 0)), 2),
                "wall_score": round(float(row["wall_score"]), 0),
                "distance_pct": round((row["strike"] - spot) / spot * 100, 2),
            })
        return walls

    # ── Gamma Pin ─────────────────────────────────────────────────────────────

    def _gamma_pin(self, ce: pd.DataFrame, pe: pd.DataFrame) -> Optional[int]:
        """Strike with maximum total (CE + PE) OI → likely pin zone into expiry."""
        ce_oi = ce.set_index("strike")["oi"].apply(pd.to_numeric, errors="coerce").fillna(0)
        pe_oi = pe.set_index("strike")["oi"].apply(pd.to_numeric, errors="coerce").fillna(0)
        total = ce_oi.add(pe_oi, fill_value=0)
        if total.empty:
            return None
        return int(total.idxmax())

    # ── Strike Scores ─────────────────────────────────────────────────────────

    def _score_strikes(
        self,
        ce: pd.DataFrame,
        pe: pd.DataFrame,
        spot: float,
        fii_composite: float,
    ) -> List[Dict]:
        """
        Score each strike for institutional activity.
        Positive = bullish activity; Negative = bearish activity.
        """
        all_strikes = sorted(set(ce["strike"].tolist() + pe["strike"].tolist()))
        ce_map = ce.set_index("strike").to_dict("index")
        pe_map = pe.set_index("strike").to_dict("index")

        results = []
        for s in all_strikes:
            ce_row = ce_map.get(s, {})
            pe_row = pe_map.get(s, {})

            ce_oi_chg = float(ce_row.get("oi_change", 0))
            pe_oi_chg = float(pe_row.get("oi_change", 0))
            ce_vol    = float(ce_row.get("volume", 0))
            pe_vol    = float(pe_row.get("volume", 0))

            # Put-side buildup is bullish (put writers are sellers)
            # Call-side buildup is bearish (call writers are sellers)
            # Raw score: net put buildup - net call buildup, normalised
            total_chg = abs(ce_oi_chg) + abs(pe_oi_chg) + 1
            raw_score = (pe_oi_chg - ce_oi_chg) / total_chg

            # Blend with FII composite (small weight)
            score = 0.8 * raw_score + 0.2 * fii_composite
            distance_pct = (s - spot) / spot * 100

            results.append({
                "strike":       int(s),
                "distance_pct": round(distance_pct, 2),
                "ce_oi_change": int(ce_oi_chg),
                "pe_oi_change": int(pe_oi_chg),
                "ce_volume":    int(ce_vol),
                "pe_volume":    int(pe_vol),
                "score":        round(score, 4),   # +ve bullish, -ve bearish
                "activity":     self._activity_label(score),
            })

        return sorted(results, key=lambda x: abs(x["score"]), reverse=True)[:10]

    @staticmethod
    def _activity_label(score: float) -> str:
        if score >  0.4: return "STRONG_BULLISH"
        if score >  0.1: return "MILD_BULLISH"
        if score > -0.1: return "NEUTRAL"
        if score > -0.4: return "MILD_BEARISH"
        return "STRONG_BEARISH"

    # ── Institutional Zones ───────────────────────────────────────────────────

    def _institutional_zones(
        self,
        call_walls: List[Dict],
        put_walls:  List[Dict],
        fii_call_bias: float,
        fii_put_bias:  float,
        spot: float,
    ) -> Dict:
        """
        Infer resistance and support zones, adjusted by FII bias.
        """
        resistance = [w["strike"] for w in call_walls]
        support    = [w["strike"] for w in put_walls]

        nearest_resistance = min(resistance, key=lambda x: abs(x - spot)) if resistance else None
        nearest_support    = max(support,    key=lambda x: abs(x - spot)) if support    else None

        # Institutional call writing (bearish for calls) → stronger resistance
        resistance_strength = "STRONG" if fii_call_bias < -0.3 else "MODERATE" if fii_call_bias < 0 else "WEAK"
        # Institutional put writing (bullish for puts) → stronger support
        support_strength    = "STRONG" if fii_put_bias  < -0.3 else "MODERATE" if fii_put_bias  < 0 else "WEAK"

        return {
            "call_resistance":          resistance,
            "put_support":              support,
            "nearest_resistance":       nearest_resistance,
            "nearest_support":          nearest_support,
            "resistance_strength":      resistance_strength,
            "support_strength":         support_strength,
        }

    # ── Direction Inference ───────────────────────────────────────────────────

    def _infer_direction(
        self,
        call_walls: List[Dict],
        put_walls:  List[Dict],
        strike_scores: List[Dict],
        fii_composite: float,
        spot: float,
    ) -> Tuple[str, float]:
        """
        Infer overall directional bias from strike + FII data.
        Returns (direction, confidence) where direction ∈ {BULLISH, BEARISH, NEUTRAL}.
        """
        # OI pressure score
        total_call_oi = sum(w["oi"] for w in call_walls)
        total_put_oi  = sum(w["oi"] for w in put_walls)
        oi_pressure   = (total_put_oi - total_call_oi) / (total_put_oi + total_call_oi + 1)

        # Combine with FII composite
        combined = 0.5 * oi_pressure + 0.5 * fii_composite

        if combined > 0.2:
            direction = "BULLISH"
        elif combined < -0.2:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        confidence = min(abs(combined), 1.0)
        return direction, confidence

    # ── Empty result ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "spot": 0,
            "call_walls": [],
            "put_walls":  [],
            "gamma_pin_strike": None,
            "institutional_zones": {},
            "strike_scores": [],
            "inferred_direction": "NEUTRAL",
            "direction_confidence": 0.0,
            "fii_composite": 0.0,
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_engine: Optional[StrikeAttributionEngine] = None


def get_strike_engine() -> StrikeAttributionEngine:
    global _engine
    if _engine is None:
        _engine = StrikeAttributionEngine()
    return _engine
