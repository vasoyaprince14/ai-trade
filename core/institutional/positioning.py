"""
Institutional Positioning Analyzer
=====================================
Combines FII/DII participant data with option chain to produce:
  - FII regime classification
  - Per-category bias scores (futures, calls, puts)
  - DII absorption analysis
  - FII/DII divergence flags
  - Position velocity (rate-of-change of positioning)

All methods are designed to work with or without live data.
"""
import sys
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


# ── Regime scoring ─────────────────────────────────────────────────────────────

REGIME_LABELS = {
    "AGGRESSIVE_BULLISH": ( 0.60,  1.00),
    "BULLISH":            ( 0.30,  0.60),
    "MILD_BULLISH":       ( 0.10,  0.30),
    "NEUTRAL":            (-0.10,  0.10),
    "MILD_BEARISH":       (-0.30, -0.10),
    "BEARISH":            (-0.60, -0.30),
    "AGGRESSIVE_BEARISH": (-1.00, -0.60),
}

# Sentiment score: +1 = max bullish, -1 = max bearish
REGIME_SCORES = {
    "AGGRESSIVE_BULLISH":  1.0,
    "BULLISH":             0.6,
    "MILD_BULLISH":        0.3,
    "NEUTRAL":             0.0,
    "MILD_BEARISH":       -0.3,
    "BEARISH":            -0.6,
    "AGGRESSIVE_BEARISH": -1.0,
}


class InstitutionalPositionAnalyzer:
    """
    Computes institutional positioning signals from raw participant data.
    Does NOT fetch data — call FIIDIIFetcher first.
    """

    def __init__(self):
        self._history: List[Dict] = []   # rolling daily snapshots

    # ── Main analysis entry point ──────────────────────────────────────────────

    def analyze(self, today_data: Dict) -> Dict:
        """
        Full institutional analysis from today_data (output of FIIDIIFetcher.get_today_data()).

        Returns enriched dict with:
          - regime, composite, per-category biases
          - dii_absorption (how much DII counters FII)
          - divergence flags
          - velocity vs previous snapshot
          - human-readable summary
        """
        self._history.append(today_data)
        if len(self._history) > 30:
            self._history.pop(0)

        fii = today_data.get("participants", {}).get("FII", {})
        dii = today_data.get("participants", {}).get("DII", {})
        cash = today_data.get("cash", {})

        composite      = today_data.get("fii_composite", 0.0)
        regime         = today_data.get("fii_regime", "NEUTRAL")
        futures_bias   = today_data.get("fii_futures_bias", 0.0)
        call_bias      = today_data.get("fii_call_bias", 0.0)
        put_bias       = today_data.get("fii_put_bias", 0.0)

        # DII absorption: how much DII buying/selling counters FII
        dii_absorption = self._calc_dii_absorption(cash)

        # Divergences
        divergences = self._detect_divergences(
            futures_bias, call_bias, put_bias, cash
        )

        # Velocity vs yesterday
        velocity = self._calc_velocity(today_data)

        # Human summary
        summary = self._build_summary(
            regime, composite, futures_bias, call_bias, put_bias,
            dii_absorption, divergences, cash
        )

        return {
            "date":              today_data.get("date"),
            "regime":            regime,
            "regime_score":      REGIME_SCORES.get(regime, 0.0),
            "composite":         composite,
            "futures_bias":      futures_bias,
            "call_bias":         call_bias,
            "put_bias":          put_bias,

            # FII raw positions
            "fii_fut_long":      fii.get("fut_index_long", 0),
            "fii_fut_short":     fii.get("fut_index_short", 0),
            "fii_fut_net":       fii.get("fut_index_net", 0),
            "fii_call_long":     fii.get("opt_idx_call_long", 0),
            "fii_call_short":    fii.get("opt_idx_call_short", 0),
            "fii_call_net":      fii.get("opt_idx_call_net", 0),
            "fii_put_long":      fii.get("opt_idx_put_long", 0),
            "fii_put_short":     fii.get("opt_idx_put_short", 0),
            "fii_put_net":       fii.get("opt_idx_put_net", 0),

            # DII
            "dii_fut_long":      dii.get("fut_index_long", 0),
            "dii_fut_short":     dii.get("fut_index_short", 0),
            "dii_fut_net":       dii.get("fut_index_long", 0) - dii.get("fut_index_short", 0),

            # Cash
            "fii_cash_net":      cash.get("fii_net", 0),
            "dii_cash_net":      cash.get("dii_net", 0),

            # Derived
            "dii_absorption":    dii_absorption,
            "divergences":       divergences,
            "velocity":          velocity,
            "summary":           summary,
        }

    # ── DII Absorption ────────────────────────────────────────────────────────

    def _calc_dii_absorption(self, cash: Dict) -> float:
        """
        Returns absorption ratio: 1.0 = DII perfectly offsets FII.
        Positive = DII buying when FII selling (absorption).
        Negative = DII and FII moving together.
        """
        fii_net = cash.get("fii_net", 0)
        dii_net = cash.get("dii_net", 0)
        total = abs(fii_net) + abs(dii_net)
        if total == 0:
            return 0.0
        # If FII sells (-) and DII buys (+): fii_net < 0, dii_net > 0 → ratio positive
        return round(-fii_net * dii_net / (total ** 2 + 1e-9), 4) if fii_net != 0 else 0.0

    # ── Divergence Detection ──────────────────────────────────────────────────

    def _detect_divergences(
        self, futures_bias: float, call_bias: float, put_bias: float, cash: Dict
    ) -> List[str]:
        """Flag notable positioning divergences."""
        flags = []

        # Futures vs options divergence
        options_bias = (call_bias + put_bias) / 2
        if futures_bias > 0.2 and options_bias < -0.2:
            flags.append("FUTURES_BULLISH_OPTIONS_BEARISH")
        elif futures_bias < -0.2 and options_bias > 0.2:
            flags.append("FUTURES_BEARISH_OPTIONS_BULLISH")

        # Call/put divergence (unusual: bullish calls + bearish puts)
        if call_bias > 0.3 and put_bias < -0.3:
            flags.append("CALL_LONG_PUT_LONG_BOTH_BEARISH_HEDGE")
        if call_bias < -0.3 and put_bias > 0.3:
            flags.append("CALL_SHORT_PUT_SHORT_PREMIUM_SELLING")

        # Cash vs futures divergence
        fii_cash = cash.get("fii_net", 0)
        if fii_cash > 1000 and futures_bias < -0.3:
            flags.append("FII_CASH_BUYING_FUTURES_HEDGED")
        elif fii_cash < -1000 and futures_bias > 0.3:
            flags.append("FII_CASH_SELLING_FUTURES_LONG")

        # DII absorbing FII selling
        if cash.get("fii_net", 0) < -500 and cash.get("dii_net", 0) > 300:
            flags.append("DII_ABSORBING_FII_SELLING")

        return flags

    # ── Velocity ─────────────────────────────────────────────────────────────

    def _calc_velocity(self, today_data: Dict) -> Dict:
        """Rate of change vs previous snapshot."""
        if len(self._history) < 2:
            return {}
        prev = self._history[-2]
        return {
            "composite_delta":   round(
                today_data.get("fii_composite", 0) - prev.get("fii_composite", 0), 4
            ),
            "futures_bias_delta": round(
                today_data.get("fii_futures_bias", 0) - prev.get("fii_futures_bias", 0), 4
            ),
            "call_bias_delta":   round(
                today_data.get("fii_call_bias", 0) - prev.get("fii_call_bias", 0), 4
            ),
            "put_bias_delta":    round(
                today_data.get("fii_put_bias", 0) - prev.get("fii_put_bias", 0), 4
            ),
        }

    # ── Summary ───────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        regime: str,
        composite: float,
        futures_bias: float,
        call_bias: float,
        put_bias: float,
        dii_absorption: float,
        divergences: List[str],
        cash: Dict,
    ) -> str:
        lines = []
        lines.append(f"FII Regime: {regime} (composite={composite:+.3f})")

        bias_map = {
            "Futures": futures_bias,
            "Calls":   call_bias,
            "Puts":    put_bias,
        }
        for label, bias in bias_map.items():
            direction = "Bullish" if bias > 0.1 else "Bearish" if bias < -0.1 else "Neutral"
            lines.append(f"  {label}: {direction} ({bias:+.3f})")

        fii_cash = cash.get("fii_net", 0)
        dii_cash = cash.get("dii_net", 0)
        if fii_cash:
            lines.append(f"  Cash: FII={fii_cash:+,}Cr  DII={dii_cash:+,}Cr")

        if dii_absorption > 0.3:
            lines.append("  DII is absorbing FII selling pressure")
        elif dii_absorption < -0.3:
            lines.append("  DII and FII moving in same direction")

        if divergences:
            lines.append(f"  Divergences: {', '.join(divergences)}")

        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────────────

_analyzer: Optional[InstitutionalPositionAnalyzer] = None


def get_analyzer() -> InstitutionalPositionAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = InstitutionalPositionAnalyzer()
    return _analyzer
