"""
Self-Learning Score Weight Adjuster
======================================
After each trade closes (SL/TP hit), record which scoring factors
were active at entry and whether the trade won or lost.

Over time, adjust factor weights: winning factors get boosted,
losing factors get reduced. This makes the scoring system learn
from its own mistakes.

Storage: data/of_score_weights.json
Format:
  {
    "weights": {
      "4H bias BULLISH": 3.2,   # started at 3, boosted after wins
      "1H BOS_UP aligned": 0.8, # started at 1, reduced after losses
      ...
    },
    "factor_stats": {
      "4H bias BULLISH": {"wins": 12, "losses": 3, "total": 15, "win_rate": 0.80}
    },
    "last_updated": "2026-09-02T23:00:00"
  }

Learning algorithm:
  win_rate = wins / total
  new_weight = base_weight * (0.5 + win_rate)  (range: 0.5x to 1.5x base)
  Minimum weight: 0.1
  Maximum weight: base * 2.0
  Minimum trades per factor before adjusting: 5
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime
from loguru import logger

WEIGHTS_FILE    = Path(__file__).parent.parent / "data" / "of_score_weights.json"
MIN_TRADES      = 5      # need at least 5 trades to start adjusting
LEARNING_RATE   = 0.1    # how aggressively to adjust weights
MIN_WEIGHT_MULT = 0.3    # floor: 30% of base weight
MAX_WEIGHT_MULT = 2.0    # ceiling: 200% of base weight

# Base weights (from of_strategy.py scoring — must match)
BASE_WEIGHTS: dict[str, float] = {
    # HTF structure
    "4H bias BULLISH":             3.0,
    "4H bias BEARISH":             3.0,
    "4H BOS_UP":                   1.0,
    "4H BOS_DOWN":                 1.0,
    "1H BOS aligned":              1.0,
    # OB
    "4H OB":                       2.0,
    "1H OB":                       2.0,
    "15m OB":                      1.0,
    "4H+1H OB stack":              1.0,
    # FVG
    "15m FVG fill":                2.0,
    "1H FVG":                      1.0,
    "HTF FVG":                     1.0,
    "Inversion FVG":               1.0,
    # Liquidity
    "Multi-TF liquidity sweep":    3.0,
    "Liquidity sweep":             2.0,
    # EQH/L
    "EQL swept":                   2.0,
    "EQH swept":                   2.0,
    "EQL support":                 1.0,
    "EQH resistance":              1.0,
    # OTE
    "OTE zone":                    3.0,
    "Near OTE":                    1.0,
    # Premium/Discount
    "DISCOUNT zone":               2.0,
    "PREMIUM zone":                2.0,
    "EQUILIBRIUM":                 1.0,
    # Displacement
    "Displacement candle":         2.0,
    # Prev day/week
    "Previous Day Low":            1.0,
    "Previous Day High":           1.0,
    "Previous Week Low":           1.0,
    "Previous Week High":          1.0,
    # CVD
    "CVD bullish divergence":      3.0,
    "CVD bearish divergence":      3.0,
    "CVD BULLISH bias":            1.0,
    "CVD BEARISH bias":            1.0,
    # Tape
    "Tape STRONGLY BULLISH":       3.0,
    "Tape STRONGLY BEARISH":       3.0,
    "Tape BULLISH":                2.0,
    "Tape BEARISH":                2.0,
    "Tape climax":                 1.0,
    "Tape absorption":             1.0,
    # Volume Profile
    "At POC":                      1.0,
    "At VAL":                      2.0,
    "At VAH":                      2.0,
    # Killzone
    "NY Silver Bullet":            3.0,
    "London KZ":                   2.0,
    "NY AM":                       2.0,
    "Asian range fade":            1.0,
    # VWAP
    "VWAP control":                1.0,
    "VWAP -2σ":                    2.0,
    "VWAP +2σ":                    2.0,
    "VWAP -1σ":                    1.0,
    "VWAP +1σ":                    1.0,
}


def _load() -> dict:
    try:
        if WEIGHTS_FILE.exists():
            return json.loads(WEIGHTS_FILE.read_text())
    except Exception:
        pass
    return {"weights": {}, "factor_stats": {}, "last_updated": ""}


def _save(data: dict):
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = datetime.now().isoformat()
    WEIGHTS_FILE.write_text(json.dumps(data, indent=2))


def get_current_weights() -> dict[str, float]:
    """Return current adjusted weights (base + learned adjustments)."""
    data = _load()
    learned  = data.get("weights", {})
    # Merge: use learned weight if available, else base
    merged = {}
    for factor, base in BASE_WEIGHTS.items():
        merged[factor] = learned.get(factor, base)
    return merged


def _match_factor(reason: str) -> str | None:
    """Map a reason string to a known factor key."""
    r = reason.lower()
    # Simple keyword matching
    mapping = [
        ("4h bias bullish",         "4H bias BULLISH"),
        ("4h bias bearish",         "4H bias BEARISH"),
        ("4h bos_up",               "4H BOS_UP"),
        ("4h bos_down",             "4H BOS_DOWN"),
        ("1h bos_up aligned",       "1H BOS aligned"),
        ("1h bos_down aligned",     "1H BOS aligned"),
        ("1h choch",                "1H BOS aligned"),
        ("4h+1h ob stack",          "4H+1H OB stack"),
        ("4h bullish ob",           "4H OB"),
        ("4h bearish ob",           "4H OB"),
        ("1h bullish ob",           "1H OB"),
        ("1h bearish ob",           "1H OB"),
        ("15m ob confluence",       "15m OB"),
        ("15m fvg fill",            "15m FVG fill"),
        ("1h fvg confluence",       "1H FVG"),
        ("4h fvg",                  "HTF FVG"),
        ("1h fvg backing",          "HTF FVG"),
        ("inversion fvg",           "Inversion FVG"),
        ("multi-tf liquidity",      "Multi-TF liquidity sweep"),
        ("liquidity swept",         "Liquidity sweep"),
        ("eql",                     "EQL swept" if "swept" in r else "EQL support"),
        ("eqh",                     "EQH swept" if "swept" in r else "EQH resistance"),
        ("ote zone",                "OTE zone"),
        ("near ote",                "Near OTE"),
        ("discount zone",           "DISCOUNT zone"),
        ("premium zone",            "PREMIUM zone"),
        ("equilibrium",             "EQUILIBRIUM"),
        ("displacement candle",     "Displacement candle"),
        ("previous day low",        "Previous Day Low"),
        ("previous day high",       "Previous Day High"),
        ("previous week low",       "Previous Week Low"),
        ("previous week high",      "Previous Week High"),
        ("cvd bullish divergence",  "CVD bullish divergence"),
        ("cvd bearish divergence",  "CVD bearish divergence"),
        ("cvd bullish bias",        "CVD BULLISH bias"),
        ("cvd bearish bias",        "CVD BEARISH bias"),
        ("tape strongly bullish",   "Tape STRONGLY BULLISH"),
        ("tape strongly bearish",   "Tape STRONGLY BEARISH"),
        ("tape bullish",            "Tape BULLISH"),
        ("tape bearish",            "Tape BEARISH"),
        ("selling climax",          "Tape climax"),
        ("buying climax",           "Tape climax"),
        ("bull_absorb",             "Tape absorption"),
        ("bear_absorb",             "Tape absorption"),
        ("at poc",                  "At POC"),
        ("at val",                  "At VAL"),
        ("at vah",                  "At VAH"),
        ("ny silver bullet",        "NY Silver Bullet"),
        ("london kz",               "London KZ"),
        ("ny am",                   "NY AM"),
        ("asian range",             "Asian range fade"),
        ("above vwap",              "VWAP control"),
        ("below vwap",              "VWAP control"),
        ("vwap -2σ",                "VWAP -2σ"),
        ("vwap +2σ",                "VWAP +2σ"),
        ("vwap -1σ",                "VWAP -1σ"),
        ("vwap +1σ",                "VWAP +1σ"),
    ]
    for kw, factor in mapping:
        if kw in r:
            return factor
    return None


def record_trade_outcome(reasons: list[str], outcome: str):
    """
    Call this when a trade closes.
    outcome: "WIN" | "LOSS"
    reasons: list of reason strings from OFSignal.reasons
    """
    data = _load()
    weights = data.get("weights", {})
    stats   = data.get("factor_stats", {})

    won = (outcome == "WIN")

    for reason in reasons:
        factor = _match_factor(reason)
        if not factor:
            continue

        if factor not in stats:
            stats[factor] = {"wins": 0, "losses": 0, "total": 0}

        stats[factor]["total"] += 1
        if won:
            stats[factor]["wins"] += 1
        else:
            stats[factor]["losses"] += 1

        # Adjust weight if we have enough data
        total = stats[factor]["total"]
        if total >= MIN_TRADES:
            win_rate = stats[factor]["wins"] / total
            base     = BASE_WEIGHTS.get(factor, 1.0)
            # win_rate=1.0 → 1.5x, win_rate=0.5 → 1.0x, win_rate=0.0 → 0.5x
            multiplier = max(MIN_WEIGHT_MULT, min(MAX_WEIGHT_MULT, 0.5 + win_rate))
            new_weight = round(base * multiplier, 3)
            weights[factor] = new_weight
            logger.info(f"[Learner] {factor}: WR={win_rate:.0%} → weight {base} → {new_weight}")

        stats[factor]["win_rate"] = round(stats[factor]["wins"] / stats[factor]["total"], 3)

    data["weights"]      = weights
    data["factor_stats"] = stats
    _save(data)


def get_factor_performance() -> list[dict]:
    """Return sorted list of factors by win rate (for dashboard display)."""
    data = _load()
    stats = data.get("factor_stats", {})
    rows  = []
    for factor, s in stats.items():
        if s["total"] >= 2:
            rows.append({
                "Factor":    factor,
                "Trades":    s["total"],
                "Wins":      s["wins"],
                "Losses":    s["losses"],
                "Win Rate":  f"{s['win_rate']*100:.0f}%",
                "Weight":    round(data.get("weights", {}).get(factor, BASE_WEIGHTS.get(factor, 1.0)), 2),
            })
    return sorted(rows, key=lambda x: float(x["Win Rate"].rstrip("%")), reverse=True)


def summary() -> dict:
    data = _load()
    stats = data.get("factor_stats", {})
    total_factors = len([f for f, s in stats.items() if s["total"] >= MIN_TRADES])
    adjusted = len(data.get("weights", {}))
    return {
        "factors_tracked": len(stats),
        "factors_adjusted": adjusted,
        "last_updated": data.get("last_updated", "never"),
        "total_trades_recorded": sum(s["total"] for s in stats.values()) // max(len(stats), 1),
    }
