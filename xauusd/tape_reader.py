"""
XAUUSD Tape Reader
===================
Approximates real-time tape reading from OHLCV 1m/5m data.

In real tape reading you watch Time & Sales — every individual trade print:
  Price | Size | Side (bid/ask aggressor) | Time

Without tick data we approximate from OHLCV using:
  - Candle body/wick structure  → buy/sell pressure ratio
  - Volume at each price level  → where the big orders are
  - Volume acceleration         → tape speeding up (momentum building)
  - Absorption pattern          → large vol, tiny range = walls absorbing
  - Climax bars                 → extreme vol spike + small body = exhaustion
  - Stacking                    → consecutive bars same direction = momentum
  - Delta divergence            → price moves opposite to net delta

Tape Bias scale:
  STRONGLY_BULLISH → buyers completely dominating, tape flying green
  BULLISH          → more buyers than sellers
  NEUTRAL          → balanced, no edge
  BEARISH          → more sellers than buyers
  STRONGLY_BEARISH → sellers completely dominating

Output:
  {
    tape_bias:        str,
    buy_pressure:     float 0-100,   # % of vol that is buying
    sell_pressure:    float 0-100,
    delta_1m:         float,          # last bar delta
    delta_5m:         float,          # 5-bar cumulative delta
    delta_15m:        float,          # 15-bar cumulative delta
    absorption:       bool,           # high vol, tiny range at current level
    absorption_side:  str,            # BULL_ABSORB / BEAR_ABSORB
    climax:           bool,           # vol spike + reversal
    climax_type:      str,            # BUYING_CLIMAX / SELLING_CLIMAX
    iceberg:          bool,           # same price hit multiple times
    large_prints:     list[dict],     # vol > 2x avg prints
    tape_speed:       str,            # FAST / NORMAL / SLOW
    momentum_bars:    int,            # consecutive bars in direction
    momentum_dir:     str,            # UP / DOWN / MIXED
    tape_events:      list[str],      # human-readable events
    last_price:       float,
    avg_vol:          float,
    session_delta:    float,          # cumulative delta for session
  }
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from loguru import logger


# ── Constants ───────────────────────────────────────────────────────────────
LARGE_PRINT_MULT  = 2.0    # vol > 2x avg = large print
CLIMAX_MULT       = 3.0    # vol > 3x avg = climax bar
ABSORPTION_MAX_RANGE = 0.0015  # body < 0.15% of price = absorption
STACK_MIN_BARS    = 3      # min consecutive bars for "stacking"
VOL_LOOKBACK      = 50     # bars for avg volume


def _buy_sell_split(row: pd.Series) -> tuple[float, float]:
    """
    Estimate buy/sell volume per bar from candle structure.
    Uses wick analysis + body direction for better accuracy.
    """
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    v = float(row.get("volume", 0) or 0)
    if v == 0:
        return 0.0, 0.0

    rng = h - l
    if rng < 1e-9:
        return v * 0.5, v * 0.5

    body     = c - o
    body_abs = abs(body)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # Body contribution: bullish body = buy pressure, bearish = sell
    body_buy_ratio = (body / rng + 1) / 2   # 0 = all sell, 1 = all buy

    # Wick adjustment: upper wick = sellers fought back, lower wick = buyers fought back
    wick_adj = (lower_wick - upper_wick) / rng * 0.3

    buy_ratio = max(0.05, min(0.95, body_buy_ratio + wick_adj))
    return v * buy_ratio, v * (1 - buy_ratio)


def _rolling_avg_vol(df: pd.DataFrame, lookback: int = VOL_LOOKBACK) -> float:
    tail = df["volume"].tail(lookback).replace(0, np.nan).dropna()
    return float(tail.mean()) if not tail.empty else 1.0


def analyze_tape(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Main tape analysis. Input: 1m or 5m OHLCV DataFrame.
    Returns comprehensive tape reading dict.
    """
    result = {
        "tape_bias": "NEUTRAL", "buy_pressure": 50.0, "sell_pressure": 50.0,
        "delta_1m": 0.0, "delta_5m": 0.0, "delta_15m": 0.0,
        "absorption": False, "absorption_side": "",
        "climax": False, "climax_type": "",
        "iceberg": False,
        "large_prints": [],
        "tape_speed": "NORMAL",
        "momentum_bars": 0, "momentum_dir": "",
        "tape_events": [],
        "last_price": 0.0, "avg_vol": 0.0, "session_delta": 0.0,
        "buy_vol_series": [], "sell_vol_series": [], "timestamps": [],
        "delta_series": [],
    }

    if df.empty or len(df) < 10:
        return result

    df = df.copy().tail(lookback)
    avg_vol = _rolling_avg_vol(df)
    result["avg_vol"]    = round(avg_vol, 0)
    result["last_price"] = float(df["close"].iloc[-1])

    # ── Per-bar buy/sell split ──────────────────────────────────────────────
    buy_vols  = []
    sell_vols = []
    deltas    = []
    for _, row in df.iterrows():
        bv, sv = _buy_sell_split(row)
        buy_vols.append(bv)
        sell_vols.append(sv)
        deltas.append(bv - sv)

    df["buy_vol"]  = buy_vols
    df["sell_vol"] = sell_vols
    df["delta"]    = deltas
    df["cvd"]      = df["delta"].cumsum()

    result["buy_vol_series"]  = [round(v) for v in buy_vols[-30:]]
    result["sell_vol_series"] = [round(v) for v in sell_vols[-30:]]
    result["delta_series"]    = [round(d) for d in deltas[-30:]]
    result["timestamps"]      = [str(t)[:16] for t in df["timestamp"].tolist()[-30:]]

    # ── Delta windows ──────────────────────────────────────────────────────
    result["delta_1m"]      = round(float(deltas[-1]), 0) if deltas else 0
    result["delta_5m"]      = round(float(sum(deltas[-5:])), 0) if len(deltas) >= 5 else 0
    result["delta_15m"]     = round(float(sum(deltas[-15:])), 0) if len(deltas) >= 15 else 0
    result["session_delta"] = round(float(sum(deltas)), 0)

    # ── Buy/sell pressure % ────────────────────────────────────────────────
    total_buy  = sum(buy_vols[-20:])
    total_sell = sum(sell_vols[-20:])
    total      = total_buy + total_sell
    if total > 0:
        result["buy_pressure"]  = round(total_buy / total * 100, 1)
        result["sell_pressure"] = round(total_sell / total * 100, 1)

    # ── Large prints ──────────────────────────────────────────────────────
    large_prints = []
    for i, (_, row) in enumerate(df.iterrows()):
        v = float(row.get("volume", 0) or 0)
        if v >= avg_vol * LARGE_PRINT_MULT:
            side = "BUY" if deltas[i] > 0 else "SELL"
            mult = round(v / avg_vol, 1)
            large_prints.append({
                "time":  str(row.get("timestamp", ""))[:16],
                "price": round(float(row["close"]), 2),
                "vol":   round(v),
                "mult":  mult,
                "side":  side,
                "delta": round(deltas[i]),
            })
    result["large_prints"] = large_prints[-10:]   # last 10

    # ── Absorption detection ───────────────────────────────────────────────
    # High volume, small body = institutions absorbing at that level
    last_row  = df.iloc[-1]
    last_vol  = float(last_row.get("volume", 0) or 0)
    last_range= float(last_row["high"]) - float(last_row["low"])
    price_range_pct = last_range / float(last_row["close"]) if last_row["close"] else 0

    if last_vol >= avg_vol * 1.5 and price_range_pct < ABSORPTION_MAX_RANGE:
        result["absorption"] = True
        # If price near high = bears absorbing (selling into buying)
        # If price near low  = bulls absorbing (buying into selling)
        mid = (float(last_row["high"]) + float(last_row["low"])) / 2
        if float(last_row["close"]) > mid:
            result["absorption_side"] = "BEAR_ABSORB"   # sellers absorbing at top
            result["tape_events"].append(f"ABSORPTION: Bears absorbing at {last_row['close']:.2f} ({last_vol/avg_vol:.1f}x vol)")
        else:
            result["absorption_side"] = "BULL_ABSORB"   # buyers absorbing at bottom
            result["tape_events"].append(f"ABSORPTION: Bulls absorbing at {last_row['close']:.2f} ({last_vol/avg_vol:.1f}x vol)")

    # ── Climax bar detection ───────────────────────────────────────────────
    # Climax = extreme volume spike with small NEXT bar (exhaustion)
    if len(df) >= 3:
        prev2 = df.iloc[-3]; prev1 = df.iloc[-2]; last = df.iloc[-1]
        p1_vol = float(prev1.get("volume", 0) or 0)
        p1_delta = deltas[-2] if len(deltas) >= 2 else 0
        l_delta  = deltas[-1] if deltas else 0

        if p1_vol >= avg_vol * CLIMAX_MULT:
            # If bullish climax then bearish follow-through = BUYING CLIMAX
            if p1_delta > 0 and l_delta < 0 and float(prev1["close"]) > float(prev1["open"]):
                result["climax"]      = True
                result["climax_type"] = "BUYING_CLIMAX"
                result["tape_events"].append(f"CLIMAX: Buying climax at {prev1['close']:.2f} ({p1_vol/avg_vol:.1f}x vol) — reversal signal")
            elif p1_delta < 0 and l_delta > 0 and float(prev1["close"]) < float(prev1["open"]):
                result["climax"]      = True
                result["climax_type"] = "SELLING_CLIMAX"
                result["tape_events"].append(f"CLIMAX: Selling climax at {prev1['close']:.2f} ({p1_vol/avg_vol:.1f}x vol) — reversal signal")

    # ── Momentum stacking ─────────────────────────────────────────────────
    # Consecutive bars in same direction = momentum
    stack_up = stack_dn = 0
    for _, row in df.tail(10).iloc[::-1].iterrows():
        if float(row["close"]) > float(row["open"]):
            if stack_dn > 0: break
            stack_up += 1
        elif float(row["close"]) < float(row["open"]):
            if stack_up > 0: break
            stack_dn += 1
        else:
            break

    if stack_up >= STACK_MIN_BARS:
        result["momentum_bars"] = stack_up
        result["momentum_dir"]  = "UP"
        result["tape_events"].append(f"STACKING: {stack_up} consecutive green bars — bullish momentum")
    elif stack_dn >= STACK_MIN_BARS:
        result["momentum_bars"] = stack_dn
        result["momentum_dir"]  = "DOWN"
        result["tape_events"].append(f"STACKING: {stack_dn} consecutive red bars — bearish momentum")

    # ── Tape speed ────────────────────────────────────────────────────────
    # Measure vol acceleration: recent vs older average
    if len(df) >= 20:
        recent_avg = df["volume"].tail(5).mean()
        older_avg  = df["volume"].iloc[-20:-5].mean()
        if older_avg > 0:
            speed_ratio = recent_avg / older_avg
            if speed_ratio >= 1.5:
                result["tape_speed"] = "FAST"
                result["tape_events"].append(f"TAPE SPEED: Accelerating ({speed_ratio:.1f}x avg) — momentum building")
            elif speed_ratio <= 0.6:
                result["tape_speed"] = "SLOW"

    # ── Iceberg detection ─────────────────────────────────────────────────
    # Same price hit multiple times in last 5 bars with similar vol = hidden large order
    last5 = df.tail(5)
    price_hits: dict[float, int] = {}
    for _, row in last5.iterrows():
        p = round(float(row["close"]), 1)
        price_hits[p] = price_hits.get(p, 0) + 1
    max_hits = max(price_hits.values()) if price_hits else 0
    if max_hits >= 3:
        hit_price = max(price_hits, key=lambda x: price_hits[x])
        result["iceberg"] = True
        result["tape_events"].append(f"ICEBERG: Price {hit_price:.2f} tested {max_hits}x — large resting order suspected")

    # ── CVD divergence ────────────────────────────────────────────────────
    if len(df) >= 10:
        price_change = float(df["close"].iloc[-1]) - float(df["close"].iloc[-10])
        cvd_change   = float(df["cvd"].iloc[-1])  - float(df["cvd"].iloc[-10])
        if price_change > 0 and cvd_change < 0:
            result["tape_events"].append("CVD DIV: Price rising but sellers absorbing — distribution likely")
        elif price_change < 0 and cvd_change > 0:
            result["tape_events"].append("CVD DIV: Price falling but buyers absorbing — accumulation likely")

    # ── Final tape bias ───────────────────────────────────────────────────
    buy_pct = result["buy_pressure"]
    d15     = result["delta_15m"]
    d5      = result["delta_5m"]
    mom_dir = result["momentum_dir"]

    # Weighted score: 15m delta direction (40%), 5m delta (30%), buy% (30%)
    score = 0
    if d15 > 0: score += 2
    elif d15 < 0: score -= 2
    if d5 > 0: score += 1.5
    elif d5 < 0: score -= 1.5
    if buy_pct > 55: score += 1.5
    elif buy_pct < 45: score -= 1.5
    if mom_dir == "UP": score += 1
    elif mom_dir == "DOWN": score -= 1

    if score >= 4:    result["tape_bias"] = "STRONGLY_BULLISH"
    elif score >= 2:  result["tape_bias"] = "BULLISH"
    elif score <= -4: result["tape_bias"] = "STRONGLY_BEARISH"
    elif score <= -2: result["tape_bias"] = "BEARISH"
    else:             result["tape_bias"] = "NEUTRAL"

    return result


def tape_summary_line(t: dict) -> str:
    """One-line human summary of tape state."""
    bias  = t["tape_bias"]
    buy   = t["buy_pressure"]
    d5    = t["delta_5m"]
    speed = t["tape_speed"]
    events = len(t["tape_events"])
    return (f"Tape: {bias} | Buy: {buy:.0f}% | Delta5: {d5:+.0f} | "
            f"Speed: {speed} | Events: {events}")
