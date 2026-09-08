"""
XAUUSD Scalping Strategy
=========================
1-minute bars + tape reading for fast entries during London/NY sessions.

Signal types:
  VWAP_BOUNCE  — price pulls to VWAP ± sigma, tape shows absorption/reversal
  CVD_DIVERGE  — price diverges from CVD, mean-reversion trade
  TAPE_BREAKOUT— strong tape (STRONGLY_BULLISH/BEARISH) + volume spike + momentum
  ABSORB_FLIP  — institutional absorption bar at key level, expect flip

Risk:
  SL:  4 pts  (tight, 1-min ATR based)
  TP1: 5 pts  (1:1.25 — move SL to BE)
  TP2: 10 pts (1:2.5 — full target)

Sessions: London (07-12 UTC), NY AM (13-17 UTC) only.
Quiet hours (Asian / NY PM) → no scalp signals.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

# ── Params ──────────────────────────────────────────────────────────────────
SL_PTS   = 4.0
TP1_PTS  = 5.0
TP2_PTS  = 10.0
MIN_ATR  = 2.0    # ignore signal if 1m ATR < $2 (too flat)
MAX_ATR  = 25.0   # ignore signal if 1m ATR > $25 (too wild, news spike)
MIN_VOL_MULT = 1.3
KEY_LEVEL_ZONE = 6.0   # pts — price must be within this of a key level

# Session windows (UTC hours)
LONDON_OPEN, LONDON_CLOSE   = 7,  12
NY_OPEN,     NY_CLOSE       = 13, 17


# ── Signal dataclass ─────────────────────────────────────────────────────────
@dataclass
class ScalpSignal:
    action:     str    # BUY | SELL | WAIT
    signal_type: str   # VWAP_BOUNCE | CVD_DIVERGE | TAPE_BREAKOUT | ABSORB_FLIP
    entry:      float  = 0.0
    sl:         float  = 0.0
    tp1:        float  = 0.0
    tp2:        float  = 0.0
    atr_1m:     float  = 0.0
    tape_bias:  str    = ""
    buy_pressure: float = 0.0
    delta_5m:   float  = 0.0
    vwap:       float  = 0.0
    vwap_dev:   float  = 0.0   # price distance from VWAP in sigma
    session:    str    = ""
    score:      int    = 0
    reasons:    list   = field(default_factory=list)
    timestamp:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_trade(self) -> bool:
        return self.action in ("BUY", "SELL")

    def telegram_html(self) -> str:
        if not self.is_trade():
            return (f"⏳ <b>SCALP WAIT</b> — {self.signal_type}\n"
                    f"Tape: {self.tape_bias} | Buy%: {self.buy_pressure:.0f}%\n"
                    f"VWAP dev: {self.vwap_dev:+.1f}σ | Score: {self.score}")
        emoji  = "⚡🟢 SCALP BUY" if self.action == "BUY" else "⚡🔴 SCALP SELL"
        rr     = round(TP2_PTS / SL_PTS, 1)
        return (
            f"{emoji} — <b>XAUUSD</b> [{self.signal_type}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Entry    : <b>${self.entry:.2f}</b>\n"
            f"🛑 SL       : <b>${self.sl:.2f}</b>  ({SL_PTS:.0f} pts)\n"
            f"✅ TP1      : <b>${self.tp1:.2f}</b>  ({TP1_PTS:.0f} pts)\n"
            f"✅ TP2      : <b>${self.tp2:.2f}</b>  ({TP2_PTS:.0f} pts)  R:R 1:{rr}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌊 Tape     : {self.tape_bias} | Buy: {self.buy_pressure:.0f}%\n"
            f"📊 Delta 5m : {self.delta_5m:+.0f} | VWAP dev: {self.vwap_dev:+.1f}σ\n"
            f"💡 {' | '.join(self.reasons[:4])}\n"
            f"⏱ {self.timestamp.strftime('%d %b %H:%M UTC')} [{self.session}]"
        )


# ── Key S/R levels ───────────────────────────────────────────────────────────
def _find_key_levels(df_1m: pd.DataFrame, df_5m: pd.DataFrame, price: float) -> dict:
    """
    Find nearby support/resistance from swing pivots, round numbers, session range.
    Returns nearest_support, nearest_resistance, at_key_level, distance_pts.
    """
    levels = []

    # 5m swing highs/lows (5-bar pivot)
    if df_5m is not None and len(df_5m) >= 15:
        for i in range(5, len(df_5m) - 5):
            h = float(df_5m["high"].iloc[i])
            l = float(df_5m["low"].iloc[i])
            if h == df_5m["high"].iloc[i-5:i+6].max():
                levels.append(h)
            if l == df_5m["low"].iloc[i-5:i+6].min():
                levels.append(l)

    # Session high/low from last 60 1m bars
    if len(df_1m) >= 30:
        levels.append(float(df_1m["high"].tail(60).max()))
        levels.append(float(df_1m["low"].tail(60).min()))

    # Round numbers every $25
    base = round(price / 25) * 25
    for off in (-75, -50, -25, 0, 25, 50, 75):
        levels.append(base + off)

    supports    = [v for v in levels if v < price - 0.5]
    resistances = [v for v in levels if v > price + 0.5]

    nearest_sup = max(supports)    if supports    else price - 999
    nearest_res = min(resistances) if resistances else price + 999
    dist_sup    = price - nearest_sup
    dist_res    = nearest_res - price
    distance    = min(dist_sup, dist_res)

    return {
        "nearest_support":    round(nearest_sup, 2),
        "nearest_resistance": round(nearest_res, 2),
        "dist_support":       round(dist_sup, 2),
        "dist_resistance":    round(dist_res, 2),
        "at_key_level":       distance <= KEY_LEVEL_ZONE,
        "distance_pts":       round(distance, 2),
    }


def _htf_trend(df_5m: pd.DataFrame) -> str:
    """5m EMA13/34 trend: BULLISH | BEARISH | NEUTRAL."""
    if df_5m is None or len(df_5m) < 35:
        return "NEUTRAL"
    close = df_5m["close"]
    e13   = close.ewm(span=13, adjust=False).mean().iloc[-1]
    e34   = close.ewm(span=34, adjust=False).mean().iloc[-1]
    price = float(close.iloc[-1])
    if e13 > e34 and price > e13:
        return "BULLISH"
    if e13 < e34 and price < e13:
        return "BEARISH"
    return "NEUTRAL"


# ── Session helper ───────────────────────────────────────────────────────────
def _get_session(now: datetime) -> str:
    h = now.hour
    if LONDON_OPEN <= h < LONDON_CLOSE:
        return "LONDON"
    if NY_OPEN <= h < NY_CLOSE:
        return "NEW_YORK"
    if 12 <= h < 13:
        return "LUNCH"
    return "QUIET"


# ── VWAP + bands ─────────────────────────────────────────────────────────────
def _compute_vwap_bands(df: pd.DataFrame):
    """Intraday VWAP with ±1σ and ±2σ bands. Resets at UTC midnight."""
    d = df.copy()
    d["tp"]   = (d["high"] + d["low"] + d["close"]) / 3
    d["pvol"] = d["tp"] * d["volume"]
    d["cum_pvol"] = d["pvol"].cumsum()
    d["cum_vol"]  = d["volume"].cumsum().replace(0, np.nan)
    d["vwap"]     = d["cum_pvol"] / d["cum_vol"]
    variance = ((d["tp"] - d["vwap"]) ** 2 * d["volume"]).cumsum() / d["cum_vol"]
    d["sigma"] = np.sqrt(variance).fillna(0)
    return d


# ── ATR ──────────────────────────────────────────────────────────────────────
def _atr(df: pd.DataFrame, n: int = 10) -> float:
    if len(df) < n + 1:
        return 0.0
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return float(pd.concat([hl, hc, lc], axis=1).max(axis=1).tail(n).mean())


# ── Core scorer ───────────────────────────────────────────────────────────────
def _score_direction(
    direction: str,  # "BUY" or "SELL"
    tape: dict,
    vwap: float,
    sigma: float,
    price: float,
    df1m: pd.DataFrame,
) -> tuple[int, list[str]]:
    """Return (score, reasons) for one direction."""
    is_buy   = direction == "BUY"
    score    = 0
    reasons  = []
    bias     = tape.get("tape_bias", "NEUTRAL")
    buy_pct  = tape.get("buy_pressure", 50.0)
    d5       = tape.get("delta_5m", 0.0)
    d15      = tape.get("delta_15m", 0.0)
    absorb   = tape.get("absorption", False)
    absorb_s = tape.get("absorption_side", "")
    climax   = tape.get("climax", False)
    climax_t = tape.get("climax_type", "")
    mom_dir  = tape.get("momentum_dir", "MIXED")
    speed    = tape.get("tape_speed", "NORMAL")

    # 1. Tape bias alignment (0-3 pts)
    if is_buy:
        if bias == "STRONGLY_BULLISH":
            score += 3; reasons.append("Tape STRONGLY BULLISH")
        elif bias == "BULLISH":
            score += 2; reasons.append("Tape BULLISH")
        elif bias in ("STRONGLY_BEARISH", "BEARISH"):
            score -= 1
    else:
        if bias == "STRONGLY_BEARISH":
            score += 3; reasons.append("Tape STRONGLY BEARISH")
        elif bias == "BEARISH":
            score += 2; reasons.append("Tape BEARISH")
        elif bias in ("STRONGLY_BULLISH", "BULLISH"):
            score -= 1

    # 2. Buy pressure (0-2 pts)
    if is_buy and buy_pct > 62:
        score += 2; reasons.append(f"Buy pressure {buy_pct:.0f}%")
    elif is_buy and buy_pct > 55:
        score += 1; reasons.append(f"Buy pressure {buy_pct:.0f}%")
    elif not is_buy and buy_pct < 38:
        score += 2; reasons.append(f"Sell pressure {100-buy_pct:.0f}%")
    elif not is_buy and buy_pct < 45:
        score += 1; reasons.append(f"Sell pressure {100-buy_pct:.0f}%")

    # 3. CVD delta (0-2 pts)
    if is_buy and d5 > 0 and d15 > 0:
        score += 2; reasons.append(f"CVD bullish Δ5m={d5:+.0f}")
    elif is_buy and d5 > 0:
        score += 1; reasons.append(f"CVD Δ5m={d5:+.0f} positive")
    elif not is_buy and d5 < 0 and d15 < 0:
        score += 2; reasons.append(f"CVD bearish Δ5m={d5:+.0f}")
    elif not is_buy and d5 < 0:
        score += 1; reasons.append(f"CVD Δ5m={d5:+.0f} negative")

    # 4. VWAP position (0-2 pts)
    if vwap > 0 and sigma > 0:
        dev = (price - vwap) / sigma
        if is_buy and -1.5 <= dev <= -0.5:
            score += 2; reasons.append(f"Price at VWAP -{abs(dev):.1f}σ (buy zone)")
        elif is_buy and dev < -1.5:
            score += 1; reasons.append(f"Price below VWAP {dev:.1f}σ (deep value)")
        elif not is_buy and 0.5 <= dev <= 1.5:
            score += 2; reasons.append(f"Price at VWAP +{dev:.1f}σ (sell zone)")
        elif not is_buy and dev > 1.5:
            score += 1; reasons.append(f"Price above VWAP +{dev:.1f}σ (overextended)")

    # 5. Absorption at key level (0-2 pts)
    if absorb:
        if (is_buy and absorb_s == "BULL_ABSORB") or (not is_buy and absorb_s == "BEAR_ABSORB"):
            score += 2; reasons.append(f"Absorption: {absorb_s} at ${price:.2f}")

    # 6. Climax exhaustion → expect reversal (0-2 pts)
    if climax:
        if (is_buy and climax_t == "SELLING_CLIMAX") or (not is_buy and climax_t == "BUYING_CLIMAX"):
            score += 2; reasons.append(f"{climax_t} — exhaustion reversal entry")

    # 7. Momentum in direction (0-1 pts)
    if is_buy and mom_dir == "UP":
        score += 1; reasons.append("Momentum stacking UP")
    elif not is_buy and mom_dir == "DOWN":
        score += 1; reasons.append("Momentum stacking DOWN")

    # 8. Tape speed (0-1 pts) — fast tape = strong conviction
    if speed == "FAST":
        score += 1; reasons.append("Tape speed FAST (vol accelerating)")

    # 9. EMA micro-trend on 1m (0-1 pts)
    if len(df1m) >= 21:
        ema8  = df1m["close"].ewm(span=8,  adjust=False).mean().iloc[-1]
        ema21 = df1m["close"].ewm(span=21, adjust=False).mean().iloc[-1]
        if is_buy and ema8 > ema21:
            score += 1; reasons.append("EMA8 > EMA21 (1m bullish)")
        elif not is_buy and ema8 < ema21:
            score += 1; reasons.append("EMA8 < EMA21 (1m bearish)")

    # 10. Counter-trend penalty — hard subtract if HTF disagrees
    # (htf_trend injected via extra kwarg, handled in analyze_scalp)

    return score, reasons


# ── Main entry ────────────────────────────────────────────────────────────────
def analyze_scalp(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> ScalpSignal:
    """
    Main scalp analysis. Call every 30s during active sessions.
    df_1m: last 60+ 1-min bars   (yahooquery GLD × ratio)
    df_5m: last 30+ 5-min bars   (for trend context)
    """
    TRADE_THRESH = 8   # out of ~15 pts max

    now     = datetime.now(timezone.utc)
    session = _get_session(now)

    _wait = lambda reason, score=0: ScalpSignal(
        action="WAIT", signal_type=reason, session=session,
        score=score, timestamp=now,
    )

    # Session gate — only London + NY
    if session in ("QUIET", "LUNCH"):
        return _wait(f"OUT_OF_SESSION ({session})")

    if df_1m.empty or len(df_1m) < 20:
        return _wait("INSUFFICIENT_DATA")

    # Compute indicators
    df1 = _compute_vwap_bands(df_1m.copy())
    atr = _atr(df_1m)
    if atr < MIN_ATR:
        return _wait(f"FLAT_MARKET (ATR={atr:.2f})")
    if atr > MAX_ATR:
        return _wait(f"TOO_VOLATILE (ATR={atr:.2f}, news?)")

    price  = float(df_1m["close"].iloc[-1])
    vwap   = float(df1["vwap"].iloc[-1])
    sigma  = float(df1["sigma"].iloc[-1]) or 1.0
    vwap_dev = (price - vwap) / sigma

    # Current bar volume vs average
    avg_vol  = float(df_1m["volume"].tail(20).mean()) or 1.0
    cur_vol  = float(df_1m["volume"].iloc[-1])
    vol_mult = cur_vol / avg_vol

    # Tape analysis (uses tape_reader)
    try:
        from xauusd.tape_reader import analyze_tape
        tape = analyze_tape(df_1m)
    except Exception as e:
        logger.warning(f"[Scalp] Tape read failed: {e}")
        tape = {"tape_bias": "NEUTRAL", "buy_pressure": 50.0,
                "delta_5m": 0.0, "delta_15m": 0.0,
                "absorption": False, "climax": False,
                "absorption_side": "", "climax_type": "",
                "momentum_dir": "MIXED", "tape_speed": "NORMAL"}

    # HTF trend (5m EMA13/34)
    htf = _htf_trend(df_5m)

    # Key S/R levels
    kl = _find_key_levels(df_1m, df_5m, price)

    # Score both directions
    buy_score,  buy_reasons  = _score_direction("BUY",  tape, vwap, sigma, price, df_1m)
    sell_score, sell_reasons = _score_direction("SELL", tape, vwap, sigma, price, df_1m)

    # HTF alignment bonus/penalty (±2 pts each direction)
    if htf == "BULLISH":
        buy_score  += 2; buy_reasons.append(f"5m trend BULLISH (EMA13>34)")
        sell_score -= 2
    elif htf == "BEARISH":
        sell_score += 2; sell_reasons.append(f"5m trend BEARISH (EMA13<34)")
        buy_score  -= 2

    # Key level proximity (0-3 pts) — only give to the direction that makes sense
    if kl["at_key_level"]:
        d_sup = kl["dist_support"]
        d_res = kl["dist_resistance"]
        if d_sup <= KEY_LEVEL_ZONE:
            pts = 3 if d_sup <= 2.0 else (2 if d_sup <= 4.0 else 1)
            buy_score += pts
            buy_reasons.append(f"Near support {kl['nearest_support']:.0f} ({d_sup:.1f}pt away)")
        if d_res <= KEY_LEVEL_ZONE:
            pts = 3 if d_res <= 2.0 else (2 if d_res <= 4.0 else 1)
            sell_score += pts
            sell_reasons.append(f"Near resistance {kl['nearest_resistance']:.0f} ({d_res:.1f}pt away)")

    # Volume confirmation — directional only (close direction determines side)
    if vol_mult >= MIN_VOL_MULT:
        last_close = float(df_1m["close"].iloc[-1])
        last_open  = float(df_1m["open"].iloc[-1])
        if last_close > last_open:
            buy_score += 1; buy_reasons.append(f"Vol spike {vol_mult:.1f}× on green bar")
        else:
            sell_score += 1; sell_reasons.append(f"Vol spike {vol_mult:.1f}× on red bar")

    # Determine action
    TRADE_THRESH = 10   # raised from 8 — require stronger confluence
    if buy_score > sell_score and buy_score >= TRADE_THRESH:
        action, score, reasons = "BUY", buy_score, buy_reasons
        signal_type = _classify_signal(tape, vwap_dev, "BUY")
    elif sell_score > buy_score and sell_score >= TRADE_THRESH:
        action, score, reasons = "SELL", sell_score, sell_reasons
        signal_type = _classify_signal(tape, vwap_dev, "SELL")
    else:
        best = max(buy_score, sell_score)
        top_reasons = (buy_reasons if buy_score >= sell_score else sell_reasons)[:3]
        if not kl["at_key_level"]:
            top_reasons = [f"Not at key level (nearest {kl['distance_pts']:.1f}pt away)"] + top_reasons
        return ScalpSignal(
            action="WAIT", signal_type="LOW_SCORE",
            tape_bias=tape["tape_bias"], buy_pressure=tape["buy_pressure"],
            delta_5m=tape["delta_5m"], vwap=vwap, vwap_dev=vwap_dev,
            session=session, score=best, reasons=top_reasons, timestamp=now,
        )

    # Build levels
    if action == "BUY":
        sl  = round(price - SL_PTS,  2)
        tp1 = round(price + TP1_PTS, 2)
        tp2 = round(price + TP2_PTS, 2)
    else:
        sl  = round(price + SL_PTS,  2)
        tp1 = round(price - TP1_PTS, 2)
        tp2 = round(price - TP2_PTS, 2)

    return ScalpSignal(
        action=action, signal_type=signal_type,
        entry=price, sl=sl, tp1=tp1, tp2=tp2,
        atr_1m=round(atr, 2),
        tape_bias=tape["tape_bias"], buy_pressure=tape["buy_pressure"],
        delta_5m=tape["delta_5m"], vwap=vwap, vwap_dev=round(vwap_dev, 2),
        session=session, score=score, reasons=reasons,
        timestamp=now,
    )


def _classify_signal(tape: dict, vwap_dev: float, direction: str) -> str:
    """Name the signal type based on what triggered it."""
    if tape.get("absorption"):
        return "ABSORB_FLIP"
    if tape.get("climax"):
        return "CLIMAX_REVERSAL"
    is_buy = direction == "BUY"
    if (is_buy and -1.5 <= vwap_dev <= -0.3) or (not is_buy and 0.3 <= vwap_dev <= 1.5):
        return "VWAP_BOUNCE"
    if abs(tape.get("delta_5m", 0)) > 0 and abs(tape.get("delta_15m", 0)) > 0:
        return "CVD_MOMENTUM"
    return "TAPE_BREAKOUT"
