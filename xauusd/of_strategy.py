"""
XAUUSD Heavy Order Flow Strategy
==================================
Multi-layer institutional order flow analysis for Gold (GC=F / XAUUSD).

Combines:
  1. Smart Money Concepts  (smartmoneyconcepts lib)
     - Order Blocks (OB)           - institutional accumulation/distribution zones
     - Fair Value Gaps (FVG)        - imbalance zones price returns to fill
     - Break of Structure (BOS)     - trend continuation confirmation
     - Change of Character (CHoCH)  - trend reversal signal
     - Liquidity sweeps             - engineered liquidity grabs before moves
     - Session levels               - Asian range, London / NY killzones

  2. Volume Profile  (market_profile lib)
     - POC  (Point of Control)      - highest-volume price = magnet
     - VAH  (Value Area High)       - 70% volume upper boundary = resistance
     - VAL  (Value Area Low)        - 70% volume lower boundary = support
     - HVN / LVN                    - high/low volume nodes

  3. Cumulative Volume Delta (CVD)
     - Approximate from OHLCV (bull candle = buy vol, bear = sell vol)
     - CVD divergence vs price      - key exhaustion signal

  4. VWAP  (session + 1σ/2σ bands)
     - Price above/below session VWAP
     - Standard deviation bands as dynamic support/resistance

  5. ICT Killzone timing
     - Asian range      00:00-07:00 UTC  (consolidation)
     - London KZ        07:00-10:00 UTC  (first expansion)
     - NY AM            13:30-16:00 UTC  (continuation)
     - NY Silver Bullet 15:00-16:00 UTC  (precision entry)

  6. Multi-timeframe confluence
     - 4H  macro structure (major OB, BOS/CHoCH)
     - 1H  intermediate order blocks + FVG
     - 15m entry triggers
     - 5m  precision entry (FVG fill, OB tap)

Scoring: 0-20 pts
  - HTF structure alignment       0-4
  - Order Block tap (1H/15m)      0-3
  - FVG fill / mitigation         0-3
  - Liquidity sweep before entry  0-2
  - CVD divergence                0-2
  - Volume profile confluence     0-2
  - Killzone timing               0-2
  - VWAP position                 0-2
  => BUY/SELL if score >= 11 (55%), STRONG if >= 14 (70%)
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

# ── libs ──────────────────────────────────────────────────────────────────────
try:
    from smartmoneyconcepts import smc
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    logger.warning("[OF] smartmoneyconcepts not installed — run: pip install smartmoneyconcepts")

MP_AVAILABLE = False   # Use manual histogram VP (reliable across all data shapes)

# ── Constants ──────────────────────────────────────────────────────────────────

SCORE_BUY_THRESHOLD   = 22     # out of 40
SCORE_STRONG_THRESHOLD= 28
SWING_LENGTH          = 10
FVG_JOIN              = False
OB_CLOSE_MIT          = False
LIQ_RANGE_PCT         = 0.005
ATR_PERIOD            = 14
CVD_LOOKBACK          = 20
VOL_PROFILE_BINS      = 100

# OTE (Optimal Trade Entry) Fibonacci levels
OTE_DEEP   = 0.705   # 70.5% — ICT OTE zone
OTE_SHALLOW= 0.618   # 61.8% — OTE start
# Premium / Discount zone thresholds
PREMIUM_ZONE = 0.50   # above midpoint of range = premium (sell)
DISCOUNT_ZONE= 0.50   # below midpoint = discount (buy)

# ICT Killzone windows (UTC hours)
KILLZONES = {
    "Asian Range":      (0,  7),
    "London KZ":        (7,  10),
    "NY AM":            (13, 16),
    "NY Silver Bullet": (15, 16),
}

# ── Signal dataclass ───────────────────────────────────────────────────────────

@dataclass
class OFSignal:
    action:        str          # BUY | SELL | WAIT
    strength:      str          # STRONG | MODERATE | WEAK | WAIT
    entry:         float = 0.0
    stop_loss:     float = 0.0
    target1:       float = 0.0
    target2:       float = 0.0
    target3:       float = 0.0
    risk_reward:   float = 0.0
    score:         int   = 0
    max_score:     int   = 20
    confidence:    float = 0.0
    atr:           float = 0.0
    killzone:      str   = ""
    htf_bias:      str   = ""       # BULLISH | BEARISH | NEUTRAL
    structure:     str   = ""       # BOS_UP | BOS_DOWN | CHOCH_UP | CHOCH_DOWN
    ob_level:      float = 0.0
    ob_type:       str   = ""       # BULLISH | BEARISH
    fvg_top:       float = 0.0
    fvg_bot:       float = 0.0
    fvg_type:      str   = ""
    poc:           float = 0.0
    vah:           float = 0.0
    val:           float = 0.0
    vwap:          float = 0.0
    cvd_bias:      str   = ""       # BULLISH | BEARISH | DIVERGENCE_BULL | DIVERGENCE_BEAR
    liq_swept:     bool  = False
    liq_level:     float = 0.0
    reasons:       list  = field(default_factory=list)
    timestamp:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_trade(self) -> bool:
        return self.action in ("BUY", "SELL")

    def summary(self) -> str:
        if not self.is_trade():
            return (f"WAIT | score={self.score}/{self.max_score} | "
                    f"HTF={self.htf_bias} | KZ={self.killzone or 'none'}")
        r = (f"{self.action} {self.strength} | score={self.score}/{self.max_score} "
             f"({self.confidence:.0%}) | Entry={self.entry:.2f} "
             f"SL={self.stop_loss:.2f} TP1={self.target1:.2f} TP2={self.target2:.2f} "
             f"RR=1:{self.risk_reward:.1f}\n"
             f"KZ={self.killzone} | HTF={self.htf_bias} | CVD={self.cvd_bias}\n"
             f"Reasons: {' | '.join(self.reasons)}")
        return r

    def telegram_html(self) -> str:
        import html as _h
        ts = self.timestamp.strftime("%d %b %Y  %H:%M UTC")
        if not self.is_trade():
            return (
                f"⏳ <b>WAIT — XAUUSD Order Flow</b>\n"
                f"Score: {self.score}/{self.max_score} | HTF: {self.htf_bias}\n"
                f"Killzone: {self.killzone or 'none active'}\n"
                f"<i>{_h.escape(' | '.join(self.reasons[:3]))}</i>\n"
                f"⏰ {ts}"
            )
        emoji  = "🟢" if self.action == "BUY" else "🔴"
        risk   = abs(self.entry - self.stop_loss)
        return (
            f"{emoji} <b>{self.action} {self.strength} — XAUUSD (Order Flow)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Entry:</b>  ${self.entry:.2f}\n"
            f"🛑 <b>SL:</b>     ${self.stop_loss:.2f}  ({risk:.2f} pts)\n"
            f"🎯 <b>TP1:</b>    ${self.target1:.2f}\n"
            f"🎯 <b>TP2:</b>    ${self.target2:.2f}\n"
            f"🎯 <b>TP3:</b>    ${self.target3:.2f}\n"
            f"📊 <b>R:R:</b>    1:{self.risk_reward:.1f}  |  Score: {self.score}/{self.max_score} ({self.confidence:.0%})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏗 <b>Structure:</b> {self.structure}\n"
            f"📦 <b>Order Block:</b> {self.ob_type} OB @ {self.ob_level:.2f}\n"
            f"🕳 <b>FVG:</b> {self.fvg_type} [{self.fvg_bot:.2f}–{self.fvg_top:.2f}]\n"
            f"📈 <b>Volume Profile:</b> POC={self.poc:.2f} VAH={self.vah:.2f} VAL={self.val:.2f}\n"
            f"🌊 <b>CVD:</b> {self.cvd_bias}\n"
            f"🕐 <b>Killzone:</b> {self.killzone}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🧠 <i>{_h.escape(' | '.join(self.reasons))}</i>\n"
            f"⏰ {ts}"
        )


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch(interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV for GC=F (Gold Futures) and normalise columns."""
    import yfinance as yf
    raw = yf.download("GC=F", period=period, interval=interval,
                      progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw.reset_index()
    # Rename index col
    for c in raw.columns:
        if c.lower() in ("datetime","date"):
            raw.rename(columns={c: "timestamp"}, inplace=True)
            break
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.dropna(subset=["open","high","low","close"])
    return raw.sort_values("timestamp").reset_index(drop=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _get_killzone(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    hour = ts.hour
    for name, (start, end) in KILLZONES.items():
        if start <= hour < end:
            return name
    return ""


def _prep_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Return clean OHLC with lowercase cols for SMC lib."""
    cols = {c: c.lower() for c in df.columns}
    out  = df.rename(columns=cols)[["open","high","low","close","volume"]].copy()
    return out.reset_index(drop=True)


# ── Volume Profile ─────────────────────────────────────────────────────────────

def compute_volume_profile(df: pd.DataFrame, bins: int = VOL_PROFILE_BINS) -> dict:
    """
    Compute POC, VAH, VAL, HVN, LVN from OHLCV data.
    Uses price-bucket histogram weighted by volume (approximate VP).
    """
    result = {"poc": 0.0, "vah": 0.0, "val": 0.0, "hvn": [], "lvn": []}
    if df.empty or "volume" not in df.columns:
        return result

    try:
        if MP_AVAILABLE:
            mp_df = df.set_index("timestamp")[["open","high","low","close","volume"]].copy()
            mp_df.columns = ["Open","High","Low","Close","Volume"]
            mp = MarketProfile(mp_df)
            sl  = mp[mp_df.index[0]:mp_df.index[-1]]
            result["poc"] = float(sl.poc_price)
            val, vah      = sl.value_area
            result["vah"] = float(vah)
            result["val"] = float(val)
            try:
                hvn = sl.high_value_nodes
                lvn = sl.low_value_nodes
                result["hvn"] = [float(p) for p in hvn.index.tolist()[:5]]
                result["lvn"] = [float(p) for p in lvn.index.tolist()[:5]]
            except Exception:
                pass
        else:
            # Manual histogram fallback
            lo = df["low"].min()
            hi = df["high"].max()
            buckets = np.linspace(lo, hi, bins + 1)
            vol_dist = np.zeros(bins)
            for _, row in df.iterrows():
                lo_r = row["low"]; hi_r = row["high"]
                vol  = row.get("volume", 0) or 0
                mask = (buckets[1:] >= lo_r) & (buckets[:-1] <= hi_r)
                n    = mask.sum()
                if n > 0:
                    vol_dist[mask] += vol / n
            poc_idx       = int(np.argmax(vol_dist))
            result["poc"] = float((buckets[poc_idx] + buckets[poc_idx+1]) / 2)
            total = vol_dist.sum()
            target = total * 0.70
            acc   = 0.0
            lo_i  = poc_idx; hi_i = poc_idx
            while acc < target and (lo_i > 0 or hi_i < bins - 1):
                lo_add = vol_dist[lo_i-1] if lo_i > 0 else 0
                hi_add = vol_dist[hi_i+1] if hi_i < bins-1 else 0
                if lo_add >= hi_add and lo_i > 0:
                    lo_i -= 1; acc += lo_add
                elif hi_i < bins - 1:
                    hi_i += 1; acc += hi_add
                else:
                    lo_i -= 1; acc += lo_add
            result["val"] = float((buckets[lo_i] + buckets[lo_i+1]) / 2)
            result["vah"] = float((buckets[hi_i] + buckets[hi_i+1]) / 2)

    except Exception as e:
        logger.debug(f"[VP] Error: {e}")

    return result


# ── VWAP ───────────────────────────────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame) -> dict:
    """Session VWAP + 1σ/2σ bands. Resets at midnight UTC."""
    out = {"vwap": 0.0, "vwap_upper1": 0.0, "vwap_lower1": 0.0,
           "vwap_upper2": 0.0, "vwap_lower2": 0.0, "above_vwap": False}
    if df.empty:
        return out
    df = df.copy()
    df["session_day"] = df["timestamp"].dt.date
    df["tp"]  = (df["high"] + df["low"] + df["close"]) / 3
    df["pvol"]= df["tp"] * df["volume"]

    today = df["session_day"].max()
    d     = df[df["session_day"] == today].copy()
    if d.empty:
        d = df.tail(50)
    cum_pvol = d["pvol"].cumsum()
    cum_vol  = d["volume"].cumsum().replace(0, np.nan)
    vwap_s   = cum_pvol / cum_vol
    vwap_now = float(vwap_s.iloc[-1])

    # Standard deviation bands
    variance = ((d["tp"] - vwap_s)**2 * d["volume"]).cumsum() / cum_vol
    std      = np.sqrt(variance)
    std_now  = float(std.iloc[-1])

    out["vwap"]        = vwap_now
    out["vwap_upper1"] = vwap_now + std_now
    out["vwap_lower1"] = vwap_now - std_now
    out["vwap_upper2"] = vwap_now + 2 * std_now
    out["vwap_lower2"] = vwap_now - 2 * std_now
    out["above_vwap"]  = float(d["close"].iloc[-1]) > vwap_now

    return out


# ── Cumulative Volume Delta ─────────────────────────────────────────────────────

def compute_cvd(df: pd.DataFrame, lookback: int = CVD_LOOKBACK) -> dict:
    """
    Approximate CVD from OHLCV:
      Bullish candle (close > open) → vol is buy vol
      Bearish candle (close < open) → vol is sell vol
      Wick analysis for partial buy/sell split
    Returns: cvd series, current bias, divergence flag.
    """
    out = {"cvd_now": 0.0, "cvd_delta": 0.0, "bias": "NEUTRAL",
           "divergence": False, "divergence_type": ""}
    if df.empty:
        return out
    df = df.tail(max(lookback * 2, 60)).copy()

    # Buy/sell vol per bar using candle body + wick analysis
    buy_vol  = []
    sell_vol = []
    for _, r in df.iterrows():
        o,h,l,c = r["open"], r["high"], r["low"], r["close"]
        v = r.get("volume", 0) or 0
        rng = h - l
        if rng < 1e-9:
            buy_vol.append(v * 0.5); sell_vol.append(v * 0.5)
            continue
        # body ratio
        body_pct = (c - o) / rng   # positive = bullish
        # upper wick (selling pressure at top), lower wick (buying at bottom)
        upper_wick = (h - max(o, c)) / rng
        lower_wick = (min(o, c) - l) / rng
        buy_ratio  = max(0.1, min(0.9, 0.5 + body_pct * 0.4 - upper_wick * 0.2 + lower_wick * 0.2))
        buy_vol.append(v * buy_ratio)
        sell_vol.append(v * (1 - buy_ratio))

    df["buy_vol"]  = buy_vol
    df["sell_vol"] = sell_vol
    df["delta"]    = df["buy_vol"] - df["sell_vol"]
    df["cvd"]      = df["delta"].cumsum()

    cvd_now   = float(df["cvd"].iloc[-1])
    cvd_prev  = float(df["cvd"].iloc[-lookback])
    price_now = float(df["close"].iloc[-1])
    price_prev= float(df["close"].iloc[-lookback])

    out["cvd_now"]   = cvd_now
    out["cvd_delta"] = cvd_now - cvd_prev

    # Bias
    out["bias"] = "BULLISH" if cvd_now > cvd_prev else "BEARISH"

    # Divergence: price direction vs CVD direction differ
    price_up = price_now > price_prev
    cvd_up   = cvd_now  > cvd_prev
    if price_up and not cvd_up:
        out["divergence"]      = True
        out["divergence_type"] = "BEARISH_DIV"   # price up, sellers absorbing = likely reversal down
    elif not price_up and cvd_up:
        out["divergence"]      = True
        out["divergence_type"] = "BULLISH_DIV"   # price down, buyers absorbing = likely reversal up

    return out


# ── SMC Layer ──────────────────────────────────────────────────────────────────

def _safe_smc_call(fn, *args, **kwargs):
    """Call SMC function; return empty DataFrame on error."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.debug(f"[SMC] {fn.__name__}: {e}")
        return pd.DataFrame()


def get_smc_analysis(df: pd.DataFrame) -> dict:
    """Run all SMC indicators on a single timeframe."""
    result = {
        "swings": pd.DataFrame(), "fvg": pd.DataFrame(),
        "ob": pd.DataFrame(), "bos_choch": pd.DataFrame(),
        "liquidity": pd.DataFrame(), "prev_hl": pd.DataFrame(),
        "retracements": pd.DataFrame(),
    }
    if not SMC_AVAILABLE or df.empty or len(df) < 30:
        return result

    ohlc = _prep_ohlc(df)

    # Swing highs/lows (prerequisite for most SMC indicators)
    swings = _safe_smc_call(smc.swing_highs_lows, ohlc, swing_length=SWING_LENGTH)
    result["swings"] = swings

    if swings.empty:
        return result

    # FVG
    result["fvg"]         = _safe_smc_call(smc.fvg, ohlc, join_consecutive=FVG_JOIN)
    # Order Blocks
    result["ob"]          = _safe_smc_call(smc.ob, ohlc, swings, close_mitigation=OB_CLOSE_MIT)
    # BOS / CHoCH
    result["bos_choch"]   = _safe_smc_call(smc.bos_choch, ohlc, swings, close_break=True)
    # Liquidity
    result["liquidity"]   = _safe_smc_call(smc.liquidity, ohlc, swings, range_percent=LIQ_RANGE_PCT)
    # Previous high/low
    result["prev_hl"]     = _safe_smc_call(smc.previous_high_low, ohlc, time_frame="1D")
    # Retracements
    result["retracements"]= _safe_smc_call(smc.retracements, ohlc, swings)

    return result


def extract_last_bos_choch(bc_df: pd.DataFrame) -> tuple[str, float]:
    """Return (type, level) of the most recent BOS or CHoCH."""
    if bc_df.empty:
        return "", 0.0
    for i in range(len(bc_df)-1, -1, -1):
        row = bc_df.iloc[i]
        bos   = row.get("BOS", 0)
        choch = row.get("CHOCH", 0)
        level = row.get("Level", 0.0)
        if bos == 1:   return "BOS_UP",   float(level or 0)
        if bos == -1:  return "BOS_DOWN",  float(level or 0)
        if choch == 1: return "CHOCH_UP",  float(level or 0)
        if choch == -1:return "CHOCH_DOWN",float(level or 0)
    return "", 0.0


def extract_nearest_ob(ob_df: pd.DataFrame, price: float, direction: str) -> dict:
    """Find the nearest unmitigated order block in the trade direction."""
    empty = {"type": "", "top": 0.0, "bot": 0.0, "mid": 0.0, "pct": 0.0}
    if ob_df.empty:
        return empty
    for i in range(len(ob_df)-1, -1, -1):
        row = ob_df.iloc[i]
        ob  = row.get("OB", 0)
        top = float(row.get("Top", 0) or 0)
        bot = float(row.get("Bottom", 0) or 0)
        pct = float(row.get("Percentage", 0) or 0)
        if top == 0 or bot == 0:
            continue
        mid = (top + bot) / 2
        # Bullish OB: price trading down into it from above → BUY setup
        if ob == 1 and direction == "BUY" and bot <= price <= top * 1.005:
            return {"type": "BULLISH", "top": top, "bot": bot, "mid": mid, "pct": pct}
        # Bearish OB: price trading up into it from below → SELL setup
        if ob == -1 and direction == "SELL" and bot * 0.995 <= price <= top:
            return {"type": "BEARISH", "top": top, "bot": bot, "mid": mid, "pct": pct}
    return empty


def extract_nearest_fvg(fvg_df: pd.DataFrame, price: float, direction: str) -> dict:
    """Find nearest unmitigated FVG."""
    empty = {"type": "", "top": 0.0, "bot": 0.0}
    if fvg_df.empty:
        return empty
    for i in range(len(fvg_df)-1, -1, -1):
        row = fvg_df.iloc[i]
        fvg = row.get("FVG", 0)
        top = float(row.get("Top", 0) or 0)
        bot = float(row.get("Bottom", 0) or 0)
        mit = row.get("MitigatedIndex", None)
        if pd.isna(mit) or mit is None:
            # Unmitigated FVG
            if fvg == 1 and direction == "BUY" and bot <= price <= top:
                return {"type": "BULLISH", "top": top, "bot": bot}
            if fvg == -1 and direction == "SELL" and bot <= price <= top:
                return {"type": "BEARISH", "top": top, "bot": bot}
    return empty


def check_liquidity_swept(liq_df: pd.DataFrame, price: float, direction: str) -> tuple[bool, float]:
    """Was liquidity swept recently (last 5 bars)?"""
    if liq_df.empty:
        return False, 0.0
    recent = liq_df.tail(5)
    for i in range(len(recent)-1, -1, -1):
        row   = recent.iloc[i]
        liq   = row.get("Liquidity", 0)
        swept = row.get("Swept", None)
        level = float(row.get("Level", 0) or 0)
        if pd.notna(swept) and swept is not None:
            # Bullish liquidity sweep (bear trap → BUY)
            if liq == 1 and direction == "BUY":
                return True, level
            # Bearish liquidity sweep (bull trap → SELL)
            if liq == -1 and direction == "SELL":
                return True, level
    return False, 0.0


# ── HTF Bias ───────────────────────────────────────────────────────────────────

def get_htf_bias(df_4h: pd.DataFrame) -> str:
    """
    Derive 4H macro bias from:
      - Last BOS/CHoCH direction
      - Price vs EMA21/55 on 4H
    """
    if df_4h.empty or len(df_4h) < 55:
        return "NEUTRAL"
    smc_4h = get_smc_analysis(df_4h)
    struct, _ = extract_last_bos_choch(smc_4h["bos_choch"])

    close = df_4h["close"].values
    ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False).mean().values

    ema_bull = close[-1] > ema21[-1] > ema55[-1]
    ema_bear = close[-1] < ema21[-1] < ema55[-1]

    if "UP" in struct or "CHOCH_UP" in struct:
        if ema_bull:  return "BULLISH"
        if ema_bear:  return "NEUTRAL"
        return "BULLISH"
    if "DOWN" in struct or "CHOCH_DOWN" in struct:
        if ema_bear:  return "BEARISH"
        if ema_bull:  return "NEUTRAL"
        return "BEARISH"
    if ema_bull:  return "BULLISH"
    if ema_bear:  return "BEARISH"
    return "NEUTRAL"


# ── Session Asian Range ─────────────────────────────────────────────────────────

def get_asian_range(df_1h: pd.DataFrame) -> tuple[float, float]:
    """Return today's Asian session (00:00-07:00 UTC) high/low."""
    now   = datetime.now(timezone.utc)
    today = now.date()
    asian = df_1h[
        (df_1h["timestamp"].dt.date == today) &
        (df_1h["timestamp"].dt.hour < 7)
    ]
    if asian.empty:
        return 0.0, 0.0
    return float(asian["high"].max()), float(asian["low"].min())


# ── Main Scoring Engine ─────────────────────────────────────────────────────────

def _score_direction(
    direction: str,
    price: float,
    atr: float,
    smc_15m: dict,
    smc_1h: dict,
    smc_4h: dict,
    smc_4h_struct: str,
    htf_bias: str,
    vp: dict,
    vwap_data: dict,
    cvd: dict,
    tape: dict,
    killzone: str,
    asian_hi: float,
    asian_lo: float,
    df_15m: pd.DataFrame = None,
    df_1h: pd.DataFrame = None,
    df_4h: pd.DataFrame = None,
) -> tuple[int, list[str]]:
    """Score a single direction (BUY/SELL). Returns (score, reasons). Max = 40 pts."""
    score   = 0
    reasons = []
    is_buy  = (direction == "BUY")

    df15  = df_15m if df_15m is not None else pd.DataFrame()
    df1h  = df_1h  if df_1h  is not None else pd.DataFrame()
    df4h  = df_4h  if df_4h  is not None else pd.DataFrame()

    # ── 1. HTF Structure alignment (0-5) ──────────────────────────────────────
    if htf_bias == ("BULLISH" if is_buy else "BEARISH"):
        score += 3; reasons.append(f"4H bias {htf_bias}")
    elif htf_bias == "NEUTRAL":
        score += 1
    if ("BOS_UP" in smc_4h_struct and is_buy) or ("BOS_DOWN" in smc_4h_struct and not is_buy):
        score += 1; reasons.append(f"4H {smc_4h_struct}")
    # 1H structure alignment
    struct_1h, _ = extract_last_bos_choch(smc_1h.get("bos_choch", pd.DataFrame()))
    if ("UP" in struct_1h and is_buy) or ("DOWN" in struct_1h and not is_buy):
        score += 1; reasons.append(f"1H {struct_1h} aligned")

    # ── 2. Order Block (0-4) ──────────────────────────────────────────────────
    ob_1h  = extract_nearest_ob(smc_1h.get("ob",  pd.DataFrame()), price, direction)
    ob_15m = extract_nearest_ob(smc_15m.get("ob", pd.DataFrame()), price, direction)
    ob_4h  = extract_nearest_ob(smc_4h.get("ob",  pd.DataFrame()), price, direction)
    if ob_4h["type"]:
        score += 2; reasons.append(f"4H {ob_4h['type']} OB @ {ob_4h['mid']:.2f}")
    elif ob_1h["type"]:
        score += 2; reasons.append(f"1H {ob_1h['type']} OB @ {ob_1h['mid']:.2f}")
    if ob_15m["type"]:
        score += 1; reasons.append(f"15m OB confluence @ {ob_15m['mid']:.2f}")
    if ob_4h["type"] and ob_1h["type"]:
        score += 1; reasons.append("4H+1H OB stack — high confluence zone")

    # ── 3. Fair Value Gap (0-4) ───────────────────────────────────────────────
    fvg_15m = extract_nearest_fvg(smc_15m.get("fvg", pd.DataFrame()), price, direction)
    fvg_1h  = extract_nearest_fvg(smc_1h.get("fvg",  pd.DataFrame()), price, direction)
    if fvg_15m["type"]:
        score += 2; reasons.append(f"15m FVG fill [{fvg_15m['bot']:.1f}-{fvg_15m['top']:.1f}]")
    if fvg_1h["type"]:
        score += 1; reasons.append(f"1H FVG confluence [{fvg_1h['bot']:.1f}-{fvg_1h['top']:.1f}]")
    # HTF FVG (4H/1H)
    htf_fvg = get_htf_fvg(smc_1h.get("fvg", pd.DataFrame()), smc_4h.get("fvg", pd.DataFrame()), price, atr)
    if htf_fvg.get("type"):
        right_dir = (htf_fvg["type"]=="BULLISH" and is_buy) or (htf_fvg["type"]=="BEARISH" and not is_buy)
        if right_dir:
            score += 1; reasons.append(f"{htf_fvg['tf']} FVG backing trade [{htf_fvg['bot']:.1f}-{htf_fvg['top']:.1f}]")
    # Inversion FVG
    if not df15.empty:
        inv_fvg = get_inversion_fvg(smc_15m.get("fvg", pd.DataFrame()), price, df15)
        if (inv_fvg["type"]=="INVERSION_SUPPORT" and is_buy) or (inv_fvg["type"]=="INVERSION_RESIST" and not is_buy):
            score += 1; reasons.append(f"Inversion FVG acting as {'support' if is_buy else 'resistance'}")

    # ── 4. Liquidity sweep (0-3) ──────────────────────────────────────────────
    swept_15m, liq_lvl_15 = check_liquidity_swept(smc_15m.get("liquidity", pd.DataFrame()), price, direction)
    swept_1h,  liq_lvl_1h = check_liquidity_swept(smc_1h.get("liquidity",  pd.DataFrame()), price, direction)
    liq_lvl = liq_lvl_15 or liq_lvl_1h
    if swept_15m and swept_1h:
        score += 3; reasons.append(f"Multi-TF liquidity sweep @ {liq_lvl:.2f} — strong reversal setup")
    elif swept_15m or swept_1h:
        score += 2; reasons.append(f"Liquidity swept @ {liq_lvl:.2f}")

    # ── 5. Equal Highs / Lows (0-2) ───────────────────────────────────────────
    if not df15.empty:
        eqhl = detect_equal_highs_lows(df15)
        if is_buy and eqhl.get("eql_swept") and eqhl["eql"] > 0:
            score += 2; reasons.append(f"EQL {eqhl['eql']:.2f} swept — bear trap (bulls incoming)")
        elif not is_buy and eqhl.get("eqh_swept") and eqhl["eqh"] > 0:
            score += 2; reasons.append(f"EQH {eqhl['eqh']:.2f} swept — bull trap (bears incoming)")
        elif is_buy and eqhl.get("eql") and abs(price - eqhl["eql"]) < atr * 0.5:
            score += 1; reasons.append(f"EQL support @ {eqhl['eql']:.2f}")
        elif not is_buy and eqhl.get("eqh") and abs(price - eqhl["eqh"]) < atr * 0.5:
            score += 1; reasons.append(f"EQH resistance @ {eqhl['eqh']:.2f}")

    # ── 6. OTE (Optimal Trade Entry) (0-3) ────────────────────────────────────
    if not df15.empty:
        sh, sl_sw = get_swing_range(smc_15m.get("swings", pd.DataFrame()), df15, lookback=30)
        ote_lo, ote_hi = get_ote_zone(sh, sl_sw, direction)
        if ote_lo and ote_hi and ote_lo <= price <= ote_hi:
            score += 3; reasons.append(f"OTE zone [{ote_lo:.2f}-{ote_hi:.2f}] — ICT 61.8-70.5% retracement")
        elif ote_lo and ote_hi and abs(price - (ote_lo + ote_hi) / 2) < atr * 0.5:
            score += 1; reasons.append(f"Near OTE zone [{ote_lo:.2f}-{ote_hi:.2f}]")

    # ── 7. Premium / Discount zone (0-2) ──────────────────────────────────────
    if not df4h.empty:
        sh4, sl4 = get_swing_range(smc_4h.get("swings", pd.DataFrame()), df4h, lookback=40)
        pd_zone  = get_premium_discount(sh4, sl4, price)
        if is_buy and pd_zone == "DISCOUNT":
            score += 2; reasons.append(f"Price in DISCOUNT zone (below 50% of 4H range) — cheap")
        elif not is_buy and pd_zone == "PREMIUM":
            score += 2; reasons.append(f"Price in PREMIUM zone (above 50% of 4H range) — expensive")
        elif pd_zone == "EQUILIBRIUM":
            score += 1

    # ── 8. Displacement confirmation (0-2) ───────────────────────────────────
    if not df15.empty:
        disp = detect_displacement(df15, atr)
        if disp["displaced"]:
            if (disp["direction"] == "UP" and is_buy) or (disp["direction"] == "DOWN" and not is_buy):
                score += 2; reasons.append(f"Displacement candle {disp['direction']} — institutional delivery")
            else:
                score += 0  # against us — slight negative not penalised

    # ── 9. Previous Day / Week levels (0-2) ──────────────────────────────────
    if not df1h.empty:
        prev = get_prev_day_levels(df1h)
        tol  = atr * 0.5
        if is_buy:
            if prev["pdl"] and abs(price - prev["pdl"]) < tol:
                score += 1; reasons.append(f"Previous Day Low {prev['pdl']:.2f} as support")
            if prev["pwl"] and abs(price - prev["pwl"]) < tol:
                score += 1; reasons.append(f"Previous Week Low {prev['pwl']:.2f} as support")
        else:
            if prev["pdh"] and abs(price - prev["pdh"]) < tol:
                score += 1; reasons.append(f"Previous Day High {prev['pdh']:.2f} as resistance")
            if prev["pwh"] and abs(price - prev["pwh"]) < tol:
                score += 1; reasons.append(f"Previous Week High {prev['pwh']:.2f} as resistance")

    # ── 10. CVD (0-3) ─────────────────────────────────────────────────────────
    cvd_bias = cvd.get("bias", "NEUTRAL")
    div_type = cvd.get("divergence_type", "")
    if cvd.get("divergence"):
        if "BULLISH_DIV" in div_type and is_buy:
            score += 3; reasons.append("CVD bullish divergence — buyers absorbing (BIG signal)")
        elif "BEARISH_DIV" in div_type and not is_buy:
            score += 3; reasons.append("CVD bearish divergence — sellers absorbing (BIG signal)")
    elif cvd_bias == ("BULLISH" if is_buy else "BEARISH"):
        score += 1; reasons.append(f"CVD {cvd_bias} bias aligned")

    # ── 11. Tape bias (0-3) ──────────────────────────────────────────────────
    tape_bias = tape.get("tape_bias", "NEUTRAL")
    if tape_bias == ("STRONGLY_BULLISH" if is_buy else "STRONGLY_BEARISH"):
        score += 3; reasons.append(f"Tape STRONGLY {'BULLISH' if is_buy else 'BEARISH'} — buyers/sellers dominating")
    elif tape_bias == ("BULLISH" if is_buy else "BEARISH"):
        score += 2; reasons.append(f"Tape {tape_bias} — aligned with direction")
    if tape.get("climax") and tape.get("climax_type"):
        if (tape["climax_type"]=="SELLING_CLIMAX" and is_buy) or (tape["climax_type"]=="BUYING_CLIMAX" and not is_buy):
            score += 1; reasons.append(f"Tape: {tape['climax_type']} — exhaustion reversal")
    if tape.get("absorption") and tape.get("absorption_side"):
        if (tape["absorption_side"]=="BULL_ABSORB" and is_buy) or (tape["absorption_side"]=="BEAR_ABSORB" and not is_buy):
            score += 1; reasons.append(f"Tape: {tape['absorption_side']} — institutions absorbing")

    # ── 12. Volume Profile (0-3) ─────────────────────────────────────────────
    poc = vp.get("poc", 0); vah = vp.get("vah", 0); val = vp.get("val", 0)
    if poc:
        tol_vp = atr * 0.4
        if abs(price - poc) < tol_vp:
            score += 1; reasons.append(f"At POC {poc:.2f} — highest volume magnet")
        if is_buy and val and abs(price - val) < tol_vp:
            score += 2; reasons.append(f"At VAL {val:.2f} — 70% value area low support")
        elif not is_buy and vah and abs(price - vah) < tol_vp:
            score += 2; reasons.append(f"At VAH {vah:.2f} — 70% value area high resistance")

    # ── 13. Killzone timing (0-3) ────────────────────────────────────────────
    if killzone == "NY Silver Bullet":
        score += 3; reasons.append("NY Silver Bullet (15:00-16:00 UTC) — highest probability 60-min window")
    elif killzone in ("London KZ", "NY AM"):
        score += 2; reasons.append(f"In {killzone} — institutional trading window")
    elif killzone == "Asian Range":
        if asian_hi and asian_lo:
            near_hi = abs(price - asian_hi) < atr * 0.5
            near_lo = abs(price - asian_lo) < atr * 0.5
            if not is_buy and near_hi:
                score += 1; reasons.append(f"Fading Asian range high {asian_hi:.2f}")
            if is_buy and near_lo:
                score += 1; reasons.append(f"Bouncing Asian range low {asian_lo:.2f}")

    # ── 14. VWAP (0-3) ───────────────────────────────────────────────────────
    vwap = vwap_data.get("vwap", 0)
    vu2  = vwap_data.get("vwap_upper2", 0)
    vl2  = vwap_data.get("vwap_lower2", 0)
    vu1  = vwap_data.get("vwap_upper1", 0)
    vl1  = vwap_data.get("vwap_lower1", 0)
    if vwap:
        above = price > vwap
        if is_buy and above:
            score += 1; reasons.append(f"Above VWAP {vwap:.2f} — bulls in control")
        elif not is_buy and not above:
            score += 1; reasons.append(f"Below VWAP {vwap:.2f} — bears in control")
        if is_buy and vl2 and price < vl2:
            score += 2; reasons.append(f"At VWAP -2σ {vl2:.2f} — statistically oversold, mean reversion")
        elif not is_buy and vu2 and price > vu2:
            score += 2; reasons.append(f"At VWAP +2σ {vu2:.2f} — statistically overbought, mean reversion")
        elif is_buy and vl1 and price < vl1:
            score += 1; reasons.append(f"At VWAP -1σ {vl1:.2f} — value buy zone")
        elif not is_buy and vu1 and price > vu1:
            score += 1; reasons.append(f"At VWAP +1σ {vu1:.2f} — value sell zone")

    return score, reasons


# ── Advanced SMC Helpers ───────────────────────────────────────────────────────

def get_ote_zone(swing_high: float, swing_low: float, direction: str) -> tuple[float, float]:
    """
    Optimal Trade Entry zone (ICT): 61.8%-70.5% Fibonacci retracement.
    BUY: pullback into 61.8-70.5% of the bullish swing
    SELL: pullback into 61.8-70.5% of the bearish swing
    """
    if swing_high <= swing_low:
        return 0.0, 0.0
    rng = swing_high - swing_low
    if direction == "BUY":
        # Retracement of bullish swing: price pulls back to 61.8-70.5% from top
        ote_hi = swing_high - rng * OTE_SHALLOW
        ote_lo = swing_high - rng * OTE_DEEP
    else:
        # Retracement of bearish swing: price bounces to 61.8-70.5% from bottom
        ote_lo = swing_low + rng * OTE_SHALLOW
        ote_hi = swing_low + rng * OTE_DEEP
    return round(ote_lo, 2), round(ote_hi, 2)


def get_premium_discount(swing_high: float, swing_low: float, price: float) -> str:
    """
    Premium zone (above 50% of range) = expensive = SELL
    Discount zone (below 50% of range) = cheap = BUY
    Equilibrium (45-55%) = neutral
    """
    if swing_high <= swing_low:
        return "NEUTRAL"
    mid = (swing_high + swing_low) / 2
    pct = (price - swing_low) / (swing_high - swing_low)
    if pct > 0.55:   return "PREMIUM"
    elif pct < 0.45: return "DISCOUNT"
    return "EQUILIBRIUM"


def detect_equal_highs_lows(df: pd.DataFrame, tolerance_pct: float = 0.001) -> dict:
    """
    Equal Highs (EQH) = double/triple top → liquidity pool above (bulls stop-hunted)
    Equal Lows (EQL)  = double/triple bottom → liquidity pool below (bears stop-hunted)
    Returns: {eqh: price, eql: price, eqh_swept: bool, eql_swept: bool}
    """
    result = {"eqh": 0.0, "eql": 0.0, "eqh_swept": False, "eql_swept": False}
    if len(df) < 20:
        return result

    highs  = df["high"].values
    lows   = df["low"].values
    close  = float(df["close"].iloc[-1])

    # Find equal highs: recent highs within 0.1% of each other
    recent_highs = highs[-50:]
    for i in range(len(recent_highs) - 1):
        for j in range(i + 1, len(recent_highs)):
            if abs(recent_highs[i] - recent_highs[j]) / recent_highs[i] < tolerance_pct:
                result["eqh"] = round(float(max(recent_highs[i], recent_highs[j])), 2)
                # Was it swept (price closed above then came back)?
                result["eqh_swept"] = close < result["eqh"] * (1 - tolerance_pct * 2)
                break
        if result["eqh"]:
            break

    recent_lows = lows[-50:]
    for i in range(len(recent_lows) - 1):
        for j in range(i + 1, len(recent_lows)):
            if abs(recent_lows[i] - recent_lows[j]) / recent_lows[i] < tolerance_pct:
                result["eql"] = round(float(min(recent_lows[i], recent_lows[j])), 2)
                result["eql_swept"] = close > result["eql"] * (1 + tolerance_pct * 2)
                break
        if result["eql"]:
            break

    return result


def detect_displacement(df: pd.DataFrame, atr: float) -> dict:
    """
    Displacement = an aggressive impulsive move leaving FVGs behind.
    Characteristics: large body (>1.5x ATR), closes above/below key levels,
    often followed by retracement back to OB/FVG.
    """
    result = {"displaced": False, "direction": "", "displacement_high": 0.0, "displacement_low": 0.0}
    if len(df) < 5:
        return result

    # Check last 3 bars for displacement candle
    for i in range(-3, 0):
        row  = df.iloc[i]
        body = abs(float(row["close"]) - float(row["open"]))
        if body >= atr * 1.5:
            direction = "UP" if float(row["close"]) > float(row["open"]) else "DOWN"
            result["displaced"]         = True
            result["direction"]         = direction
            result["displacement_high"] = float(row["high"])
            result["displacement_low"]  = float(row["low"])
            break

    return result


def get_prev_day_levels(df_1h: pd.DataFrame) -> dict:
    """Previous day High/Low/Close — key S/R levels."""
    result = {"pdh": 0.0, "pdl": 0.0, "pdc": 0.0, "pwh": 0.0, "pwl": 0.0}
    if df_1h.empty:
        return result
    df_1h = df_1h.copy()
    df_1h["date"] = df_1h["timestamp"].dt.date
    by_day = df_1h.groupby("date")
    dates  = sorted(by_day.groups.keys())

    if len(dates) >= 2:
        prev_day = by_day.get_group(dates[-2])
        result["pdh"] = float(prev_day["high"].max())
        result["pdl"] = float(prev_day["low"].min())
        result["pdc"] = float(prev_day["close"].iloc[-1])

    # Previous week H/L (last 5+ trading days)
    if len(dates) >= 6:
        week_data = df_1h[df_1h["date"].isin(dates[-6:-1])]
        result["pwh"] = float(week_data["high"].max())
        result["pwl"] = float(week_data["low"].min())

    return result


def get_inversion_fvg(fvg_df: pd.DataFrame, price: float, df: pd.DataFrame) -> dict:
    """
    Inversion FVG: a mitigated FVG that flips polarity.
    - Bullish FVG filled → now acts as resistance (bearish)
    - Bearish FVG filled → now acts as support (bullish)
    """
    result = {"type": "", "top": 0.0, "bot": 0.0}
    if fvg_df.empty:
        return result
    current_price = float(df["close"].iloc[-1])

    for i in range(len(fvg_df) - 1, max(len(fvg_df) - 30, -1), -1):
        row = fvg_df.iloc[i]
        fvg = row.get("FVG", 0)
        top = float(row.get("Top", 0) or 0)
        bot = float(row.get("Bottom", 0) or 0)
        mit = row.get("MitigatedIndex", None)

        if top == 0 or bot == 0 or pd.isna(mit) or mit is None:
            continue

        # Mitigated FVG — check if price is retesting it
        mid = (top + bot) / 2
        if abs(current_price - mid) < (top - bot) * 1.5:
            if fvg == 1:   # was bullish, now flipped = resistance
                result = {"type": "INVERSION_RESIST", "top": top, "bot": bot}
            elif fvg == -1:
                result = {"type": "INVERSION_SUPPORT", "top": top, "bot": bot}
            break
    return result


def get_htf_fvg(fvg_1h: pd.DataFrame, fvg_4h: pd.DataFrame, price: float, atr: float) -> dict:
    """Check if price is inside a 1H or 4H FVG (higher confluence)."""
    tol = atr * 1.0
    for fvg_df, label in [(fvg_4h, "4H"), (fvg_1h, "1H")]:
        if fvg_df.empty:
            continue
        for i in range(len(fvg_df)-1, max(len(fvg_df)-20,-1), -1):
            row = fvg_df.iloc[i]
            fvg = row.get("FVG", 0)
            top = float(row.get("Top", 0) or 0)
            bot = float(row.get("Bottom", 0) or 0)
            mit = row.get("MitigatedIndex", None)
            if top == 0 or bot == 0 or not (pd.isna(mit) or mit is None):
                continue
            if bot - tol <= price <= top + tol:
                t = "BULLISH" if fvg == 1 else "BEARISH"
                return {"type": t, "tf": label, "top": top, "bot": bot}
    return {}


def get_swing_range(swings_df: pd.DataFrame, df: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    """Get recent swing high and swing low for OTE / P/D calculation."""
    if swings_df.empty:
        tail = df.tail(lookback)
        return float(tail["high"].max()), float(tail["low"].min())

    highs = []
    lows  = []
    swing_tail = swings_df.tail(lookback * 2)
    for i, row in swing_tail.iterrows():
        hl = row.get("HighLow", 0)
        lv = row.get("Level", 0)
        if hl == 1 and lv:  highs.append(float(lv))
        if hl == -1 and lv: lows.append(float(lv))

    sh = max(highs) if highs else float(df.tail(lookback)["high"].max())
    sl = min(lows)  if lows  else float(df.tail(lookback)["low"].min())
    return sh, sl


# ── SL / TP Calculation ────────────────────────────────────────────────────────

def _calc_sl_tp(direction: str, price: float, atr: float,
                ob: dict, fvg: dict, vp: dict, asian_hi: float, asian_lo: float,
                liq_level: float) -> tuple[float, float, float, float]:
    """
    Calculate SL and 3 TPs based on:
      SL: below/above OB extreme + 0.3×ATR buffer
      TP1: 1.5× risk (first partial exit)
      TP2: 2.5× risk OR next significant level
      TP3: 4× risk OR major structure level
    """
    is_buy = (direction == "BUY")

    # SL placement
    if ob["type"]:
        sl_raw = ob["bot"] - atr * 0.3 if is_buy else ob["top"] + atr * 0.3
    elif fvg["type"]:
        sl_raw = fvg["bot"] - atr * 0.3 if is_buy else fvg["top"] + atr * 0.3
    else:
        sl_raw = price - atr * 1.5 if is_buy else price + atr * 1.5

    risk = abs(price - sl_raw)
    if risk < atr * 0.5:
        risk = atr * 0.5
        sl_raw = price - risk if is_buy else price + risk

    # TP levels
    tp1 = price + risk * 1.5  if is_buy else price - risk * 1.5
    tp2 = price + risk * 2.5  if is_buy else price - risk * 2.5
    tp3 = price + risk * 4.0  if is_buy else price - risk * 4.0

    # Snap TP2 to volume profile levels if close
    poc = vp.get("poc", 0); vah = vp.get("vah", 0); val = vp.get("val", 0)
    snap_levels = [l for l in [poc, vah, val, asian_hi, asian_lo, liq_level] if l > 0]
    for lvl in snap_levels:
        if is_buy and tp1 < lvl < tp3:
            # Snap TP2 toward this level
            if abs(lvl - tp2) < risk * 0.5:
                tp2 = lvl; break
        elif not is_buy and tp3 < lvl < tp1:
            if abs(lvl - tp2) < risk * 0.5:
                tp2 = lvl; break

    rr = abs(tp2 - price) / risk if risk > 0 else 0

    return round(sl_raw, 2), round(tp1, 2), round(tp2, 2), round(tp3, 2)


# ── Master Analysis ─────────────────────────────────────────────────────────────

def analyze_order_flow(df_15m: pd.DataFrame = None,
                        df_1h: pd.DataFrame = None,
                        df_4h: pd.DataFrame = None,
                        df_5m: pd.DataFrame = None) -> OFSignal:
    """
    Full order flow analysis. Fetches data if DataFrames not provided.
    Returns OFSignal with BUY/SELL/WAIT + full context.
    """
    logger.info("[OF] Running order flow analysis...")

    # ── Fetch data if not provided ─────────────────────────────────────────────
    if df_15m is None:
        df_15m = _fetch("15m", "5d")
    if df_1h is None:
        df_1h  = _fetch("1h",  "60d")
    if df_4h is None:
        df_4h  = _fetch("4h",  "120d")
    if df_5m is None:
        df_5m  = _fetch("5m",  "3d")

    if df_15m.empty:
        return OFSignal(action="WAIT", strength="WAIT", reasons=["No data available"])

    price = float(df_15m["close"].iloc[-1])
    ts    = df_15m["timestamp"].iloc[-1]
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()

    atr       = _atr(df_15m)
    killzone  = _get_killzone(ts)

    # ── SMC on each timeframe ──────────────────────────────────────────────────
    logger.info("[OF] Running SMC indicators...")
    smc_5m  = get_smc_analysis(df_5m)
    smc_15m = get_smc_analysis(df_15m)
    smc_1h  = get_smc_analysis(df_1h)
    smc_4h  = get_smc_analysis(df_4h)

    # ── 4H structure ──────────────────────────────────────────────────────────
    htf_bias   = get_htf_bias(df_4h)
    struct_4h, struct_level = extract_last_bos_choch(smc_4h.get("bos_choch", pd.DataFrame()))
    struct_1h, _            = extract_last_bos_choch(smc_1h.get("bos_choch", pd.DataFrame()))

    # ── Volume Profile (today's session) ──────────────────────────────────────
    logger.info("[OF] Computing volume profile...")
    today = ts.date() if hasattr(ts, "date") else datetime.now().date()
    df_today = df_15m[df_15m["timestamp"].dt.date == today]
    if df_today.empty:
        df_today = df_15m.tail(50)
    vp = compute_volume_profile(df_today)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    vwap_data = compute_vwap(df_15m)

    # ── CVD ───────────────────────────────────────────────────────────────────
    cvd = compute_cvd(df_15m)

    # ── Tape reader (5m for precision) ────────────────────────────────────────
    logger.info("[OF] Reading tape...")
    from xauusd.tape_reader import analyze_tape
    tape = analyze_tape(df_5m if not df_5m.empty else df_15m)

    # ── Asian range ────────────────────────────────────────────────────────────
    asian_hi, asian_lo = get_asian_range(df_1h)

    # ── Score both directions ──────────────────────────────────────────────────
    logger.info("[OF] Scoring BUY direction (40-pt system)...")
    buy_score,  buy_reasons  = _score_direction(
        "BUY",  price, atr, smc_15m, smc_1h, smc_4h, struct_4h, htf_bias,
        vp, vwap_data, cvd, tape, killzone, asian_hi, asian_lo,
        df_15m, df_1h, df_4h
    )
    logger.info("[OF] Scoring SELL direction (40-pt system)...")
    sell_score, sell_reasons = _score_direction(
        "SELL", price, atr, smc_15m, smc_1h, smc_4h, struct_4h, htf_bias,
        vp, vwap_data, cvd, tape, killzone, asian_hi, asian_lo,
        df_15m, df_1h, df_4h
    )

    logger.info(f"[OF] BUY={buy_score} SELL={sell_score} threshold={SCORE_BUY_THRESHOLD}/40")

    # ── Decision ──────────────────────────────────────────────────────────────
    if buy_score >= SCORE_BUY_THRESHOLD and buy_score > sell_score:
        direction = "BUY"
        score     = buy_score
        reasons   = buy_reasons
    elif sell_score >= SCORE_BUY_THRESHOLD and sell_score > buy_score:
        direction = "SELL"
        score     = sell_score
        reasons   = sell_reasons
    else:
        all_r = buy_reasons[:3] if buy_score > sell_score else sell_reasons[:3]
        return OFSignal(
            action="WAIT", strength="WAIT", score=max(buy_score, sell_score),
            max_score=40,
            entry=price, atr=atr, killzone=killzone, htf_bias=htf_bias,
            structure=struct_4h, poc=vp.get("poc",0), vah=vp.get("vah",0),
            val=vp.get("val",0), vwap=vwap_data.get("vwap",0),
            cvd_bias=tape.get("tape_bias","") or cvd.get("bias",""),
            reasons=all_r or ["Score below threshold — wait for killzone + OB/FVG confluence"],
            timestamp=ts if isinstance(ts, datetime) else datetime.now(timezone.utc),
        )

    strength = "STRONG" if score >= SCORE_STRONG_THRESHOLD else "MODERATE"

    # ── SL / TP ───────────────────────────────────────────────────────────────
    ob_used  = extract_nearest_ob(smc_1h.get("ob", pd.DataFrame()), price, direction)
    fvg_used = extract_nearest_fvg(smc_15m.get("fvg", pd.DataFrame()), price, direction)
    swept, liq_lvl = check_liquidity_swept(smc_15m.get("liquidity", pd.DataFrame()), price, direction)

    sl, tp1, tp2, tp3 = _calc_sl_tp(direction, price, atr, ob_used, fvg_used,
                                      vp, asian_hi, asian_lo, liq_lvl)
    risk = abs(price - sl)
    rr   = abs(tp2 - price) / risk if risk > 0 else 0

    return OFSignal(
        action       = direction,
        strength     = strength,
        entry        = round(price, 2),
        stop_loss    = sl,
        target1      = tp1,
        target2      = tp2,
        target3      = tp3,
        risk_reward  = round(rr, 2),
        score        = score,
        max_score    = 40,
        confidence   = round(score / 40, 2),
        atr          = round(atr, 2),
        killzone     = killzone,
        htf_bias     = htf_bias,
        structure    = struct_4h,
        ob_level     = ob_used.get("mid", 0.0),
        ob_type      = ob_used.get("type", ""),
        fvg_top      = fvg_used.get("top", 0.0),
        fvg_bot      = fvg_used.get("bot", 0.0),
        fvg_type     = fvg_used.get("type", ""),
        poc          = vp.get("poc", 0.0),
        vah          = vp.get("vah", 0.0),
        val          = vp.get("val", 0.0),
        vwap         = vwap_data.get("vwap", 0.0),
        cvd_bias     = tape.get("tape_bias","") or cvd.get("divergence_type", "") or cvd.get("bias", ""),
        liq_swept    = swept,
        liq_level    = liq_lvl,
        reasons      = reasons,
        timestamp    = ts if isinstance(ts, datetime) else datetime.now(timezone.utc),
    )


# ── Standalone test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching data and running order flow analysis...")
    sig = analyze_order_flow()
    print("\n" + "="*60)
    print(sig.summary())
    print("="*60)
