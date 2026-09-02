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

SCORE_BUY_THRESHOLD   = 11
SCORE_STRONG_THRESHOLD= 14
SWING_LENGTH          = 10      # bars for swing high/low detection
FVG_JOIN              = False
OB_CLOSE_MIT          = False
LIQ_RANGE_PCT         = 0.005   # 0.5% clustering for liquidity levels
ATR_PERIOD            = 14
CVD_LOOKBACK          = 20      # bars for CVD divergence check
VOL_PROFILE_BINS      = 100     # price buckets for volume profile

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
    smc_4h_struct: str,
    htf_bias: str,
    vp: dict,
    vwap_data: dict,
    cvd: dict,
    killzone: str,
    asian_hi: float,
    asian_lo: float,
) -> tuple[int, list[str]]:
    """Score a single direction (BUY or SELL). Returns (score, reasons)."""
    score   = 0
    reasons = []
    is_buy  = (direction == "BUY")

    # ── 1. HTF Structure (0-4) ────────────────────────────────────────────────
    if htf_bias == ("BULLISH" if is_buy else "BEARISH"):
        score += 3; reasons.append(f"4H bias {htf_bias}")
    elif htf_bias == "NEUTRAL":
        score += 1
    # Recent 4H BOS in trade direction
    if ("BOS_UP" in smc_4h_struct and is_buy) or ("BOS_DOWN" in smc_4h_struct and not is_buy):
        score += 1; reasons.append(f"4H {smc_4h_struct} confirms")

    # ── 2. Order Block (1H, 0-3) ──────────────────────────────────────────────
    ob_1h = extract_nearest_ob(smc_1h.get("ob", pd.DataFrame()), price, direction)
    if ob_1h["type"]:
        score += 2; reasons.append(f"1H {ob_1h['type']} OB @ {ob_1h['mid']:.2f}")
        # Even better if OB on 15m also aligned
        ob_15m = extract_nearest_ob(smc_15m.get("ob", pd.DataFrame()), price, direction)
        if ob_15m["type"]:
            score += 1; reasons.append(f"15m OB confluence @ {ob_15m['mid']:.2f}")

    # ── 3. Fair Value Gap (0-3) ───────────────────────────────────────────────
    fvg_15m = extract_nearest_fvg(smc_15m.get("fvg", pd.DataFrame()), price, direction)
    if fvg_15m["type"]:
        score += 2; reasons.append(f"15m FVG fill ({fvg_15m['bot']:.2f}-{fvg_15m['top']:.2f})")
        fvg_1h = extract_nearest_fvg(smc_1h.get("fvg", pd.DataFrame()), price, direction)
        if fvg_1h["type"]:
            score += 1; reasons.append("1H FVG confluence")

    # ── 4. Liquidity Sweep (0-2) ──────────────────────────────────────────────
    swept_15m, liq_lvl = check_liquidity_swept(smc_15m.get("liquidity", pd.DataFrame()), price, direction)
    swept_1h,  _       = check_liquidity_swept(smc_1h.get("liquidity", pd.DataFrame()), price, direction)
    if swept_15m or swept_1h:
        score += 2; reasons.append(f"Liquidity swept @ {liq_lvl:.2f}")

    # ── 5. CVD (0-2) ──────────────────────────────────────────────────────────
    cvd_bias = cvd.get("bias", "NEUTRAL")
    div_type = cvd.get("divergence_type", "")
    if cvd.get("divergence"):
        if ("BULLISH_DIV" in div_type and is_buy):
            score += 2; reasons.append("CVD bullish divergence (buyers absorbing)")
        elif ("BEARISH_DIV" in div_type and not is_buy):
            score += 2; reasons.append("CVD bearish divergence (sellers absorbing)")
    elif cvd_bias == ("BULLISH" if is_buy else "BEARISH"):
        score += 1; reasons.append(f"CVD {cvd_bias}")

    # ── 6. Volume Profile (0-2) ───────────────────────────────────────────────
    poc = vp.get("poc", 0); vah = vp.get("vah", 0); val = vp.get("val", 0)
    if poc:
        dist_poc = abs(price - poc)
        dist_vah = abs(price - vah)
        dist_val = abs(price - val)
        tol = atr * 0.3
        if dist_poc < tol:
            score += 1; reasons.append(f"Price at POC {poc:.2f}")
        if is_buy and dist_val < tol:
            score += 1; reasons.append(f"Price at VAL {val:.2f} (support)")
        elif not is_buy and dist_vah < tol:
            score += 1; reasons.append(f"Price at VAH {vah:.2f} (resistance)")

    # ── 7. Killzone timing (0-2) ──────────────────────────────────────────────
    if killzone in ("London KZ", "NY AM", "NY Silver Bullet"):
        score += 2; reasons.append(f"In {killzone}")
    elif killzone == "Asian Range":
        # Fade Asian range extremes
        if asian_hi and asian_lo:
            near_hi = abs(price - asian_hi) < atr * 0.5
            near_lo = abs(price - asian_lo) < atr * 0.5
            if not is_buy and near_hi:
                score += 1; reasons.append(f"Asian range high rejection {asian_hi:.2f}")
            if is_buy and near_lo:
                score += 1; reasons.append(f"Asian range low bounce {asian_lo:.2f}")

    # ── 8. VWAP Position (0-2) ───────────────────────────────────────────────
    vwap = vwap_data.get("vwap", 0)
    vu1  = vwap_data.get("vwap_upper1", 0)
    vl1  = vwap_data.get("vwap_lower1", 0)
    vu2  = vwap_data.get("vwap_upper2", 0)
    vl2  = vwap_data.get("vwap_lower2", 0)
    if vwap:
        above = price > vwap
        if is_buy and above:
            score += 1; reasons.append(f"Price above VWAP {vwap:.2f}")
        elif not is_buy and not above:
            score += 1; reasons.append(f"Price below VWAP {vwap:.2f}")
        # 2σ mean-reversion extreme
        if is_buy and price < vl2:
            score += 1; reasons.append(f"At VWAP -2σ {vl2:.2f} (oversold)")
        elif not is_buy and price > vu2:
            score += 1; reasons.append(f"At VWAP +2σ {vu2:.2f} (overbought)")

    return score, reasons


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
                        df_4h: pd.DataFrame = None) -> OFSignal:
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

    # ── Asian range ────────────────────────────────────────────────────────────
    asian_hi, asian_lo = get_asian_range(df_1h)

    # ── Score both directions ──────────────────────────────────────────────────
    logger.info("[OF] Scoring BUY direction...")
    buy_score,  buy_reasons  = _score_direction(
        "BUY", price, atr, smc_15m, smc_1h, struct_4h, htf_bias,
        vp, vwap_data, cvd, killzone, asian_hi, asian_lo
    )
    logger.info("[OF] Scoring SELL direction...")
    sell_score, sell_reasons = _score_direction(
        "SELL", price, atr, smc_15m, smc_1h, struct_4h, htf_bias,
        vp, vwap_data, cvd, killzone, asian_hi, asian_lo
    )

    logger.info(f"[OF] BUY={buy_score} SELL={sell_score} threshold={SCORE_BUY_THRESHOLD}")

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
        # WAIT — include best reasons for transparency
        all_r = buy_reasons[:3] if buy_score > sell_score else sell_reasons[:3]
        return OFSignal(
            action="WAIT", strength="WAIT", score=max(buy_score, sell_score),
            entry=price, atr=atr, killzone=killzone, htf_bias=htf_bias,
            structure=struct_4h, poc=vp.get("poc",0), vah=vp.get("vah",0),
            val=vp.get("val",0), vwap=vwap_data.get("vwap",0),
            cvd_bias=cvd.get("bias",""), reasons=all_r or ["Score below threshold"],
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
        confidence   = round(score / 20, 2),
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
        cvd_bias     = cvd.get("divergence_type", "") or cvd.get("bias", ""),
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
