"""
Nifty Intraday Options Signal Engine
======================================
Rule-based, no ML. Clear conditions. Fires real signals during market hours.

Signal types:
  BUY_CE    — Buy Call Option  (market going UP)
  BUY_PE    — Buy Put Option   (market going DOWN)
  SELL_STRADDLE — Sell ATM straddle (market ranging, high IV)
  WAIT      — No setup

Scoring (20 points max):
  VWAP position          : 3 pts  (above VWAP = CE bias, below = PE bias)
  EMA trend (15m)        : 3 pts  (EMA9 > EMA21 = bullish)
  PCR signal             : 3 pts  (PCR < 0.75 = CE writers dominant = bullish)
  Candle structure       : 2 pts  (last 3 candles in direction)
  Session (timing)       : 3 pts  (1st hr = 0, power hr = 3, rest = 1)
  IV environment         : 3 pts  (low IV = directional, high IV = straddle)
  ATM CE/PE OI imbalance : 3 pts  (more PE OI = bullish, more CE OI = bearish)

Threshold: 12/20 = signal fires

Run standalone: python3 nifty/strategy.py
"""

from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")

# ── Constants ────────────────────────────────────────────────────────────────
NIFTY_YF     = "^NSEI"
BN_YF        = "^NSEBANK"
TRADE_THRESH = 10      # out of 22 (was 12/20 — lowered so RSI+VIX can substitute NSE data)
STRONG_THRESH= 15
LOT_SIZE     = 75      # Nifty lot size
STRADDLE_IV  = 16.0    # IV% above which prefer straddle over directional

@dataclass
class NiftySignal:
    action:     str        # BUY_CE | BUY_PE | SELL_STRADDLE | WAIT
    strength:   str        # STRONG | MODERATE | WAIT
    score:      int
    max_score:  int = 24
    spot:       float = 0.0
    atm_strike: int   = 0
    expiry:     str   = ""
    entry_ce:   float = 0.0   # ATM CE LTP (for BUY_CE)
    entry_pe:   float = 0.0   # ATM PE LTP (for BUY_PE)
    sl_pts:     float = 0.0   # points SL on spot (e.g. 80 pts)
    sl_spot:    float = 0.0
    target_spot: float = 0.0
    rr:         float = 0.0
    pcr:        float = 0.0
    atm_iv:     float = 0.0
    vwap:       float = 0.0
    ema9_15m:   float = 0.0
    ema21_15m:  float = 0.0
    trend_15m:  str   = ""
    session:    str   = ""    # OPENING | MID_SESSION | POWER_HOUR | CLOSED
    reasons:    list  = field(default_factory=list)
    timestamp:  datetime = field(default_factory=lambda: datetime.now(IST))

    @property
    def confidence(self) -> float:
        return round(self.score / self.max_score, 2)

    def is_trade(self) -> bool:
        return self.action != "WAIT"

    def telegram_html(self) -> str:
        import html
        if not self.is_trade():
            return (f"⏳ <b>NIFTY WAIT</b> — Score {self.score}/{self.max_score}\n"
                    f"PCR: {self.pcr:.2f} | VWAP: {self.vwap:.0f} | {self.session}\n"
                    f"<i>{' | '.join(self.reasons[:2])}</i>")

        action_str = {"BUY_CE": "🟢 BUY CE", "BUY_PE": "🔴 BUY PE",
                      "SELL_STRADDLE": "⚡ SELL STRADDLE"}.get(self.action, self.action)
        body = (f"{action_str} — <b>NIFTY</b> [{self.strength}]\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📍 Spot     : <b>{self.spot:.0f}</b>\n"
                f"🎯 ATM      : <b>{self.atm_strike}</b>  [{self.expiry}]\n")
        if self.action == "BUY_CE":
            body += f"💰 CE Entry : <b>~{self.entry_ce:.0f}</b>\n"
        elif self.action == "BUY_PE":
            body += f"💰 PE Entry : <b>~{self.entry_pe:.0f}</b>\n"
        elif self.action == "SELL_STRADDLE":
            body += f"💰 CE+PE    : <b>~{self.entry_ce:.0f} + {self.entry_pe:.0f}</b>\n"
        body += (f"🛑 SL Spot  : <b>{self.sl_spot:.0f}</b> ({self.sl_pts:.0f} pts)\n"
                 f"✅ Target   : <b>{self.target_spot:.0f}</b>  R:R 1:{self.rr:.1f}\n"
                 f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                 f"📊 PCR: {self.pcr:.2f} | IV: {self.atm_iv:.1f}% | Score: {self.score}/20\n"
                 f"💡 {' | '.join(html.escape(r) for r in self.reasons[:4])}\n"
                 f"⏱ {self.timestamp.strftime('%d %b %H:%M IST')}")
        return body


# ── Data helpers ─────────────────────────────────────────────────────────────

def _fetch_nifty(interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    df = yf.download(NIFTY_YF, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = "timestamp"
    df = df.reset_index()
    # Convert to IST for filtering
    if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].dt.tz_convert(IST)
    return df.dropna()


def _compute_vwap(df: pd.DataFrame) -> float:
    """Session VWAP — resets each day."""
    if df.empty:
        return 0.0
    today = df["timestamp"].dt.date.iloc[-1]
    today_df = df[df["timestamp"].dt.date == today].copy()
    if today_df.empty:
        return 0.0
    tp = (today_df["high"] + today_df["low"] + today_df["close"]) / 3
    vol = today_df["volume"].replace(0, 1)
    return float((tp * vol).sum() / vol.sum())


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _get_nse_data() -> dict:
    """
    Fetch Nifty spot, PCR, ATM strike, ATM IV and option LTPs from NSE.
    Returns dict with keys: spot, pcr, atm_strike, atm_iv, ce_ltp, pe_ltp,
                            expiry, ce_oi_atm, pe_oi_atm
    """
    result = {
        "spot": 0.0, "pcr": 1.0, "atm_strike": 0,
        "atm_iv": 15.0, "ce_ltp": 0.0, "pe_ltp": 0.0,
        "expiry": "", "ce_oi_atm": 0, "pe_oi_atm": 0,
        "total_ce_oi": 0, "total_pe_oi": 0,
    }
    try:
        from core.data.nse_scraper import NSEScraper
        sc = NSEScraper()
        raw = sc.get_option_chain_raw("NIFTY")
        if not isinstance(raw, dict):
            return result

        records = raw.get("records", {})
        spot = float(records.get("underlyingValue", 0))
        if not spot:
            return result
        result["spot"] = spot

        expiries = records.get("expiryDates", [])
        expiry = expiries[0] if expiries else ""
        result["expiry"] = expiry

        data = records.get("data", [])
        if not data:
            return result

        # ATM strike (round to nearest 50)
        atm = int(round(spot / 50) * 50)
        result["atm_strike"] = atm

        # Total OI for PCR
        ce_total = sum(r.get("CE", {}).get("openInterest", 0)
                       for r in data if r.get("CE") and r.get("expiryDate") == expiry)
        pe_total = sum(r.get("PE", {}).get("openInterest", 0)
                       for r in data if r.get("PE") and r.get("expiryDate") == expiry)
        result["total_ce_oi"] = ce_total
        result["total_pe_oi"] = pe_total
        if ce_total:
            result["pcr"] = round(pe_total / ce_total, 3)

        # ATM strike data
        for row in data:
            if row.get("expiryDate") != expiry:
                continue
            strike = row.get("strikePrice", 0)
            if abs(int(strike) - atm) < 26:  # within 1 strike
                ce = row.get("CE", {})
                pe = row.get("PE", {})
                if ce:
                    result["ce_ltp"]    = float(ce.get("lastPrice", 0))
                    result["atm_iv"]    = float(ce.get("impliedVolatility", 15))
                    result["ce_oi_atm"] = int(ce.get("openInterest", 0))
                if pe:
                    result["pe_ltp"]    = float(pe.get("lastPrice", 0))
                    if not result["atm_iv"]:
                        result["atm_iv"] = float(pe.get("impliedVolatility", 15))
                    result["pe_oi_atm"] = int(pe.get("openInterest", 0))
                break

    except Exception as e:
        logger.warning(f"[Nifty] NSE data error: {e}")
    return result


def _get_session(now_ist: datetime) -> str:
    h, m = now_ist.hour, now_ist.minute
    mins = h * 60 + m
    if mins < 9 * 60 + 15:
        return "PRE_MARKET"
    if mins <= 9 * 60 + 45:
        return "OPENING"         # first 30 min — avoid
    if mins <= 14 * 60 + 30:
        return "MID_SESSION"
    if mins <= 15 * 60 + 15:
        return "POWER_HOUR"     # best signals
    if mins <= 15 * 60 + 30:
        return "CLOSING"
    return "CLOSED"


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_nifty() -> NiftySignal:
    """
    Core signal generator. Called every 5 min by the engine.
    Returns NiftySignal with action, levels, reasons.
    """
    now_ist  = datetime.now(IST)
    session  = _get_session(now_ist)

    # Fetch price data
    df_5m  = _fetch_nifty("5m",  "5d")
    df_15m = _fetch_nifty("15m", "15d")

    nse = _get_nse_data()
    spot     = nse["spot"]
    pcr      = nse["pcr"]
    atm      = nse["atm_strike"]
    atm_iv   = nse["atm_iv"]
    ce_ltp   = nse["ce_ltp"]
    pe_ltp   = nse["pe_ltp"]
    expiry   = nse["expiry"]
    ce_oi    = nse["ce_oi_atm"]
    pe_oi    = nse["pe_oi_atm"]

    # Fallback spot from yfinance
    if not spot and not df_5m.empty:
        spot = float(df_5m["close"].iloc[-1])
        atm  = int(round(spot / 50) * 50)

    if not spot:
        return NiftySignal(action="WAIT", strength="WAIT", score=0,
                           reasons=["No market data"], session=session)

    # VWAP
    vwap = _compute_vwap(df_5m)

    # EMA trend on 15m
    ema9 = ema21 = trend_15m = ""
    rsi_14 = 50.0
    if not df_15m.empty and len(df_15m) >= 21:
        close_15m = df_15m["close"]
        ema9_s  = _ema(close_15m, 9)
        ema21_s = _ema(close_15m, 21)
        ema9    = float(ema9_s.iloc[-1])
        ema21   = float(ema21_s.iloc[-1])
        if ema9 > ema21 and spot > ema9:
            trend_15m = "BULLISH"
        elif ema9 < ema21 and spot < ema9:
            trend_15m = "BEARISH"
        else:
            trend_15m = "NEUTRAL"
        # RSI 14 on 15m
        if len(close_15m) >= 14:
            d = close_15m.diff()
            up = d.clip(lower=0).rolling(14).mean()
            dn = (-d.clip(upper=0)).rolling(14).mean()
            rs = up / dn.replace(0, 1e-9)
            rsi_series = 100 - 100 / (1 + rs)
            rsi_14 = float(rsi_series.iloc[-1])

    # India VIX from yfinance (always available as fallback)
    india_vix = 15.0
    try:
        vix_df = yf.download("^INDIAVIX", period="2d", interval="1d", progress=False)
        if not vix_df.empty:
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = [c[0].lower() for c in vix_df.columns]
            else:
                vix_df.columns = [c.lower() for c in vix_df.columns]
            india_vix = float(vix_df["close"].iloc[-1])
            if not atm_iv:
                atm_iv = india_vix  # use VIX as IV proxy if NSE data unavailable
    except Exception:
        pass

    # Last 3 candles structure (5m)
    candle_dir = "NEUTRAL"
    if not df_5m.empty and len(df_5m) >= 3:
        last3 = df_5m.tail(3)
        bulls = sum(1 for _, r in last3.iterrows() if float(r["close"]) > float(r["open"]))
        bears = sum(1 for _, r in last3.iterrows() if float(r["close"]) < float(r["open"]))
        if bulls == 3:   candle_dir = "BULLISH"
        elif bears == 3: candle_dir = "BEARISH"

    # ── Scoring ───────────────────────────────────────────────────────────────
    buy_ce_score  = 0
    buy_pe_score  = 0
    straddle_score= 0
    reasons_ce    = []
    reasons_pe    = []
    reasons_str   = []

    # 1. VWAP (3 pts each)
    if vwap and spot > vwap * 1.001:
        buy_ce_score  += 3
        reasons_ce.append(f"Spot {spot:.0f} above VWAP {vwap:.0f} — bullish")
    elif vwap and spot < vwap * 0.999:
        buy_pe_score  += 3
        reasons_pe.append(f"Spot {spot:.0f} below VWAP {vwap:.0f} — bearish")
    else:
        straddle_score += 1
        reasons_str.append("Spot near VWAP — ranging")

    # 2. EMA trend 15m (3 pts)
    if trend_15m == "BULLISH":
        buy_ce_score  += 3
        reasons_ce.append(f"15m EMA9({ema9:.0f}) > EMA21({ema21:.0f}) — uptrend")
    elif trend_15m == "BEARISH":
        buy_pe_score  += 3
        reasons_pe.append(f"15m EMA9({ema9:.0f}) < EMA21({ema21:.0f}) — downtrend")
    else:
        straddle_score += 1
        reasons_str.append("15m trend NEUTRAL — sideways")

    # 3. PCR signal (3 pts)
    if pcr < 0.70:
        buy_ce_score  += 3
        reasons_ce.append(f"PCR {pcr:.2f} low — CE writers dominant (bullish)")
    elif pcr < 0.85:
        buy_ce_score  += 2
        reasons_ce.append(f"PCR {pcr:.2f} — slight CE bias (bullish)")
    elif pcr > 1.35:
        buy_pe_score  += 3
        reasons_pe.append(f"PCR {pcr:.2f} high — PE writers dominant (bearish)")
    elif pcr > 1.15:
        buy_pe_score  += 2
        reasons_pe.append(f"PCR {pcr:.2f} — slight PE bias (bearish)")
    elif 0.90 <= pcr <= 1.10:
        straddle_score += 2
        reasons_str.append(f"PCR {pcr:.2f} neutral — no direction bias")

    # 4. Candle structure 5m (2 pts)
    if candle_dir == "BULLISH":
        buy_ce_score  += 2
        reasons_ce.append("3 consecutive green candles — momentum up")
    elif candle_dir == "BEARISH":
        buy_pe_score  += 2
        reasons_pe.append("3 consecutive red candles — momentum down")

    # 5. Session timing (3 pts)
    session_pts = {"OPENING": 0, "MID_SESSION": 1, "POWER_HOUR": 3,
                   "CLOSING": 1, "PRE_MARKET": 0, "CLOSED": 0}
    sp = session_pts.get(session, 0)
    buy_ce_score  += sp
    buy_pe_score  += sp
    straddle_score += sp
    if session == "POWER_HOUR":
        reasons_ce.append("Power hour (14:30-15:15) — high probability window")
        reasons_pe.append("Power hour (14:30-15:15) — high probability window")
    elif session == "OPENING":
        reasons_ce.append("Opening range — cautious (low pts)")
        reasons_pe.append("Opening range — cautious (low pts)")

    # 6. IV environment (3 pts)
    if atm_iv > STRADDLE_IV:
        straddle_score += 3
        reasons_str.append(f"ATM IV {atm_iv:.1f}% high — straddle sell favored")
    elif atm_iv < 12:
        buy_ce_score  += 3
        buy_pe_score  += 3
        reasons_ce.append(f"ATM IV {atm_iv:.1f}% low — cheap options, directional buy")
        reasons_pe.append(f"ATM IV {atm_iv:.1f}% low — cheap options, directional buy")
    elif atm_iv < STRADDLE_IV:
        buy_ce_score  += 2
        buy_pe_score  += 2
        reasons_ce.append(f"ATM IV {atm_iv:.1f}% moderate — directional ok")
        reasons_pe.append(f"ATM IV {atm_iv:.1f}% moderate — directional ok")

    # 7. ATM CE/PE OI imbalance (3 pts)
    if ce_oi and pe_oi:
        ratio = pe_oi / ce_oi if ce_oi else 1
        if ratio > 1.3:
            buy_ce_score += 3
            reasons_ce.append(f"PE OI {pe_oi:,} > CE OI {ce_oi:,} at ATM — put writers bullish")
        elif ratio < 0.7:
            buy_pe_score += 3
            reasons_pe.append(f"CE OI {ce_oi:,} > PE OI {pe_oi:,} at ATM — call writers bearish")
        else:
            straddle_score += 1
            reasons_str.append(f"ATM CE/PE OI balanced ({ce_oi:,}/{pe_oi:,})")

    # 8. RSI 14 on 15m (2 pts) — always available from yfinance
    if rsi_14 >= 55 and rsi_14 <= 72:
        buy_ce_score  += 2
        reasons_ce.append(f"RSI {rsi_14:.0f} bullish momentum (55-72)")
    elif rsi_14 >= 28 and rsi_14 <= 45:
        buy_pe_score  += 2
        reasons_pe.append(f"RSI {rsi_14:.0f} bearish momentum (28-45)")
    elif rsi_14 > 72:
        buy_pe_score  += 1
        straddle_score += 1
        reasons_pe.append(f"RSI {rsi_14:.0f} overbought — PE bias")
    elif rsi_14 < 28:
        buy_ce_score  += 1
        straddle_score += 1
        reasons_ce.append(f"RSI {rsi_14:.0f} oversold — CE bias")

    # 9. India VIX regime (2 pts) — always available from yfinance
    if india_vix < 14:
        buy_ce_score  += 2
        buy_pe_score  += 2
        reasons_ce.append(f"India VIX {india_vix:.1f} very low — cheap options, directional trade")
        reasons_pe.append(f"India VIX {india_vix:.1f} very low — cheap options, directional trade")
    elif india_vix < 18:
        buy_ce_score  += 1
        buy_pe_score  += 1
        reasons_ce.append(f"India VIX {india_vix:.1f} low — directional ok")
        reasons_pe.append(f"India VIX {india_vix:.1f} low — directional ok")
    elif india_vix > 22:
        straddle_score += 2
        reasons_str.append(f"India VIX {india_vix:.1f} elevated — straddle/hedge preferred")

    # ── Determine direction ───────────────────────────────────────────────────
    # Opening range = no trade regardless of score
    if session in ("OPENING", "PRE_MARKET", "CLOSED"):
        return NiftySignal(
            action="WAIT", strength="WAIT", score=0, max_score=24,
            spot=spot, atm_strike=atm, expiry=expiry, pcr=pcr,
            atm_iv=atm_iv, vwap=vwap, ema9_15m=ema9 or 0, ema21_15m=ema21 or 0,
            trend_15m=trend_15m or "NEUTRAL", session=session,
            entry_ce=ce_ltp, entry_pe=pe_ltp,
            reasons=[f"Session: {session} — waiting for mid-session"],
            timestamp=now_ist,
        )

    best_dir   = "WAIT"
    best_score = 0
    best_reasons = []

    if buy_ce_score >= buy_pe_score and buy_ce_score >= straddle_score:
        best_dir     = "BUY_CE"
        best_score   = buy_ce_score
        best_reasons = reasons_ce
    elif buy_pe_score > buy_ce_score and buy_pe_score >= straddle_score:
        best_dir     = "BUY_PE"
        best_score   = buy_pe_score
        best_reasons = reasons_pe
    elif straddle_score >= TRADE_THRESH:
        best_dir     = "SELL_STRADDLE"
        best_score   = straddle_score
        best_reasons = reasons_str

    if best_score < TRADE_THRESH:
        best_dir     = "WAIT"
        best_score   = max(buy_ce_score, buy_pe_score, straddle_score)
        best_reasons = (reasons_ce if buy_ce_score >= buy_pe_score else reasons_pe) or reasons_str

    strength = "WAIT"
    if best_dir != "WAIT":
        strength = "STRONG" if best_score >= STRONG_THRESH else "MODERATE"

    # ── Trade levels ──────────────────────────────────────────────────────────
    atr_pts = 0.0
    if not df_5m.empty and len(df_5m) >= 14:
        hl = df_5m["high"] - df_5m["low"]
        hc = (df_5m["high"] - df_5m["close"].shift()).abs()
        lc = (df_5m["low"]  - df_5m["close"].shift()).abs()
        atr_pts = float(pd.concat([hl, hc, lc], axis=1).max(axis=1).tail(14).mean())

    sl_pts     = round(max(atr_pts * 1.5, spot * 0.003), 0)  # min 0.3% of spot
    target_pts = round(sl_pts * 2.0, 0)                       # 1:2 R:R

    if best_dir == "BUY_CE":
        sl_spot     = round(spot - sl_pts, 0)
        target_spot = round(spot + target_pts, 0)
    elif best_dir == "BUY_PE":
        sl_spot     = round(spot + sl_pts, 0)
        target_spot = round(spot - target_pts, 0)
    else:
        sl_spot = target_spot = 0.0

    rr = round(target_pts / sl_pts, 1) if sl_pts else 0.0

    return NiftySignal(
        action=best_dir, strength=strength, score=best_score, max_score=24,
        spot=spot, atm_strike=atm, expiry=expiry,
        entry_ce=ce_ltp, entry_pe=pe_ltp,
        sl_pts=sl_pts, sl_spot=sl_spot, target_spot=target_spot, rr=rr,
        pcr=pcr, atm_iv=atm_iv, vwap=vwap,
        ema9_15m=ema9 or 0, ema21_15m=ema21 or 0, trend_15m=trend_15m or "NEUTRAL",
        session=session, reasons=best_reasons,
        timestamp=now_ist,
    )


if __name__ == "__main__":
    from loguru import logger
    sig = analyze_nifty()
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  NIFTY SIGNAL  [{sig.action}] [{sig.strength}]")
    print(f"  Score: {sig.score}/{sig.max_score}  Threshold: {TRADE_THRESH}/24")
    print(sep)
    print(f"  Spot      : {sig.spot:.0f}")
    print(f"  ATM       : {sig.atm_strike}  [{sig.expiry}]")
    print(f"  PCR       : {sig.pcr:.3f}")
    print(f"  ATM IV    : {sig.atm_iv:.1f}%")
    print(f"  VWAP      : {sig.vwap:.0f}")
    print(f"  15m Trend : {sig.trend_15m}")
    print(f"  Session   : {sig.session}")
    if sig.is_trade():
        print(sep)
        print(f"  CE LTP    : {sig.entry_ce:.0f}")
        print(f"  PE LTP    : {sig.entry_pe:.0f}")
        print(f"  SL Spot   : {sig.sl_spot:.0f}  ({sig.sl_pts:.0f} pts)")
        print(f"  Target    : {sig.target_spot:.0f}  R:R 1:{sig.rr}")
    print(sep)
    print(f"  Reasons:")
    for r in sig.reasons:
        print(f"    • {r}")
    print(sep)
