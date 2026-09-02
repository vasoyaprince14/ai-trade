"""
XAUUSD Multi-Timeframe Strategy
=================================
Framework:
  1H  — Trend filter   (EMA 21/55/200)
  15m — Entry timing   (RSI, MACD crossover, ATR)
  Macro — DXY direction, VIX spike (safe-haven demand)
  Session — London (07-16 UTC) + NY (13-21 UTC) only

Signal scoring (0-10):
  >= 6  → trade
  < 6   → WAIT

SL  : 1.5 × ATR(14)
TP  : 2.5 × ATR(14)   → R:R ≈ 1.67
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d    = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs   = gain / loss.replace(0, np.nan).fillna(1e-9)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def _macd(s: pd.Series, fast=12, slow=26, sig=9) -> Tuple[pd.Series, pd.Series]:
    line   = _ema(s, fast) - _ema(s, slow)
    signal = _ema(line, sig)
    return line, signal


def _session(utc_hour: int) -> str:
    if 7 <= utc_hour < 13:
        return "LONDON"
    elif 13 <= utc_hour < 17:
        return "LONDON+NY"   # most liquid overlap
    elif 17 <= utc_hour < 21:
        return "NEW_YORK"
    elif 0 <= utc_hour < 6:
        return "ASIA"
    return "OFF"


# ── Signal dataclass ───────────────────────────────────────────────────────────

@dataclass
class GoldSignal:
    action:       str        # BUY | SELL | WAIT
    entry:        float
    stop_loss:    float
    target:       float
    risk_reward:  float
    confidence:   float      # 0-1
    score:        int        # raw score out of 10
    reason:       str
    atr:          float
    rsi:          float
    trend_1h:     str        # BULLISH | BEARISH | RANGING
    session:      str
    macro:        dict
    timestamp:    datetime

    def is_trade(self) -> bool:
        return self.action in ("BUY", "SELL")

    def card(self) -> str:
        """WhatsApp/Telegram-style signal card."""
        ts  = self.timestamp.strftime("%d %b %Y  %H:%M UTC")
        dxy = self.macro.get("dxy", 0)
        tnx = self.macro.get("us10y", 0)
        vix = self.macro.get("vix", 0)

        if not self.is_trade():
            return (
                f"⏳  WAIT — XAUUSD\n"
                f"{'─'*38}\n"
                f"Price   : ${self.entry:.2f}\n"
                f"Trend   : {self.trend_1h}\n"
                f"RSI(14) : {self.rsi:.1f}\n"
                f"Session : {self.session}\n"
                f"DXY     : {dxy:.2f}  |  10Y: {tnx:.2f}%  |  VIX: {vix:.1f}\n"
                f"{'─'*38}\n"
                f"Reason  : {self.reason[:280]}\n"
                f"Time    : {ts}"
            )

        emoji = "🟢 LONG " if self.action == "BUY" else "🔴 SHORT"
        risk  = abs(self.entry - self.stop_loss)
        return (
            f"{emoji}  XAUUSD\n"
            f"{'='*38}\n"
            f"Entry   : ${self.entry:.2f}\n"
            f"SL      : ${self.stop_loss:.2f}   ({risk:.2f} pts)\n"
            f"Target  : ${self.target:.2f}\n"
            f"R:R     : 1:{self.risk_reward:.1f}\n"
            f"Score   : {self.score}/10  |  Conf: {self.confidence:.0%}\n"
            f"{'─'*38}\n"
            f"ATR(14) : {self.atr:.2f}\n"
            f"RSI(14) : {self.rsi:.1f}\n"
            f"Trend   : {self.trend_1h}\n"
            f"Session : {self.session}\n"
            f"DXY     : {dxy:.2f}  |  10Y: {tnx:.2f}%  |  VIX: {vix:.1f}\n"
            f"{'─'*38}\n"
            f"Reason  : {self.reason}\n"
            f"Time    : {ts}"
        )

    def telegram_html(self) -> str:
        """Telegram HTML-formatted message."""
        import html as _html
        ts     = self.timestamp.strftime("%d %b %Y  %H:%M UTC")
        dxy    = self.macro.get("dxy", 0)
        tnx    = self.macro.get("us10y", 0)
        vix    = self.macro.get("vix", 0)
        reason = _html.escape(self.reason)

        if not self.is_trade():
            return (
                f"⏳ <b>WAIT — XAUUSD</b>\n"
                f"Price: <b>${self.entry:.2f}</b> | Trend: {self.trend_1h} | RSI: {self.rsi:.1f}\n"
                f"Session: {self.session} | DXY: {dxy:.2f}\n"
                f"<i>{reason[:200]}</i>\n"
                f"⏰ {ts}"
            )

        emoji = "🟢" if self.action == "BUY" else "🔴"
        risk  = abs(self.entry - self.stop_loss)
        return (
            f"{emoji} <b>{self.action} XAUUSD</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Entry:</b>   ${self.entry:.2f}\n"
            f"🛑 <b>SL:</b>      ${self.stop_loss:.2f}  ({risk:.2f} pts)\n"
            f"🎯 <b>Target:</b>  ${self.target:.2f}\n"
            f"📊 <b>R:R:</b>     1:{self.risk_reward:.1f}   Score: {self.score}/10\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 <b>1H Trend:</b> {self.trend_1h}\n"
            f"📉 <b>RSI(14):</b> {self.rsi:.1f}\n"
            f"🌐 <b>DXY:</b> {dxy:.2f}  |  <b>10Y:</b> {tnx:.2f}%  |  <b>VIX:</b> {vix:.1f}\n"
            f"🌍 <b>Session:</b> {self.session}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🧠 <i>{reason}</i>\n"
            f"⏰ {ts}"
        )


# ── Core analysis ──────────────────────────────────────────────────────────────

def analyze(df_15m: pd.DataFrame, df_1h: pd.DataFrame, macro: dict) -> GoldSignal:
    """
    Main entry point. Returns a GoldSignal.
    df_15m / df_1h must have columns: open, close, high, low, volume
    """
    now_utc = datetime.now(timezone.utc)
    session = _session(now_utc.hour)
    price   = float(df_15m["close"].iloc[-1]) if not df_15m.empty else 0.0

    def _wait(reason: str, rsi: float = 50.0, trend: str = "UNKNOWN") -> GoldSignal:
        return GoldSignal("WAIT", price, 0, 0, 0, 0.3, 0, reason,
                          0, rsi, trend, session, macro, now_utc)

    if len(df_15m) < 60 or len(df_1h) < 55:
        return _wait("Not enough bars yet")

    # ── 1H trend ──────────────────────────────────────────────────────────────
    h = df_1h.copy()
    h["e21"]  = _ema(h["close"], 21)
    h["e55"]  = _ema(h["close"], 55)
    h["e200"] = _ema(h["close"], 200)

    lh = h.iloc[-1]
    if lh["e21"] > lh["e55"] > lh["e200"] and lh["close"] > lh["e21"]:
        trend_1h = "BULLISH"
    elif lh["e21"] < lh["e55"] < lh["e200"] and lh["close"] < lh["e21"]:
        trend_1h = "BEARISH"
    else:
        trend_1h = "RANGING"

    # ── 15m indicators ────────────────────────────────────────────────────────
    m = df_15m.copy()
    m["e21"]       = _ema(m["close"], 21)
    m["e55"]       = _ema(m["close"], 55)
    m["rsi"]       = _rsi(m["close"], 14)
    m["atr"]       = _atr(m, 14)
    m["macd"], m["macd_sig"] = _macd(m["close"])
    m["ema_spread"] = m["e21"] - m["e55"]    # positive = short-term bullish

    cur  = m.iloc[-1]
    prev = m.iloc[-2]

    atr  = float(cur["atr"])  if not pd.isna(cur["atr"])  else 6.0
    rsi  = float(cur["rsi"])  if not pd.isna(cur["rsi"])  else 50.0
    macd = float(cur["macd"]) if not pd.isna(cur["macd"]) else 0.0
    msig = float(cur["macd_sig"]) if not pd.isna(cur["macd_sig"]) else 0.0
    pmacd= float(prev["macd"]) if not pd.isna(prev["macd"]) else 0.0
    pmsig= float(prev["macd_sig"]) if not pd.isna(prev["macd_sig"]) else 0.0

    # Macro
    dxy   = macro.get("dxy", 104.0)
    dxy_p = macro.get("dxy_prev", dxy)
    tnx   = macro.get("us10y", 4.5)
    tnx_p = macro.get("us10y_prev", tnx)
    vix   = macro.get("vix", 15.0)

    dxy_falling = dxy < dxy_p - 0.05
    dxy_rising  = dxy > dxy_p + 0.05
    tnx_falling = tnx < tnx_p - 0.02
    tnx_rising  = tnx > tnx_p + 0.02
    vix_elevated = vix > 20

    # ── Scoring ───────────────────────────────────────────────────────────────
    # Each condition adds to buy or sell score (max ~10)
    buy_score, sell_score = 0, 0
    buy_reasons, sell_reasons = [], []

    def _b(cond, pts, reason):
        nonlocal buy_score
        if cond:
            buy_score += pts
            buy_reasons.append(reason)

    def _s(cond, pts, reason):
        nonlocal sell_score
        if cond:
            sell_score += pts
            sell_reasons.append(reason)

    # Trend alignment (most important — 2 pts each)
    _b(trend_1h == "BULLISH",                 2, "1H uptrend (EMA stack)")
    _s(trend_1h == "BEARISH",                 2, "1H downtrend (EMA stack)")

    # Price vs 15m EMAs
    _b(float(cur["close"]) > float(cur["e21"]), 1, "15m price above EMA21")
    _s(float(cur["close"]) < float(cur["e21"]), 1, "15m price below EMA21")
    _b(float(cur["e21"]) > float(cur["e55"]),   1, "15m EMA21 > EMA55")
    _s(float(cur["e21"]) < float(cur["e55"]),   1, "15m EMA21 < EMA55")

    # RSI
    _b(40 <= rsi <= 60,  1, f"RSI {rsi:.0f} — momentum neutral (room to run)")
    _b(rsi < 40,         1, f"RSI {rsi:.0f} — oversold bounce potential")
    _s(60 <= rsi <= 75,  1, f"RSI {rsi:.0f} — momentum neutral (room to fall)")
    _s(rsi > 75,         1, f"RSI {rsi:.0f} — overbought reversal risk")

    # MACD (crossover gets 2 pts, holding gets 1)
    _b(macd > msig and pmacd <= pmsig, 2, "MACD bullish crossover")
    _b(macd > msig and pmacd > pmsig,  1, "MACD above signal")
    _s(macd < msig and pmacd >= pmsig, 2, "MACD bearish crossover")
    _s(macd < msig and pmacd < pmsig,  1, "MACD below signal")

    # Macro — DXY (gold is inverse to USD)
    _b(dxy_falling, 1, f"DXY falling ({dxy:.2f}) — gold tailwind")
    _s(dxy_rising,  1, f"DXY rising ({dxy:.2f}) — gold headwind")

    # Macro — 10Y yield
    _b(tnx_falling, 1, f"10Y yield falling ({tnx:.2f}%) — gold bullish")
    _s(tnx_rising,  1, f"10Y yield rising ({tnx:.2f}%) — gold bearish")

    # Safe-haven demand
    _b(vix_elevated, 1, f"VIX {vix:.1f} elevated — safe-haven buying")
    _s(not vix_elevated and vix < 15, 0, "")  # risk-on slightly bearish for gold

    # Session bonus — London/NY overlap is best
    session_pts = 1 if session in ("LONDON+NY", "LONDON", "NEW_YORK") else 0
    _b(session_pts > 0, session_pts, f"{session} session active")
    _s(session_pts > 0, session_pts, f"{session} session active")

    # ── Additional scoring factors (improve accuracy) ──────────────────────
    # 1. Volume spike: current bar volume > 1.5x 20-bar average
    if "volume" in m.columns:
        vol_now = float(m["volume"].iloc[-1])
        vol_avg = float(m["volume"].iloc[-20:].mean())
        if vol_avg > 0 and vol_now > vol_avg * 1.5:
            _b(float(cur["close"]) > float(prev["close"]), 1, f"Volume spike {vol_now/vol_avg:.1f}x avg — bullish")
            _s(float(cur["close"]) < float(prev["close"]), 1, f"Volume spike {vol_now/vol_avg:.1f}x avg — bearish")

    # 2. EMA21 slope: rate of change over 3 bars (momentum direction)
    ema21_now  = float(m["e21"].iloc[-1])
    ema21_3ago = float(m["e21"].iloc[-4]) if len(m) > 4 else ema21_now
    ema_slope  = ema21_now - ema21_3ago
    _b(ema_slope > 0.05 * atr, 1, f"EMA21 rising slope (+{ema_slope:.2f})")
    _s(ema_slope < -0.05 * atr, 1, f"EMA21 falling slope ({ema_slope:.2f})")

    # 3. Previous 3 bars closed in same direction (momentum confirmation)
    closes = m["close"].iloc[-4:].values
    all_green = all(closes[i] > closes[i-1] for i in range(1, 4))
    all_red   = all(closes[i] < closes[i-1] for i in range(1, 4))
    _b(all_green, 1, "3 consecutive green bars — momentum")
    _s(all_red,   1, "3 consecutive red bars — momentum")

    # 4. Price distance from EMA55 (overextension check)
    e55_dist_pct = (float(cur["close"]) - float(cur["e55"])) / float(cur["e55"]) * 100
    if e55_dist_pct > 0.5:     # price > 0.5% above EMA55 — overextended for buy
        _s(True, 1, f"Price {e55_dist_pct:.2f}% above EMA55 — stretched short")
    elif e55_dist_pct < -0.5:  # price > 0.5% below EMA55 — oversold for sell
        _b(True, 1, f"Price {abs(e55_dist_pct):.2f}% below EMA55 — oversold long")

    # 5. 1H EMA21 slope (medium-term direction)
    if len(h) >= 4:
        h_ema21_now  = float(h["e21"].iloc[-1])
        h_ema21_prev = float(h["e21"].iloc[-3])
        h_slope = h_ema21_now - h_ema21_prev
        _b(h_slope > 0, 1, f"1H EMA21 rising (+{h_slope:.2f})")
        _s(h_slope < 0, 1, f"1H EMA21 falling ({h_slope:.2f})")

    # Asia session — lower threshold (more conservative, need score ≥7)
    TRADE_THRESHOLD = 7 if session == "ASIA" else 6

    # ── Decision ──────────────────────────────────────────────────────────────
    SL_ATR  = 1.5
    TP_ATR  = 2.5
    rr      = round(TP_ATR / SL_ATR, 2)
    MAX_SC  = 10

    if buy_score >= TRADE_THRESHOLD and buy_score > sell_score + 1:
        sl     = round(price - SL_ATR * atr, 2)
        target = round(price + TP_ATR * atr, 2)
        conf   = round(min(0.95, buy_score / MAX_SC), 2)
        reason = " | ".join(buy_reasons)
        if session == "ASIA":
            reason = "[ASIA — tighter filter] " + reason
        return GoldSignal("BUY", price, sl, target, rr, conf,
                          buy_score, reason, atr, rsi, trend_1h, session, macro, now_utc)

    elif sell_score >= TRADE_THRESHOLD and sell_score > buy_score + 1:
        sl     = round(price + SL_ATR * atr, 2)
        target = round(price - TP_ATR * atr, 2)
        conf   = round(min(0.95, sell_score / MAX_SC), 2)
        reason = " | ".join(sell_reasons)
        if session == "ASIA":
            reason = "[ASIA — tighter filter] " + reason
        return GoldSignal("SELL", price, sl, target, rr, conf,
                          sell_score, reason, atr, rsi, trend_1h, session, macro, now_utc)

    else:
        reason = f"No clear edge — buy={buy_score} sell={sell_score}. "
        all_r  = (buy_reasons + sell_reasons)[:5]
        reason += " | ".join(all_r)
        return GoldSignal("WAIT", price, 0, 0, 0, 0.4, 0,
                          reason, atr, rsi, trend_1h, session, macro, now_utc)
