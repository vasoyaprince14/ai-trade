"""
Nifty DOM Order Flow Strategy
==============================
Uses real 20-level Dhan market depth to detect institutional order flow.

Unlike the main strategy (which uses PCR + OI from NSE scraper),
this strategy reads the actual live order book — how many lots are
resting on each bid/ask price level and how many individual orders.

Signal logic:
  BUY_CE    — Strong bid-side DOM pressure + call-friendly OI flow
  BUY_PE    — Strong ask-side DOM pressure + put-friendly OI flow
  WAIT      — DOM balanced or session closed

Scoring (20 pts):
  1. DOM qty imbalance   (0-4 pts)  top-5 bid vs ask quantity
  2. DOM order imbalance (0-2 pts)  number of orders (spoof filter)
  3. Large order walls   (0-3 pts)  resting >500 lots = support/resistance
  4. OI delta flow       (0-3 pts)  CE OI change vs PE OI change direction
  5. Spread tightness    (0-2 pts)  tight spread = active, wide = thin
  6. Session timing      (0-3 pts)  power hour best
  7. Price vs VWAP       (0-3 pts)  above VWAP = CE bias

Threshold: 10/20 fires signal, 14/20 = STRONG

Run:
    python3 nifty/dom_strategy.py
"""

from __future__ import annotations
import sys, os
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pytz

from loguru import logger

IST = pytz.timezone("Asia/Kolkata")

TRADE_THRESH  = 10
STRONG_THRESH = 14
WALL_LOT_MIN  = 500    # lots — resting order counts as "wall"
SPREAD_TIGHT  = 1.5    # points — spread below this = active book


@dataclass
class DOMSignal:
    action:       str          # BUY_CE | BUY_PE | WAIT
    strength:     str          # STRONG | MODERATE | WAIT
    score:        int
    max_score:    int = 20

    # Market data
    spot:         float = 0.0
    atm_strike:   int   = 0
    expiry:       str   = ""
    entry_ce:     float = 0.0
    entry_pe:     float = 0.0
    sl_pts:       float = 0.0
    sl_spot:      float = 0.0
    target_spot:  float = 0.0
    rr:           float = 0.0

    # DOM metrics
    book_imbalance:     float = 0.0   # +1=all bids, -1=all asks
    order_imbalance:    float = 0.0   # imbalance by order count
    bid_qty_5:          int   = 0
    ask_qty_5:          int   = 0
    spread:             float = 0.0
    bid_wall_price:     float = 0.0
    bid_wall_qty:       int   = 0
    ask_wall_price:     float = 0.0
    ask_wall_qty:       int   = 0
    depth_level:        int   = 0

    # OI flow
    ce_doi:       int   = 0    # CE OI change
    pe_doi:       int   = 0    # PE OI change
    ce_oi:        int   = 0
    pe_oi:        int   = 0
    pcr:          float = 0.0
    atm_iv:       float = 0.0

    # Context
    vwap:         float = 0.0
    session:      str   = ""
    reasons:      list  = field(default_factory=list)
    timestamp:    datetime = field(default_factory=lambda: datetime.now(IST))

    @property
    def confidence(self) -> float:
        return round(self.score / self.max_score, 2)

    def is_trade(self) -> bool:
        return self.action != "WAIT"

    def telegram_html(self) -> str:
        import html as _html
        action_str = {"BUY_CE": "🟢 BUY CE", "BUY_PE": "🔴 BUY PE"}.get(self.action, self.action)
        imb_bar = _imb_bar(self.book_imbalance)

        if not self.is_trade():
            return (
                f"📊 <b>NIFTY DOM — WAIT</b>  Score {self.score}/{self.max_score}\n"
                f"📍 Spot: {self.spot:.0f} | Book: {imb_bar} {self.book_imbalance:+.2f}\n"
                f"Spread: {self.spread:.1f}pt | Bid:{self.bid_qty_5} Ask:{self.ask_qty_5} lots\n"
                f"⏱ {self.timestamp.strftime('%H:%M IST')}"
            )

        wall_line = ""
        if self.bid_wall_qty >= WALL_LOT_MIN:
            wall_line += f"🧱 Bid wall: {self.bid_wall_qty} lots @ {self.bid_wall_price:.0f}\n"
        if self.ask_wall_qty >= WALL_LOT_MIN:
            wall_line += f"🧱 Ask wall: {self.ask_wall_qty} lots @ {self.ask_wall_price:.0f}\n"

        return (
            f"📊 <b>NIFTY DOM — {action_str}</b>  [{self.strength}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Spot     : <b>{self.spot:.0f}</b>\n"
            f"🎯 ATM      : <b>{self.atm_strike}</b>  [{self.expiry}]\n"
            + (f"💰 CE Entry : <b>~{self.entry_ce:.0f}</b>\n" if self.action == "BUY_CE" else
               f"💰 PE Entry : <b>~{self.entry_pe:.0f}</b>\n")
            + f"🛑 SL Spot  : <b>{self.sl_spot:.0f}</b> ({self.sl_pts:.0f} pts)\n"
            f"✅ Target   : <b>{self.target_spot:.0f}</b>  R:R 1:{self.rr:.1f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📖 Book Imb : {imb_bar} <b>{self.book_imbalance:+.2f}</b>\n"
            f"   Bid      : {self.bid_qty_5:,} lots | Ask: {self.ask_qty_5:,} lots\n"
            f"   Orders   : imb {self.order_imbalance:+.2f} | Spread: {self.spread:.1f}pt\n"
            + wall_line
            + f"📈 OI Flow  : CE Δ{self.ce_doi:+,} | PE Δ{self.pe_doi:+,}\n"
            f"   PCR: {self.pcr:.2f} | IV: {self.atm_iv:.1f}%\n"
            f"💡 {' | '.join(_html.escape(r) for r in self.reasons[:4])}\n"
            f"⏱ {self.timestamp.strftime('%d %b %H:%M IST')}"
        )


def _imb_bar(imb: float) -> str:
    """ASCII bar: ████░░░░ for imbalance visualisation."""
    filled = int((imb + 1) / 2 * 8)
    filled = max(0, min(8, filled))
    return "█" * filled + "░" * (8 - filled)


# ── Session ────────────────────────────────────────────────────────────────────

def _get_session(now_ist: datetime) -> str:
    h, m = now_ist.hour, now_ist.minute
    mins  = h * 60 + m
    if mins < 9 * 60 + 15:  return "PRE_MARKET"
    if mins <= 9 * 60 + 45: return "OPENING"
    if mins <= 14 * 60 + 30: return "MID_SESSION"
    if mins <= 15 * 60 + 15: return "POWER_HOUR"
    if mins <= 15 * 60 + 30: return "CLOSING"
    return "CLOSED"


# ── VWAP from yfinance ─────────────────────────────────────────────────────────

def _get_vwap() -> float:
    try:
        import yfinance as yf
        import pandas as pd
        df = yf.download("^NSEI", period="1d", interval="5m", progress=False)
        if df.empty:
            return 0.0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        tp  = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"].replace(0, 1)
        return float((tp * vol).sum() / vol.sum())
    except Exception:
        return 0.0


# ── Main DOM analysis ──────────────────────────────────────────────────────────

def analyze_dom() -> DOMSignal:
    """
    Core DOM signal generator using Dhan real 20-level order book.
    Called every 30 seconds by dom_engine.py.
    """
    now_ist = datetime.now(IST)
    session = _get_session(now_ist)

    def _wait(reason: str, **kw) -> DOMSignal:
        return DOMSignal(action="WAIT", strength="WAIT", score=0,
                         session=session, reasons=[reason],
                         timestamp=now_ist, **kw)

    # ── Require Dhan ──────────────────────────────────────────────────────────
    if not (os.getenv("DHAN_CLIENT_ID") and os.getenv("DHAN_ACCESS_TOKEN")):
        return _wait("Dhan credentials not set")

    try:
        from core.data.dhan_feed import get_dhan_feed
        feed = get_dhan_feed()
    except Exception as e:
        return _wait(f"Dhan init error: {e}")

    if not feed.connected:
        return _wait("Dhan not connected")

    # ── Fetch DOM depth ───────────────────────────────────────────────────────
    try:
        depth = feed.get_depth("NIFTY", levels=20)
    except Exception as e:
        return _wait(f"Depth fetch error: {e}")

    if not depth.ltp or depth.ltp < 1000:
        return _wait("No depth data from Dhan")

    spot   = depth.ltp
    spread = depth.spread()
    imb    = depth.imbalance(5)
    o_imb  = depth.order_count_imbalance(5)
    bid5   = depth.total_bid_qty(5)
    ask5   = depth.total_ask_qty(5)
    walls  = depth.large_orders(WALL_LOT_MIN)

    # ── Fetch option chain ────────────────────────────────────────────────────
    ce_doi = pe_doi = ce_oi = pe_oi = 0
    ce_ltp = pe_ltp = atm_iv = pcr = 0.0
    atm    = int(round(spot / 50) * 50)
    expiry = ""

    try:
        chain = feed.get_option_chain("NIFTY")
        if chain.strikes:
            expiry = chain.expiry
            ce_total = pe_total = 0
            for s in chain.strikes:
                ce_total += s.get("ce_oi", 0)
                pe_total += s.get("pe_oi", 0)
                if abs(s["strike"] - atm) < 26:
                    ce_doi  = s.get("ce_doi", 0)
                    pe_doi  = s.get("pe_doi", 0)
                    ce_oi   = s.get("ce_oi", 0)
                    pe_oi   = s.get("pe_oi", 0)
                    ce_ltp  = s.get("ce_ltp", 0.0)
                    pe_ltp  = s.get("pe_ltp", 0.0)
                    atm_iv  = s.get("ce_iv") or s.get("pe_iv") or 15.0
            if ce_total:
                pcr = round(pe_total / ce_total, 3)
    except Exception as e:
        logger.debug(f"[DOM] Option chain error: {e}")

    # ── VWAP ─────────────────────────────────────────────────────────────────
    vwap = _get_vwap()

    # ── Scoring ───────────────────────────────────────────────────────────────
    ce_score = 0
    pe_score = 0
    reasons_ce: list[str] = []
    reasons_pe: list[str] = []

    # 1. DOM qty imbalance (4 pts)
    if imb >= 0.30:
        ce_score += 4
        reasons_ce.append(f"DOM: strong bid pressure (imb={imb:+.2f}, {bid5:,} vs {ask5:,} lots)")
    elif imb >= 0.15:
        ce_score += 3
        reasons_ce.append(f"DOM: bid dominant (imb={imb:+.2f})")
    elif imb >= 0.05:
        ce_score += 1
        reasons_ce.append(f"DOM: slight bid edge (imb={imb:+.2f})")
    elif imb <= -0.30:
        pe_score += 4
        reasons_pe.append(f"DOM: strong ask pressure (imb={imb:+.2f}, {ask5:,} vs {bid5:,} lots)")
    elif imb <= -0.15:
        pe_score += 3
        reasons_pe.append(f"DOM: ask dominant (imb={imb:+.2f})")
    elif imb <= -0.05:
        pe_score += 1
        reasons_pe.append(f"DOM: slight ask edge (imb={imb:+.2f})")

    # 2. Order count imbalance — spoof filter (2 pts)
    # If qty imbalance and order count imbalance agree → real signal
    if o_imb >= 0.10 and imb >= 0.10:
        ce_score += 2
        reasons_ce.append(f"Order count confirms bid pressure (order imb={o_imb:+.2f})")
    elif o_imb <= -0.10 and imb <= -0.10:
        pe_score += 2
        reasons_pe.append(f"Order count confirms ask pressure (order imb={o_imb:+.2f})")
    elif (o_imb > 0) != (imb > 0) and abs(imb) > 0.1:
        # Disagreement = possible spoofing, reduce trust
        ce_score = max(0, ce_score - 1)
        pe_score = max(0, pe_score - 1)
        reasons_ce.append(f"Qty/order imbalance mismatch — possible spoof, reducing score")

    # 3. Large order walls (3 pts)
    bid_wall = next((w for w in walls if w["side"] == "BID"), None)
    ask_wall = next((w for w in walls if w["side"] == "ASK"), None)
    bid_wall_price = bid_wall["price"] if bid_wall else 0
    bid_wall_qty   = bid_wall["qty"]   if bid_wall else 0
    ask_wall_price = ask_wall["price"] if ask_wall else 0
    ask_wall_qty   = ask_wall["qty"]   if ask_wall else 0

    if bid_wall and bid_wall["price"] < spot:
        # Large bid wall below spot = support
        ce_score += 3
        reasons_ce.append(f"Bid wall: {bid_wall['qty']:,} lots @ {bid_wall['price']:.0f} — support")
    if ask_wall and ask_wall["price"] > spot:
        # Large ask wall above spot = resistance
        pe_score += 3
        reasons_pe.append(f"Ask wall: {ask_wall['qty']:,} lots @ {ask_wall['price']:.0f} — resistance")

    # 4. OI delta flow (3 pts)
    # CE OI increasing = call writing = bearish; PE OI increasing = put writing = bullish
    if pe_doi > 0 and ce_doi <= 0:
        ce_score += 3
        reasons_ce.append(f"OI flow bullish: PE OI +{pe_doi:,}, CE OI {ce_doi:+,}")
    elif pe_doi > 0 and ce_doi > 0:
        ce_score += 1
        reasons_ce.append(f"Mixed OI: PE +{pe_doi:,} CE +{ce_doi:,}")
    elif ce_doi > 0 and pe_doi <= 0:
        pe_score += 3
        reasons_pe.append(f"OI flow bearish: CE OI +{ce_doi:,}, PE OI {pe_doi:+,}")
    elif ce_doi > 0 and pe_doi > 0:
        pe_score += 1
        reasons_pe.append(f"Mixed OI: CE +{ce_doi:,} PE +{pe_doi:,}")

    # 5. Spread tightness (2 pts) — tight spread = active market, both sides benefit
    if spread > 0 and spread <= SPREAD_TIGHT:
        ce_score += 2
        pe_score += 2
        reasons_ce.append(f"Tight spread {spread:.1f}pt — active book")
        reasons_pe.append(f"Tight spread {spread:.1f}pt — active book")
    elif spread > 4:
        ce_score = max(0, ce_score - 1)
        pe_score = max(0, pe_score - 1)
        reasons_ce.append(f"Wide spread {spread:.1f}pt — thin book")

    # 6. Session timing (3 pts)
    session_pts = {
        "OPENING": 0, "MID_SESSION": 1,
        "POWER_HOUR": 3, "CLOSING": 2,
        "PRE_MARKET": 0, "CLOSED": 0,
    }
    sp = session_pts.get(session, 0)
    ce_score += sp
    pe_score += sp
    if session == "POWER_HOUR":
        reasons_ce.append("Power hour — high conviction window")
        reasons_pe.append("Power hour — high conviction window")

    # 7. VWAP position (3 pts)
    if vwap and spot > vwap * 1.001:
        ce_score += 3
        reasons_ce.append(f"Spot {spot:.0f} above VWAP {vwap:.0f} — bullish")
    elif vwap and spot < vwap * 0.999:
        pe_score += 3
        reasons_pe.append(f"Spot {spot:.0f} below VWAP {vwap:.0f} — bearish")

    # ── Determine direction ───────────────────────────────────────────────────
    if session in ("PRE_MARKET", "CLOSED"):
        return _wait(f"Session: {session}", spot=spot, atm_strike=atm,
                     book_imbalance=imb, spread=spread, depth_level=depth.depth_level)

    best_dir   = "WAIT"
    best_score = 0
    best_reasons = []

    if ce_score >= pe_score and ce_score >= TRADE_THRESH:
        best_dir     = "BUY_CE"
        best_score   = ce_score
        best_reasons = reasons_ce
    elif pe_score > ce_score and pe_score >= TRADE_THRESH:
        best_dir     = "BUY_PE"
        best_score   = pe_score
        best_reasons = reasons_pe
    else:
        best_score   = max(ce_score, pe_score)
        best_reasons = (reasons_ce if ce_score >= pe_score else reasons_pe)

    strength = "WAIT"
    if best_dir != "WAIT":
        strength = "STRONG" if best_score >= STRONG_THRESH else "MODERATE"

    # OPENING: require STRONG only
    if session == "OPENING" and strength == "MODERATE":
        best_dir = "WAIT"
        strength = "WAIT"

    # ── Trade levels ─────────────────────────────────────────────────────────
    sl_pts     = round(max(spot * 0.003, 50), 0)   # min 50 pts
    target_pts = round(sl_pts * 2.0, 0)
    if best_dir == "BUY_CE":
        sl_spot     = round(spot - sl_pts, 0)
        target_spot = round(spot + target_pts, 0)
    elif best_dir == "BUY_PE":
        sl_spot     = round(spot + sl_pts, 0)
        target_spot = round(spot - target_pts, 0)
    else:
        sl_spot = target_spot = 0.0
    rr = round(target_pts / sl_pts, 1) if sl_pts else 0.0

    return DOMSignal(
        action=best_dir, strength=strength, score=best_score, max_score=20,
        spot=spot, atm_strike=atm, expiry=expiry,
        entry_ce=ce_ltp, entry_pe=pe_ltp,
        sl_pts=sl_pts, sl_spot=sl_spot, target_spot=target_spot, rr=rr,
        book_imbalance=round(imb, 3),
        order_imbalance=round(o_imb, 3),
        bid_qty_5=bid5, ask_qty_5=ask5,
        spread=round(spread, 2),
        bid_wall_price=bid_wall_price, bid_wall_qty=bid_wall_qty,
        ask_wall_price=ask_wall_price, ask_wall_qty=ask_wall_qty,
        depth_level=depth.depth_level,
        ce_doi=ce_doi, pe_doi=pe_doi,
        ce_oi=ce_oi, pe_oi=pe_oi,
        pcr=pcr, atm_iv=atm_iv, vwap=vwap,
        session=session, reasons=best_reasons,
        timestamp=now_ist,
    )


if __name__ == "__main__":
    sig = analyze_dom()
    print(f"\n{'='*60}")
    print(f"  NIFTY DOM SIGNAL  [{sig.action}] [{sig.strength}]")
    print(f"  Score: {sig.score}/{sig.max_score}  Threshold: {TRADE_THRESH}/20")
    print(f"{'='*60}")
    print(f"  Spot     : {sig.spot:.0f}")
    print(f"  Book Imb : {_imb_bar(sig.book_imbalance)} {sig.book_imbalance:+.3f}")
    print(f"  Bid5     : {sig.bid_qty_5:,} lots | Ask5: {sig.ask_qty_5:,} lots")
    print(f"  Orders   : {sig.order_imbalance:+.3f} | Spread: {sig.spread:.1f}pt")
    if sig.bid_wall_qty:
        print(f"  Bid Wall : {sig.bid_wall_qty:,} lots @ {sig.bid_wall_price:.0f}")
    if sig.ask_wall_qty:
        print(f"  Ask Wall : {sig.ask_wall_qty:,} lots @ {sig.ask_wall_price:.0f}")
    print(f"  OI Flow  : CE Δ{sig.ce_doi:+,} | PE Δ{sig.pe_doi:+,}")
    print(f"  PCR      : {sig.pcr:.2f} | IV: {sig.atm_iv:.1f}%")
    print(f"  Session  : {sig.session}")
    print(f"{'='*60}")
    if sig.is_trade():
        print(f"  ATM      : {sig.atm_strike}  [{sig.expiry}]")
        print(f"  CE LTP   : {sig.entry_ce:.0f}  PE LTP: {sig.entry_pe:.0f}")
        print(f"  SL Spot  : {sig.sl_spot:.0f}  ({sig.sl_pts:.0f} pts)")
        print(f"  Target   : {sig.target_spot:.0f}  R:R 1:{sig.rr}")
        print(f"{'='*60}")
    print(f"  Reasons:")
    for r in sig.reasons:
        print(f"    • {r}")
    print(f"{'='*60}\n")
