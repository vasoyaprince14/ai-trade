"""
Nifty DOM Engine
================
Polls Dhan 20-level order book every 30 seconds during market hours.
Sends Telegram alerts when DOM order flow signals fire.

Alerts sent:
  - New DOM signal (entry)
  - Trade activated (spot moves 10 pts in signal direction)
  - Signal cancelled (spot hits SL before activation)
  - Reversal (DOM flips direction)

Run:
    python3 nifty/dom_engine.py
    python3 nifty/dom_engine.py --once     # single scan
    python3 nifty/dom_engine.py --debug    # verbose
"""

from __future__ import annotations
import sys, os, json, time, argparse
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
import pytz

IST = pytz.timezone("Asia/Kolkata")

POLL_SECS       = 30
SIGNAL_COOLDOWN = 300      # 5 min between same-direction alerts
ACTIVATION_PTS  = 10       # spot must move 10 pts to confirm entry
SIG_FILE        = "/tmp/nifty_dom_signal.json"

MARKET_OPEN_H,  MARKET_OPEN_M  = 9, 15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30

_last_action    = "WAIT"
_last_signal_ts = 0.0
_pending: dict | None = None   # pending activation tracking


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    mins       = now.hour * 60 + now.minute
    open_mins  = MARKET_OPEN_H  * 60 + MARKET_OPEN_M
    close_mins = MARKET_CLOSE_H * 60 + MARKET_CLOSE_M
    return open_mins <= mins <= close_mins


def _tg(msg: str):
    try:
        from core.alerts.telegram_bot import send_message
        send_message(msg, parse_mode="HTML")
    except Exception as e:
        logger.debug(f"[DOM] Telegram: {e}")


def _write_sig_json(sig):
    try:
        data = {
            "action":          sig.action,
            "strength":        sig.strength,
            "score":           sig.score,
            "max_score":       sig.max_score,
            "spot":            sig.spot,
            "atm_strike":      sig.atm_strike,
            "expiry":          sig.expiry,
            "entry_ce":        sig.entry_ce,
            "entry_pe":        sig.entry_pe,
            "sl_pts":          sig.sl_pts,
            "sl_spot":         sig.sl_spot,
            "target_spot":     sig.target_spot,
            "rr":              sig.rr,
            "book_imbalance":  sig.book_imbalance,
            "order_imbalance": sig.order_imbalance,
            "bid_qty_5":       sig.bid_qty_5,
            "ask_qty_5":       sig.ask_qty_5,
            "spread":          sig.spread,
            "bid_wall_price":  sig.bid_wall_price,
            "bid_wall_qty":    sig.bid_wall_qty,
            "ask_wall_price":  sig.ask_wall_price,
            "ask_wall_qty":    sig.ask_wall_qty,
            "ce_doi":          sig.ce_doi,
            "pe_doi":          sig.pe_doi,
            "ce_oi":           sig.ce_oi,
            "pe_oi":           sig.pe_oi,
            "pcr":             sig.pcr,
            "atm_iv":          sig.atm_iv,
            "vwap":            sig.vwap,
            "session":         sig.session,
            "reasons":         sig.reasons,
            "pending":         _pending,
            "timestamp":       sig.timestamp.isoformat(),
        }
        with open(SIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _check_activation(spot: float) -> str | None:
    """
    Returns Telegram message if pending signal is activated or cancelled.
    Clears _pending on either outcome.
    """
    global _pending
    if not _pending:
        return None

    action     = _pending["action"]
    sig_spot   = _pending["spot"]
    sl_spot    = _pending["sl_spot"]
    tgt_spot   = _pending["target_spot"]
    atm        = _pending["atm_strike"]
    expiry     = _pending["expiry"]
    entry_ce   = _pending["entry_ce"]
    entry_pe   = _pending["entry_pe"]

    activated = sl_hit = False
    if action == "BUY_CE":
        activated = spot >= sig_spot + ACTIVATION_PTS
        sl_hit    = spot <= sl_spot
    elif action == "BUY_PE":
        activated = spot <= sig_spot - ACTIVATION_PTS
        sl_hit    = spot >= sl_spot

    if sl_hit:
        _pending = None
        return (f"❌ <b>DOM SIGNAL CANCELLED — SL HIT</b>\n"
                f"Action: {action} | Signal spot: {sig_spot:.0f}\n"
                f"Current spot: <b>{spot:.0f}</b> hit SL {sl_spot:.0f}\n"
                f"⏱ {datetime.now(IST).strftime('%d %b %H:%M IST')}")

    if activated:
        action_str = "🟢 BUY CE" if action == "BUY_CE" else "🔴 BUY PE"
        entry_line = (f"💰 CE Entry: <b>~{entry_ce:.0f}</b>" if action == "BUY_CE"
                      else f"💰 PE Entry: <b>~{entry_pe:.0f}</b>")
        _pending = None
        return (f"🚀 <b>DOM TRADE ACTIVATED — {action_str}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📍 Spot: <b>{spot:.0f}</b>  (moved {abs(spot - sig_spot):.0f} pts)\n"
                f"🎯 ATM: <b>{atm}</b>  [{expiry}]\n"
                f"{entry_line}\n"
                f"🛑 SL: <b>{sl_spot:.0f}</b> | ✅ Target: <b>{tgt_spot:.0f}</b>\n"
                f"⏱ {datetime.now(IST).strftime('%d %b %H:%M IST')}")

    return None


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(once: bool = False):
    global _last_action, _last_signal_ts, _pending

    from nifty.dom_strategy import analyze_dom

    print("=" * 60)
    print("  Nifty DOM Engine  v1")
    print("  Source: Dhan 20-level real order book")
    print(f"  Threshold: {10}/20 | Strong: {14}/20 | Activation: {ACTIVATION_PTS} pts")
    print("  Poll: 30s | Sessions: 09:15-15:30 IST")
    print("=" * 60)

    while True:
        if not _is_market_open() and not once:
            now_ist = datetime.now(IST)
            print(f"\r  Market closed — {now_ist.strftime('%a %H:%M IST')}   ", end="", flush=True)
            time.sleep(30)
            continue

        try:
            sig = analyze_dom()
            _write_sig_json(sig)

            # Check pending activation
            if _pending:
                act_msg = _check_activation(sig.spot)
                if act_msg:
                    logger.info("[DOM] Sending activation alert")
                    _tg(act_msg)

            # Clear stale pending when signal drops to WAIT
            if sig.action == "WAIT" and _pending:
                _pending = None

            # Log
            imb_str = f"{sig.book_imbalance:+.2f}"
            if sig.is_trade():
                logger.info(
                    f"[DOM] *** {sig.action} score={sig.score}/{sig.max_score} "
                    f"[{sig.strength}] imb={imb_str} spot={sig.spot:.0f} ***"
                )
            else:
                logger.info(
                    f"[DOM] WAIT score={sig.score}/{sig.max_score} "
                    f"imb={imb_str} bid={sig.bid_qty_5:,} ask={sig.ask_qty_5:,} "
                    f"spot={sig.spot:.0f} session={sig.session}"
                )

            # Alert logic
            now_ts  = time.time()
            elapsed = now_ts - _last_signal_ts
            should_alert = (
                sig.is_trade()
                and _pending is None
                and (sig.action != _last_action or elapsed >= SIGNAL_COOLDOWN)
            )

            if should_alert:
                _tg(sig.telegram_html())
                _last_action    = sig.action
                _last_signal_ts = now_ts
                _pending = {
                    "action":      sig.action,
                    "spot":        sig.spot,
                    "sl_spot":     sig.sl_spot,
                    "target_spot": sig.target_spot,
                    "atm_strike":  sig.atm_strike,
                    "expiry":      sig.expiry,
                    "entry_ce":    sig.entry_ce,
                    "entry_pe":    sig.entry_pe,
                }
                logger.info(f"[DOM] Alert sent: {sig.action} | Watching for {ACTIVATION_PTS}pt activation")

            # Reversal: DOM flipped while pending
            elif _pending and sig.is_trade() and sig.action != _pending["action"]:
                _tg(f"↩️ <b>DOM REVERSAL</b>\n"
                    f"Was: {_pending['action']} | Now: {sig.action}\n"
                    f"Spot: {sig.spot:.0f} | Score: {sig.score}/{sig.max_score}\n"
                    f"⏱ {datetime.now(IST).strftime('%H:%M IST')}")
                _pending     = None
                _last_action = "WAIT"

        except Exception as e:
            logger.error(f"[DOM] Loop error: {e}", exc_info=True)

        if once:
            break
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once",  action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    run(once=args.once)
