"""
Nifty Intraday Signal Engine
==============================
Runs the rule-based Nifty signal on a 5-min loop during market hours.
Sends Telegram alerts + saves to /tmp/nifty_signal.json

Run:
    python3 nifty/engine.py
    python3 nifty/engine.py --once   (single run, for testing)
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

POLL_INTERVAL   = 300      # 5 min between scans
SIGNAL_COOLDOWN = 600      # 10 min between same-direction alerts
ACTIVATION_PTS  = 10       # pts spot must move to confirm entry
SIG_FILE        = "/tmp/nifty_signal.json"
HISTORY_FILE    = ROOT / "data" / "nifty_signal_history.json"
MAX_HISTORY     = 200

MARKET_OPEN_H, MARKET_OPEN_M   = 9, 15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30

_last_action    = "WAIT"
_last_signal_ts = 0.0
_history: list  = []

# Pending signal awaiting activation confirmation
_pending_activation: dict | None = None


def _is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:   # Saturday=5, Sunday=6 — NSE closed
        return False
    mins = now.hour * 60 + now.minute
    open_mins  = MARKET_OPEN_H  * 60 + MARKET_OPEN_M
    close_mins = MARKET_CLOSE_H * 60 + MARKET_CLOSE_M
    return open_mins <= mins <= close_mins


def _mins_to_open() -> int:
    now  = datetime.now(IST)
    open_today = now.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
    if now > open_today:
        return 0
    return int((open_today - now).total_seconds() / 60)


def _load_history() -> list:
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text())
    except Exception:
        pass
    return []


def _save_history():
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(_history[-MAX_HISTORY:], indent=2, default=str))
    except Exception as e:
        logger.warning(f"[Nifty-Engine] History save error: {e}")


def _write_signal_json(sig):
    try:
        data = {
            "action":       sig.action,
            "strength":     sig.strength,
            "score":        sig.score,
            "max_score":    sig.max_score,
            "confidence":   sig.confidence,
            "spot":         sig.spot,
            "atm_strike":   sig.atm_strike,
            "expiry":       sig.expiry,
            "entry_ce":     sig.entry_ce,
            "entry_pe":     sig.entry_pe,
            "sl_pts":       sig.sl_pts,
            "sl_spot":      sig.sl_spot,
            "target_spot":  sig.target_spot,
            "rr":           sig.rr,
            "pcr":          sig.pcr,
            "atm_iv":       sig.atm_iv,
            "vwap":         sig.vwap,
            "trend_15m":    sig.trend_15m,
            "session":      sig.session,
            "reasons":      sig.reasons,
            "timestamp":    sig.timestamp.isoformat(),
        }
        with open(SIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"[Nifty-Engine] Signal file write error: {e}")


def _send_telegram(msg: str):
    try:
        from core.alerts.telegram_bot import send_message
        send_message(msg, parse_mode="HTML")
    except Exception as e:
        logger.warning(f"[Nifty-Engine] Telegram error: {e}")


def _should_alert(sig, prev_action: str, prev_ts: float) -> bool:
    now = time.time()
    if sig.action == "WAIT":
        return False
    if sig.action != prev_action:
        return True
    return (now - prev_ts) >= SIGNAL_COOLDOWN


def _check_activation(current_spot: float) -> str | None:
    """
    Returns a Telegram HTML message if the pending signal's entry is confirmed,
    or None if not yet activated. Clears pending on activation or SL breach.
    """
    global _pending_activation
    if not _pending_activation:
        return None

    action    = _pending_activation["action"]
    sig_spot  = _pending_activation["spot"]
    sl_spot   = _pending_activation["sl_spot"]
    tgt_spot  = _pending_activation["target_spot"]
    atm       = _pending_activation["atm_strike"]
    expiry    = _pending_activation["expiry"]
    entry_ce  = _pending_activation["entry_ce"]
    entry_pe  = _pending_activation["entry_pe"]

    activated = False
    sl_hit    = False

    if action == "BUY_CE":
        activated = current_spot >= sig_spot + ACTIVATION_PTS
        sl_hit    = current_spot <= sl_spot
    elif action == "BUY_PE":
        activated = current_spot <= sig_spot - ACTIVATION_PTS
        sl_hit    = current_spot >= sl_spot
    elif action == "SELL_STRADDLE":
        activated = abs(current_spot - atm) <= ACTIVATION_PTS
        sl_hit    = False  # straddle SL is premium-based, skip here

    if sl_hit:
        msg = (f"❌ <b>SIGNAL CANCELLED — SL HIT</b>\n"
               f"Action: {action} | Signal spot: {sig_spot:.0f}\n"
               f"Current spot: <b>{current_spot:.0f}</b> crossed SL {sl_spot:.0f}\n"
               f"⏱ {datetime.now(IST).strftime('%d %b %H:%M IST')}")
        _pending_activation = None
        return msg

    if activated:
        action_str = {"BUY_CE": "🟢 BUY CE", "BUY_PE": "🔴 BUY PE",
                      "SELL_STRADDLE": "⚡ SELL STRADDLE"}.get(action, action)
        if action == "BUY_CE":
            entry_line = f"💰 CE Entry  : <b>~{entry_ce:.0f}</b> (buy now)\n"
        elif action == "BUY_PE":
            entry_line = f"💰 PE Entry  : <b>~{entry_pe:.0f}</b> (buy now)\n"
        else:
            entry_line = f"💰 CE+PE     : <b>~{entry_ce:.0f} + {entry_pe:.0f}</b>\n"
        msg = (f"🚀 <b>TRADE ACTIVATED — {action_str}</b>\n"
               f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
               f"📍 Current Spot : <b>{current_spot:.0f}</b>\n"
               f"🎯 ATM          : <b>{atm}</b>  [{expiry}]\n"
               f"{entry_line}"
               f"🛑 SL Spot      : <b>{sl_spot:.0f}</b>\n"
               f"✅ Target       : <b>{tgt_spot:.0f}</b>\n"
               f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
               f"⏱ {datetime.now(IST).strftime('%d %b %H:%M IST')}")
        _pending_activation = None
        return msg

    return None


def _print_signal(sig):
    sep = "=" * 60
    if not sig.is_trade():
        logger.info(
            f"WAIT | score={sig.score}/{sig.max_score} | "
            f"PCR={sig.pcr:.2f} | IV={sig.atm_iv:.1f}% | "
            f"Session={sig.session} | {' | '.join(sig.reasons[:2])}"
        )
        return
    emoji = "BUY CE ▲" if sig.action == "BUY_CE" else ("SELL PE ▼" if sig.action == "BUY_PE" else "STRADDLE ⚡")
    print(f"\n{sep}")
    print(f"  {emoji}  NIFTY  [{sig.strength}]")
    print(sep)
    print(f"  Spot     : {sig.spot:.0f}")
    print(f"  ATM      : {sig.atm_strike}  [{sig.expiry}]")
    print(f"  CE LTP   : {sig.entry_ce:.0f}")
    print(f"  PE LTP   : {sig.entry_pe:.0f}")
    print(f"  SL Spot  : {sig.sl_spot:.0f}  ({sig.sl_pts:.0f} pts)")
    print(f"  Target   : {sig.target_spot:.0f}  R:R 1:{sig.rr}")
    print(sep)
    print(f"  PCR      : {sig.pcr:.3f}")
    print(f"  ATM IV   : {sig.atm_iv:.1f}%")
    print(f"  VWAP     : {sig.vwap:.0f}")
    print(f"  15m Trend: {sig.trend_15m}")
    print(f"  Session  : {sig.session}")
    print(f"  Score    : {sig.score}/{sig.max_score} ({sig.confidence:.0%} conf)")
    print(sep)
    print(f"  Reasons  : {' | '.join(sig.reasons)}")
    print(f"  Time     : {sig.timestamp}")
    print(f"{sep}\n")


def run(once: bool = False):
    global _last_action, _last_signal_ts, _history, _pending_activation
    _history = _load_history()

    from nifty.strategy import analyze_nifty

    print("=" * 60)
    print("  Nifty Intraday Signal Engine  v1")
    print("  VWAP + EMA + PCR + IV + ATM OI | 5-min scan")
    print(f"  Market: 09:15–15:30 IST | Threshold: 12/20 | Strong: 16/20")
    print("=" * 60)

    while True:
        now_ist = datetime.now(IST)

        if not _is_market_open() and not once:
            mins = _mins_to_open()
            if mins > 0:
                print(f"\r  Market opens in {mins//60:02d}h {mins%60:02d}m   ", end="", flush=True)
            else:
                print(f"\r  Market closed (15:30 IST)   ", end="", flush=True)
            time.sleep(30)
            continue

        print()  # newline after countdown
        try:
            logger.info(f"[Nifty-Engine] Scanning at {now_ist.strftime('%H:%M IST')}...")
            sig = analyze_nifty()
            _write_signal_json(sig)
            _print_signal(sig)

            # Append to history
            _history.append({
                "action":   sig.action,
                "strength": sig.strength,
                "score":    sig.score,
                "spot":     sig.spot,
                "pcr":      sig.pcr,
                "atm_iv":   sig.atm_iv,
                "session":  sig.session,
                "reasons":  sig.reasons,
                "timestamp": str(sig.timestamp),
            })
            _save_history()

            # Check if a pending signal has been activated
            activation_msg = _check_activation(sig.spot)
            if activation_msg:
                logger.info("[Nifty-Engine] Trade activated — sending alert...")
                _send_telegram(activation_msg)

            # New signal alert
            if _should_alert(sig, _last_action, _last_signal_ts):
                logger.info(f"[Nifty-Engine] Sending {sig.action} alert...")
                _send_telegram(sig.telegram_html())
                _last_action    = sig.action
                _last_signal_ts = time.time()
                # Register for activation tracking
                if sig.is_trade():
                    _pending_activation = {
                        "action":       sig.action,
                        "spot":         sig.spot,
                        "atm_strike":   sig.atm_strike,
                        "expiry":       sig.expiry,
                        "entry_ce":     sig.entry_ce,
                        "entry_pe":     sig.entry_pe,
                        "sl_spot":      sig.sl_spot,
                        "target_spot":  sig.target_spot,
                    }

        except Exception as e:
            logger.error(f"[Nifty-Engine] Loop error: {e}")
            import traceback; traceback.print_exc()

        if once:
            break
        logger.info(f"[Nifty-Engine] Sleeping {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once",  action="store_true", help="Run once and exit")
    ap.add_argument("--debug", action="store_true", help="Verbose logging")
    ap.add_argument("--force", action="store_true", help="Run even if market is closed (testing)")
    args = ap.parse_args()

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    if args.force:
        # Override market hours check for testing
        from nifty import strategy as _strat
        _strat._get_session = lambda _: "MID_SESSION"

    run(once=args.once)
