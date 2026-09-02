"""
XAUUSD Order Flow Engine
=========================
Runs the heavy order flow strategy on a loop.
Sends Telegram alerts + saves to /tmp/xauusd_of_signal.json

Run:
    python3 xauusd/of_engine.py

Flags:
    --once   Run once and exit (for cron / testing)
    --debug  Verbose logging
"""

from __future__ import annotations
import sys, os, json, time, argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger

# ── Config ─────────────────────────────────────────────────────────────────────
POLL_INTERVAL   = 300      # seconds between full analyses (5 min)
SIGNAL_COOLDOWN = 900      # 15 min between same-direction alerts
SIG_FILE        = "/tmp/xauusd_of_signal.json"
HISTORY_FILE    = ROOT / "data" / "xauusd_of_trades.json"
MAX_HISTORY     = 200

_last_action    = "WAIT"
_last_signal_ts = 0.0
_history: list  = []


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
        logger.warning(f"[OF-Engine] History save error: {e}")


def _write_signal_json(sig):
    try:
        data = {
            "action":       sig.action,
            "strength":     sig.strength,
            "entry":        sig.entry,
            "stop_loss":    sig.stop_loss,
            "target1":      sig.target1,
            "target2":      sig.target2,
            "target3":      sig.target3,
            "risk_reward":  sig.risk_reward,
            "score":        sig.score,
            "confidence":   sig.confidence,
            "atr":          sig.atr,
            "killzone":     sig.killzone,
            "htf_bias":     sig.htf_bias,
            "structure":    sig.structure,
            "ob_level":     sig.ob_level,
            "ob_type":      sig.ob_type,
            "fvg_top":      sig.fvg_top,
            "fvg_bot":      sig.fvg_bot,
            "fvg_type":     sig.fvg_type,
            "poc":          sig.poc,
            "vah":          sig.vah,
            "val":          sig.val,
            "vwap":         sig.vwap,
            "cvd_bias":     sig.cvd_bias,
            "liq_swept":    sig.liq_swept,
            "liq_level":    sig.liq_level,
            "reasons":      sig.reasons,
            "timestamp":    sig.timestamp.isoformat() if hasattr(sig.timestamp, "isoformat") else str(sig.timestamp),
        }
        with open(SIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"[OF-Engine] Signal file write error: {e}")


def _send_telegram(msg: str):
    try:
        from core.alerts.telegram_bot import send_message
        send_message(msg, parse_mode="HTML")
    except Exception as e:
        logger.warning(f"[OF-Engine] Telegram error: {e}")


def _should_alert(sig, prev_action: str, prev_ts: float) -> bool:
    now = time.time()
    if sig.action == "WAIT":
        return False
    if sig.action != prev_action:
        return True
    # Same direction — only re-alert after cooldown
    return (now - prev_ts) >= SIGNAL_COOLDOWN


def _print_signal(sig):
    sep = "=" * 60
    if not sig.is_trade():
        logger.info(f"WAIT | score={sig.score}/20 | HTF={sig.htf_bias} | KZ={sig.killzone}")
        return
    emoji = "BUY ▲" if sig.action == "BUY" else "SELL ▼"
    print(f"\n{sep}")
    print(f"  {emoji}  XAUUSD  [{sig.strength}]")
    print(sep)
    print(f"  Entry    : ${sig.entry:.2f}")
    print(f"  SL       : ${sig.stop_loss:.2f}  ({abs(sig.entry-sig.stop_loss):.2f} pts)")
    print(f"  TP1      : ${sig.target1:.2f}")
    print(f"  TP2      : ${sig.target2:.2f}")
    print(f"  TP3      : ${sig.target3:.2f}")
    print(f"  R:R      : 1:{sig.risk_reward:.1f}   Score: {sig.score}/20 ({sig.confidence:.0%})")
    print(sep)
    print(f"  HTF Bias : {sig.htf_bias}")
    print(f"  Structure: {sig.structure}")
    print(f"  OB       : {sig.ob_type} @ {sig.ob_level:.2f}" if sig.ob_level else "  OB       : none")
    print(f"  FVG      : {sig.fvg_type} [{sig.fvg_bot:.2f}-{sig.fvg_top:.2f}]" if sig.fvg_top else "  FVG      : none")
    print(f"  VP       : POC={sig.poc:.2f}  VAH={sig.vah:.2f}  VAL={sig.val:.2f}")
    print(f"  VWAP     : {sig.vwap:.2f}")
    print(f"  CVD      : {sig.cvd_bias}")
    print(f"  Killzone : {sig.killzone or 'none'}")
    print(f"  Liq Swept: {'YES @ '+str(sig.liq_level) if sig.liq_swept else 'NO'}")
    print(sep)
    print(f"  Reasons  : {' | '.join(sig.reasons)}")
    print(f"  Time     : {sig.timestamp}")
    print(f"{sep}\n")


def run(once: bool = False):
    global _last_action, _last_signal_ts, _history

    _history = _load_history()

    from xauusd.of_strategy import analyze_order_flow, _fetch

    print("=" * 60)
    print("  XAUUSD Order Flow Engine")
    print("  Strategy: SMC + Volume Profile + CVD + VWAP + Killzones")
    print("  Interval: 5 min  |  Threshold: 11/20")
    print("=" * 60)

    while True:
        try:
            logger.info("[OF-Engine] Fetching multi-timeframe data...")
            df_15m = _fetch("15m", "5d")
            df_1h  = _fetch("1h",  "60d")
            df_4h  = _fetch("4h",  "120d")

            sig = analyze_order_flow(df_15m, df_1h, df_4h)
            _write_signal_json(sig)
            _print_signal(sig)

            # Append to history
            _history.append({
                "action":     sig.action,
                "strength":   sig.strength,
                "entry":      sig.entry,
                "stop_loss":  sig.stop_loss,
                "target1":    sig.target1,
                "target2":    sig.target2,
                "score":      sig.score,
                "htf_bias":   sig.htf_bias,
                "killzone":   sig.killzone,
                "reasons":    sig.reasons,
                "timestamp":  str(sig.timestamp),
            })
            _save_history()

            # Telegram alert
            if _should_alert(sig, _last_action, _last_signal_ts):
                logger.info(f"[OF-Engine] Sending {sig.action} alert...")
                _send_telegram(sig.telegram_html())
                _last_action    = sig.action
                _last_signal_ts = time.time()

        except Exception as e:
            logger.error(f"[OF-Engine] Loop error: {e}")

        if once:
            break
        logger.info(f"[OF-Engine] Sleeping {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once",  action="store_true", help="Run once and exit")
    ap.add_argument("--debug", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    run(once=args.once)
