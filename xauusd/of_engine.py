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
ACTIVATION_PTS  = 2.0     # $2 move in trade direction to confirm entry
SIG_FILE        = "/tmp/xauusd_of_signal.json"
HISTORY_FILE    = ROOT / "data" / "xauusd_of_trades.json"
MAX_HISTORY     = 200

_last_action    = "WAIT"
_last_signal_ts = 0.0
_history: list  = []
_active_trade   = None   # {entry, sl, tp1, tp2, direction, reasons, open_ts, activated}


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


def _write_signal_json(sig, news_ctx: dict | None = None):
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
            "max_score":    sig.max_score,
            "timestamp":    sig.timestamp.isoformat() if hasattr(sig.timestamp, "isoformat") else str(sig.timestamp),
            "news_sentiment":      news_ctx.get("sentiment", "NEUTRAL") if news_ctx else "NEUTRAL",
            "news_sentiment_score": news_ctx.get("sentiment_score", 0.0) if news_ctx else 0.0,
            "news_filter":         news_ctx.get("news_filter", False) if news_ctx else False,
            "news_filter_reason":  news_ctx.get("news_filter_reason", "") if news_ctx else "",
            "news_headlines":      (news_ctx.get("headlines", [])[:6]) if news_ctx else [],
            "news_calendar":       (news_ctx.get("calendar", [])[:8]) if news_ctx else [],
            "news_fetched_at":     news_ctx.get("fetched_at", "") if news_ctx else "",
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
        logger.info(f"WAIT | score={sig.score}/{sig.max_score} | HTF={sig.htf_bias} | KZ={sig.killzone or 'none'} | {' | '.join(sig.reasons[:2])}")
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

    global _last_action, _last_signal_ts, _history, _active_trade
    _history = _load_history()

    from xauusd.of_strategy import analyze_order_flow, _fetch
    from xauusd.score_learner import record_trade_outcome
    from xauusd.news_feed import get_news_context, print_news_context, news_summary_line

    print("=" * 60)
    print("  XAUUSD Order Flow Engine  v3")
    print("  SMC + VP + CVD + Tape + OTE + P/D + EQH/L + Killzones + News")
    print("  Interval: 5 min  |  Threshold: 22/40  |  Strong: 28/40")
    print("=" * 60)

    while True:
        try:
            logger.info("[OF-Engine] Fetching multi-timeframe data (5m/15m/1H/4H)...")
            df_5m  = _fetch("5m",  "3d")
            df_15m = _fetch("15m", "5d")
            df_1h  = _fetch("1h",  "60d")
            df_4h  = _fetch("4h",  "120d")

            # ── News + calendar ────────────────────────────────────────────
            logger.info("[OF-Engine] Fetching news + economic calendar...")
            news_ctx = get_news_context()
            print_news_context(news_ctx)

            sig = analyze_order_flow(df_15m, df_1h, df_4h, df_5m)
            _write_signal_json(sig, news_ctx)
            _print_signal(sig)

            # If news filter active, warn + send Telegram alert
            if news_ctx["news_filter"]:
                logger.warning(f"[OF-Engine] NEWS FILTER: {news_ctx['news_filter_reason']} — high-impact event imminent!")
                _send_telegram(f"⚠️ <b>NEWS FILTER</b>: {news_ctx['news_filter_reason']}\n"
                               f"Avoid new trades — high-impact USD event imminent!")

            # ── Track active trade SL/TP for learning ─────────────────────────
            from xauusd.data import get_price as _get_price
            _live = _get_price()
            price_now = _live if _live > 0 else sig.entry
            # Check activation (price moved ACTIVATION_PTS in trade direction)
            if _active_trade and not _active_trade.get("activated"):
                t = _active_trade
                if (t["direction"] == "BUY"  and price_now >= t["entry"] + ACTIVATION_PTS) or \
                   (t["direction"] == "SELL" and price_now <= t["entry"] - ACTIVATION_PTS):
                    _active_trade["activated"] = True
                    action_emoji = "🟢 BUY" if t["direction"] == "BUY" else "🔴 SELL"
                    _send_telegram(
                        f"🚀 <b>TRADE ACTIVATED — {action_emoji} XAUUSD (OF)</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📍 Current Price : <b>${price_now:.2f}</b>\n"
                        f"💰 Entry         : <b>${t['entry']:.2f}</b>\n"
                        f"🛑 Stop Loss     : <b>${t['sl']:.2f}</b>\n"
                        f"✅ TP1/TP2       : <b>${t['tp1']:.2f} / ${t['tp2']:.2f}</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"⏰ {datetime.now(timezone.utc).strftime('%d %b %Y  %H:%M UTC')}"
                    )
                    logger.info(f"[OF-Engine] Trade activated @ ${price_now:.2f}")
            if _active_trade:
                t = _active_trade
                if t["direction"] == "BUY":
                    if price_now <= t["sl"]:
                        logger.warning(f"[OF-Engine] TRADE CLOSED — SL HIT — LOSS @ {price_now:.2f}")
                        record_trade_outcome(t["reasons"], "LOSS")
                        _active_trade = None
                        _send_telegram(f"🛑 <b>OF SL HIT</b> @ ${price_now:.2f} | Learning from loss...")
                    elif price_now >= t["tp2"]:
                        logger.info(f"[OF-Engine] TRADE CLOSED — TP2 HIT — WIN @ {price_now:.2f}")
                        record_trade_outcome(t["reasons"], "WIN")
                        _active_trade = None
                        _send_telegram(f"🎯 <b>OF TP2 HIT</b> @ ${price_now:.2f} | Boosting winning factors!")
                elif t["direction"] == "SELL":
                    if price_now >= t["sl"]:
                        logger.warning(f"[OF-Engine] TRADE CLOSED — SL HIT — LOSS @ {price_now:.2f}")
                        record_trade_outcome(t["reasons"], "LOSS")
                        _active_trade = None
                        _send_telegram(f"🛑 <b>OF SL HIT</b> @ ${price_now:.2f} | Learning from loss...")
                    elif price_now <= t["tp2"]:
                        logger.info(f"[OF-Engine] TRADE CLOSED — TP2 HIT — WIN @ {price_now:.2f}")
                        record_trade_outcome(t["reasons"], "WIN")
                        _active_trade = None
                        _send_telegram(f"🎯 <b>OF TP2 HIT</b> @ ${price_now:.2f} | Boosting winning factors!")

            if sig.is_trade() and _should_alert(sig, _last_action, _last_signal_ts):
                _active_trade = {
                    "direction": sig.action, "entry": sig.entry,
                    "sl": sig.stop_loss, "tp1": sig.target1, "tp2": sig.target2,
                    "reasons": sig.reasons, "open_ts": str(sig.timestamp),
                    "activated": False,
                }

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
