"""
XAUUSD Live Trading Engine
============================
- Polls every 60s
- 15m entry + 1H trend filter + macro (DXY, 10Y, VIX)
- Prints signal card to terminal
- Sends Telegram on every NEW actionable signal

Run:
    python3 xauusd/engine.py
    python3 xauusd/engine.py --once        # single check, no loop
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from xauusd.data     import get_bars, get_macro, get_price
from xauusd.strategy import analyze

POLL_SECS = 60
DIVIDER   = "=" * 50


# ── Telegram helper ────────────────────────────────────────────────────────────

def _telegram(text: str):
    try:
        from core.alerts.telegram_bot import send_message
        send_message(text)
    except Exception as e:
        logger.debug(f"Telegram: {e}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(once: bool = False):
    print(DIVIDER)
    print("  XAUUSD Live Signal Engine")
    print("  Timeframe : 15m entry + 1H trend")
    print("  Sessions  : London + NY (07-21 UTC)")
    print("  SL / TP   : 1.5x / 2.5x ATR(14)")
    print(DIVIDER)

    last_poll    = 0.0
    last_action  = None
    prev_macro   = {}

    _telegram(
        "🟡 <b>XAUUSD Engine Started</b>\n"
        "Monitoring Gold 24/7\n"
        "Signals fire on London + NY sessions.\n"
        f"Started: {datetime.now(timezone.utc).strftime('%d %b %Y  %H:%M UTC')}"
    )

    while True:
        now_ts = time.time()

        if now_ts - last_poll >= POLL_SECS or once:
            last_poll = now_ts
            try:
                # ── Fetch data ──────────────────────────────────────────────
                logger.info("Fetching XAUUSD 15m + 1H bars...")
                df_15m = get_bars("15m", "5d")
                df_1h  = get_bars("1h",  "60d")
                macro  = get_macro()

                # Carry over previous values for direction
                macro["dxy_prev"]    = prev_macro.get("dxy",   macro["dxy"])
                macro["us10y_prev"]  = prev_macro.get("us10y", macro["us10y"])
                prev_macro = macro.copy()

                logger.info(
                    f"15m bars={len(df_15m)}  1H bars={len(df_1h)}  "
                    f"DXY={macro['dxy']:.2f}  10Y={macro['us10y']:.2f}%  VIX={macro['vix']:.1f}"
                )

                # ── Analyze ─────────────────────────────────────────────────
                signal = analyze(df_15m, df_1h, macro)

                # ── Print card ──────────────────────────────────────────────
                print()
                print(DIVIDER)
                print(signal.card())
                print(DIVIDER)
                print()

                # ── Telegram on new actionable signal ───────────────────────
                if signal.is_trade() and signal.action != last_action:
                    logger.info(f"*** NEW SIGNAL: {signal.action} @ ${signal.entry:.2f} ***")
                    _telegram(signal.telegram_html())

                # ── Telegram when signal clears (new WAIT after a trade) ────
                elif not signal.is_trade() and last_action in ("BUY", "SELL"):
                    _telegram(
                        f"⏳ <b>Signal cleared — XAUUSD</b>\n"
                        f"Previous: {last_action}\n"
                        f"Now: WAIT ({signal.reason[:150]})\n"
                        f"Price: ${signal.entry:.2f}\n"
                        f"⏰ {signal.timestamp.strftime('%d %b %Y  %H:%M UTC')}"
                    )

                last_action = signal.action

                # Save latest signal to file for dashboard
                import json
                sig_dict = {
                    "action":      signal.action,
                    "entry":       signal.entry,
                    "stop_loss":   signal.stop_loss,
                    "target":      signal.target,
                    "risk_reward": signal.risk_reward,
                    "confidence":  signal.confidence,
                    "score":       signal.score,
                    "reason":      signal.reason,
                    "atr":         signal.atr,
                    "rsi":         signal.rsi,
                    "trend_1h":    signal.trend_1h,
                    "session":     signal.session,
                    "macro":       signal.macro,
                    "timestamp":   signal.timestamp.isoformat(),
                }
                with open("/tmp/xauusd_signal.json", "w") as f:
                    json.dump(sig_dict, f)

                # Append to history (keep last 50)
                hist_path = "/tmp/xauusd_history.json"
                try:
                    with open(hist_path) as f:
                        history = json.load(f)
                except Exception:
                    history = []
                history.append(sig_dict)
                history = history[-50:]
                with open(hist_path, "w") as f:
                    json.dump(history, f)

            except Exception as e:
                logger.error(f"Poll error: {e}", exc_info=True)

        if once:
            break

        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Single check then exit")
    args = parser.parse_args()
    run(once=args.once)
