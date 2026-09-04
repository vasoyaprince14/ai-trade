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
import json
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

POLL_SECS        = 60
DIVIDER          = "=" * 50
SIGNAL_COOLDOWN  = 900    # 15 min — don't resend same direction unless entry moves > ATR*0.5
ACTIVATION_PTS   = 2.0    # $2 move in trade direction to confirm entry
TRADE_LOG_PATH   = ROOT / "data" / "xauusd_trades.json"
HIST_PATH        = "/tmp/xauusd_history.json"
SIG_PATH         = "/tmp/xauusd_signal.json"


# ── Telegram helper ────────────────────────────────────────────────────────────

def _telegram(text: str):
    try:
        from core.alerts.telegram_bot import send_message
        send_message(text)
    except Exception as e:
        logger.debug(f"Telegram: {e}")


# ── Trade Tracker — monitors open trades for SL/Target hits ───────────────────

class TradeTracker:
    """
    Keeps one active trade at a time.
    On each poll, checks current price against SL and Target.
    Sends Telegram alert when either is hit.
    Saves P&L log to disk.
    """

    def __init__(self):
        self.active : dict | None = None   # current open trade
        self._log   : list        = self._load_log()

    def _load_log(self) -> list:
        try:
            TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            if TRADE_LOG_PATH.exists():
                return json.loads(TRADE_LOG_PATH.read_text())
        except Exception:
            pass
        return []

    def _save_log(self):
        try:
            TRADE_LOG_PATH.write_text(json.dumps(self._log, indent=2))
        except Exception as e:
            logger.warning(f"Trade log save error: {e}")

    def open_trade(self, signal):
        """Record a new trade when signal fires."""
        if self.active:
            return   # already tracking one
        self.active = {
            "id":        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            "action":    signal.action,
            "entry":     signal.entry,
            "stop_loss": signal.stop_loss,
            "target":    signal.target,
            "rr":        signal.risk_reward,
            "score":     signal.score,
            "session":   signal.session,
            "open_ts":   datetime.now(timezone.utc).isoformat(),
            "close_ts":  None,
            "outcome":   None,   # "TP_HIT" | "SL_HIT" | "MANUAL"
            "close_px":  None,
            "pnl_pts":   None,
            "activated": False,
        }
        logger.info(f"[TradeTracker] Trade opened: {self.active['action']} @ {self.active['entry']:.2f}")

    def check_activation(self, current_price: float) -> str | None:
        """Returns a Telegram message when price moves ACTIVATION_PTS in trade direction."""
        if not self.active or self.active.get("activated"):
            return None
        action = self.active["action"]
        entry  = self.active["entry"]
        sl     = self.active["stop_loss"]
        tp     = self.active["target"]
        if action == "BUY" and current_price >= entry + ACTIVATION_PTS:
            self.active["activated"] = True
        elif action == "SELL" and current_price <= entry - ACTIVATION_PTS:
            self.active["activated"] = True
        else:
            return None
        action_emoji = "🟢 BUY" if action == "BUY" else "🔴 SELL"
        return (f"🚀 <b>TRADE ACTIVATED — {action_emoji} XAUUSD</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📍 Current Price : <b>${current_price:.2f}</b>\n"
                f"💰 Entry         : <b>${entry:.2f}</b>\n"
                f"🛑 Stop Loss     : <b>${sl:.2f}</b>\n"
                f"✅ Target        : <b>${tp:.2f}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ {datetime.now(timezone.utc).strftime('%d %b %Y  %H:%M UTC')}")

    def check(self, current_price: float) -> str | None:
        """
        Call on every poll with current price.
        Returns outcome string if trade closed, else None.
        """
        if not self.active:
            return None

        action = self.active["action"]
        entry  = self.active["entry"]
        sl     = self.active["stop_loss"]
        tp     = self.active["target"]

        outcome = None
        close_px = current_price

        if action == "BUY":
            if current_price <= sl:
                outcome  = "SL_HIT"
                pnl_pts  = sl - entry
            elif current_price >= tp:
                outcome  = "TP_HIT"
                pnl_pts  = tp - entry
            else:
                return None
        else:  # SELL
            if current_price >= sl:
                outcome  = "SL_HIT"
                pnl_pts  = entry - sl
                pnl_pts  = -abs(pnl_pts)
            elif current_price <= tp:
                outcome  = "TP_HIT"
                pnl_pts  = entry - tp
            else:
                return None

        # Trade closed
        pnl_pts = round(current_price - entry if action == "BUY" else entry - current_price, 2)
        self.active.update({
            "close_ts":  datetime.now(timezone.utc).isoformat(),
            "outcome":   outcome,
            "close_px":  round(current_price, 2),
            "pnl_pts":   pnl_pts,
        })
        self._log.append(self.active)
        self._save_log()

        closed = self.active
        self.active = None
        logger.info(f"[TradeTracker] Trade closed: {outcome} | PnL={pnl_pts:+.2f} pts")
        return outcome, closed

    def stats(self) -> dict:
        """Return P&L summary from trade log."""
        if not self._log:
            return {"trades": 0}
        closed  = [t for t in self._log if t.get("outcome")]
        wins    = [t for t in closed if t["outcome"] == "TP_HIT"]
        losses  = [t for t in closed if t["outcome"] == "SL_HIT"]
        pnls    = [t["pnl_pts"] for t in closed if t["pnl_pts"] is not None]
        return {
            "trades":   len(closed),
            "wins":     len(wins),
            "losses":   len(losses),
            "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "total_pnl": round(sum(pnls), 2),
            "avg_win":   round(sum(p for p in pnls if p > 0) / max(len(wins),1), 2),
            "avg_loss":  round(sum(p for p in pnls if p < 0) / max(len(losses),1), 2),
            "best":      round(max(pnls), 2) if pnls else 0,
            "worst":     round(min(pnls), 2) if pnls else 0,
        }

    def log(self) -> list:
        return self._log


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(once: bool = False):
    print(DIVIDER)
    print("  XAUUSD Live Signal Engine  v2")
    print("  Timeframe : 15m entry + 1H trend")
    print("  Sessions  : London + NY (07-21 UTC)")
    print("  SL / TP   : 1.5x / 2.5x ATR(14)")
    print("  Cooldown  : 15min between same-direction signals")
    print("  Tracking  : SL/TP hit alerts + P&L log")
    print(DIVIDER)

    last_poll       = 0.0
    last_action     = None
    last_signal_ts  = 0.0     # when we last sent a trade signal
    last_entry      = 0.0     # entry of last sent signal
    prev_macro      = {}
    tracker         = TradeTracker()

    _telegram(
        "🟡 <b>XAUUSD Engine v2 Started</b>\n"
        "✅ Signal cooldown: 15min\n"
        "✅ SL/Target hit alerts active\n"
        "✅ P&L tracking enabled\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%d %b %Y  %H:%M UTC')}"
    )

    while True:
        now_ts = time.time()

        if now_ts - last_poll >= POLL_SECS or once:
            last_poll = now_ts
            try:
                # ── Fetch data ──────────────────────────────────────────────
                df_15m = get_bars("15m", "5d")
                df_1h  = get_bars("1h",  "60d")
                macro  = get_macro()
                macro["dxy_prev"]   = prev_macro.get("dxy",   macro["dxy"])
                macro["us10y_prev"] = prev_macro.get("us10y", macro["us10y"])
                prev_macro = macro.copy()

                signal = analyze(df_15m, df_1h, macro)
                current_price = signal.entry

                # ── Check open trade for activation ─────────────────────────
                act_msg = tracker.check_activation(current_price)
                if act_msg:
                    logger.info("[TradeTracker] Trade activated — sending alert...")
                    _telegram(act_msg)

                # ── Check open trade for SL/TP ──────────────────────────────
                if tracker.active:
                    result = tracker.check(current_price)
                    if result:
                        outcome, closed = result
                        is_win  = outcome == "TP_HIT"
                        emoji   = "🎯" if is_win else "🛑"
                        pnl_txt = f"+{closed['pnl_pts']:.2f}" if closed['pnl_pts'] > 0 else f"{closed['pnl_pts']:.2f}"
                        stats   = tracker.stats()
                        _telegram(
                            f"{emoji} <b>{'TARGET HIT' if is_win else 'STOP LOSS HIT'} — XAUUSD</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"Trade: <b>{closed['action']}</b> opened @ <b>${closed['entry']:.2f}</b>\n"
                            f"Closed @ <b>${closed['close_px']:.2f}</b>\n"
                            f"P&L: <b>{pnl_txt} pts</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 Session stats: {stats['wins']}W / {stats['losses']}L  "
                            f"({stats['win_rate']}% WR) | Total P&L: {stats['total_pnl']:+.1f} pts\n"
                            f"⏰ {datetime.now(timezone.utc).strftime('%d %b %Y  %H:%M UTC')}"
                        )

                # ── Print card ──────────────────────────────────────────────
                print()
                print(DIVIDER)
                print(signal.card())
                print(DIVIDER)
                print()

                # ── Signal logic with cooldown ───────────────────────────────
                should_alert = False
                cooldown_elapsed = now_ts - last_signal_ts

                if signal.is_trade():
                    entry_moved = abs(signal.entry - last_entry) > (signal.atr * 0.5)
                    direction_changed = signal.action != last_action

                    if direction_changed:
                        # Always alert on direction change
                        should_alert = True
                    elif cooldown_elapsed >= SIGNAL_COOLDOWN and entry_moved:
                        # Same direction but price moved significantly after 15min
                        should_alert = True
                    else:
                        mins_left = int((SIGNAL_COOLDOWN - cooldown_elapsed) / 60)
                        if cooldown_elapsed < SIGNAL_COOLDOWN:
                            logger.info(
                                f"Signal {signal.action} @ {signal.entry:.2f} suppressed "
                                f"(cooldown {mins_left}min left, entry moved {abs(signal.entry-last_entry):.2f}pts)"
                            )

                    if should_alert:
                        logger.info(f"*** NEW SIGNAL: {signal.action} @ ${signal.entry:.2f} score={signal.score} ***")
                        _telegram(signal.telegram_html())
                        last_signal_ts = now_ts
                        last_entry     = signal.entry
                        # Open trade in tracker
                        if not tracker.active:
                            tracker.open_trade(signal)

                elif last_action in ("BUY", "SELL"):
                    # Signal cleared
                    _telegram(
                        f"⏳ <b>Signal cleared — XAUUSD</b>\n"
                        f"Was: {last_action} | Now: WAIT\n"
                        f"Price: ${current_price:.2f}\n"
                        f"<i>{signal.reason[:150]}</i>\n"
                        f"⏰ {datetime.now(timezone.utc).strftime('%d %b %Y  %H:%M UTC')}"
                    )
                    # If trade still open with no SL/TP hit → manual close
                    if tracker.active:
                        pnl = (current_price - tracker.active["entry"]
                               if tracker.active["action"] == "BUY"
                               else tracker.active["entry"] - current_price)
                        tracker.active.update({
                            "close_ts": datetime.now(timezone.utc).isoformat(),
                            "outcome":  "MANUAL",
                            "close_px": round(current_price, 2),
                            "pnl_pts":  round(pnl, 2),
                        })
                        tracker._log.append(tracker.active)
                        tracker._save_log()
                        tracker.active = None

                last_action = signal.action

                # ── Save signal JSON ─────────────────────────────────────────
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
                    "active_trade": tracker.active,
                    "pnl_stats":   tracker.stats(),
                }
                with open(SIG_PATH, "w") as f:
                    json.dump(sig_dict, f)

                # ── Append to history (keep last 200) ────────────────────────
                try:
                    history = json.loads(open(HIST_PATH).read()) if Path(HIST_PATH).exists() else []
                except Exception:
                    history = []
                history.append({k: v for k, v in sig_dict.items() if k not in ("active_trade","pnl_stats")})
                history = history[-200:]
                with open(HIST_PATH, "w") as f:
                    json.dump(history, f)

                logger.info(
                    f"Price=${current_price:.2f} | {signal.action} score={signal.score} "
                    f"| Trend={signal.trend_1h} | Session={signal.session} "
                    f"| ActiveTrade={'YES' if tracker.active else 'NO'}"
                )

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
