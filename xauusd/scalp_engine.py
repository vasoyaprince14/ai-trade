"""
XAUUSD Scalping Engine
=======================
Runs every 30 seconds during London + NY sessions.
Uses 1m bars + tape reading for fast entries.

Sends Telegram on:
  - New scalp signal (entry)
  - TP1 hit (move SL to breakeven)
  - TP2 hit (full exit)
  - SL hit (loss)
  - Trade cancelled (signal reversed)

Run:
    python3 xauusd/scalp_engine.py
    python3 xauusd/scalp_engine.py --once    # single scan
    python3 xauusd/scalp_engine.py --debug   # verbose logs
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

# ── Config ──────────────────────────────────────────────────────────────────
POLL_SECS       = 30          # scan every 30 seconds
SIGNAL_COOLDOWN = 300         # 5 min between same-direction scalp alerts
ACTIVATION_PTS  = 1.5         # $1.5 move in direction to confirm entry
SIG_FILE        = "/tmp/xauusd_scalp_signal.json"
HIST_FILE       = ROOT / "data" / "xauusd_scalp_trades.json"
MAX_HIST        = 500

_last_action    = "WAIT"
_last_signal_ts = 0.0
_history: list  = []
_active_scalp   = None   # {direction, entry, sl, tp1, tp2, tp1_hit, activated, open_ts}


# ── Telegram ─────────────────────────────────────────────────────────────────
def _tg(msg: str):
    try:
        from core.alerts.telegram_bot import send_message
        send_message(msg, parse_mode="HTML")
    except Exception as e:
        logger.debug(f"[Scalp] Telegram: {e}")


# ── History ───────────────────────────────────────────────────────────────────
def _load_hist() -> list:
    try:
        if HIST_FILE.exists():
            return json.loads(HIST_FILE.read_text())
    except Exception:
        pass
    return []


def _save_hist():
    try:
        HIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        HIST_FILE.write_text(json.dumps(_history[-MAX_HIST:], indent=2, default=str))
    except Exception as e:
        logger.warning(f"[Scalp] Hist save: {e}")


# ── Active trade tracker ──────────────────────────────────────────────────────
def _check_trade(price: float) -> str | None:
    """
    Check active scalp against live price.
    Returns outcome string or None if still open.
    """
    global _active_scalp
    if not _active_scalp:
        return None

    t   = _active_scalp
    dir = t["direction"]
    sl  = t["sl"]
    tp1 = t["tp1"]
    tp2 = t["tp2"]

    # Activation check (price must move 1.5 pts in trade direction first)
    if not t.get("activated"):
        if (dir == "BUY"  and price >= t["entry"] + ACTIVATION_PTS) or \
           (dir == "SELL" and price <= t["entry"] - ACTIVATION_PTS):
            t["activated"] = True
            emoji = "🟢 BUY" if dir == "BUY" else "🔴 SELL"
            _tg(
                f"⚡ <b>SCALP ACTIVATED — {emoji}</b>\n"
                f"📍 Price: <b>${price:.2f}</b> | Entry: ${t['entry']:.2f}\n"
                f"🛑 SL: ${sl:.2f} | ✅ TP1: ${tp1:.2f} | TP2: ${tp2:.2f}\n"
                f"⏱ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
            )
            logger.info(f"[Scalp] Trade activated @ ${price:.2f}")

    if not t.get("activated"):
        # Before activation: cancel if price hits SL
        sl_hit = (dir == "BUY" and price <= sl) or (dir == "SELL" and price >= sl)
        if sl_hit:
            _tg(f"❌ <b>SCALP CANCELLED — SL before activation</b>\n"
                f"Price ${price:.2f} hit SL ${sl:.2f} before entry confirmed.\n"
                f"⏱ {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
            _active_scalp = None
            return "CANCELLED"
        return None

    # TP1
    if not t.get("tp1_hit"):
        tp1_hit = (dir == "BUY" and price >= tp1) or (dir == "SELL" and price <= tp1)
        if tp1_hit:
            t["tp1_hit"] = True
            # Move SL to breakeven
            t["sl"] = t["entry"]
            pnl = abs(tp1 - t["entry"])
            _tg(
                f"✅ <b>SCALP TP1 HIT — {dir}</b>\n"
                f"Price: <b>${price:.2f}</b> | P&L so far: <b>+${pnl:.2f}</b>\n"
                f"🔒 SL moved to breakeven ${t['entry']:.2f}\n"
                f"🎯 TP2 target: ${tp2:.2f}\n"
                f"⏱ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
            )
            logger.info(f"[Scalp] TP1 hit @ ${price:.2f}, SL → breakeven")
            return None  # trade still open

    # SL (or breakeven after TP1)
    sl_hit = (dir == "BUY" and price <= t["sl"]) or (dir == "SELL" and price >= t["sl"])
    if sl_hit:
        pnl  = price - t["entry"] if dir == "BUY" else t["entry"] - price
        be   = t.get("tp1_hit", False)
        emoji = "🔒" if be else "🛑"
        label = "BREAKEVEN" if be else "STOP LOSS"
        _tg(
            f"{emoji} <b>SCALP {label} — {dir}</b>\n"
            f"Closed @ <b>${price:.2f}</b> | Entry: ${t['entry']:.2f}\n"
            f"P&L: <b>{'±$0' if be else f'${pnl:.2f}'}</b>\n"
            f"⏱ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        _record_outcome(t, price, "SL" if not be else "BE")
        _active_scalp = None
        return "SL_HIT"

    # TP2
    tp2_hit = (dir == "BUY" and price >= tp2) or (dir == "SELL" and price <= tp2)
    if tp2_hit:
        pnl = abs(tp2 - t["entry"])
        _tg(
            f"🎯 <b>SCALP TP2 HIT — {dir}</b>\n"
            f"Price: <b>${price:.2f}</b> | Full target reached!\n"
            f"P&L: <b>+${pnl:.2f} pts</b>  🎉\n"
            f"⏱ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        _record_outcome(t, price, "TP2")
        _active_scalp = None
        return "TP2_HIT"

    return None


def _record_outcome(trade: dict, close_px: float, outcome: str):
    global _history
    entry = trade["entry"]
    dir   = trade["direction"]
    pnl   = (close_px - entry) if dir == "BUY" else (entry - close_px)
    _history.append({
        "open_ts":  trade.get("open_ts", ""),
        "close_ts": datetime.now(timezone.utc).isoformat(),
        "direction": dir,
        "entry":    entry,
        "close_px": round(close_px, 2),
        "outcome":  outcome,
        "pnl_pts":  round(pnl, 2),
        "signal_type": trade.get("signal_type", ""),
        "session":  trade.get("session", ""),
    })
    _save_hist()


def _pnl_stats() -> dict:
    if not _history:
        return {}
    wins   = [h for h in _history if h.get("outcome") in ("TP1", "TP2")]
    losses = [h for h in _history if h.get("outcome") == "SL"]
    pnls   = [h.get("pnl_pts", 0) for h in _history]
    total  = len(_history)
    return {
        "trades": total,
        "wins":   len(wins),
        "losses": len(losses),
        "wr":     round(len(wins) / total * 100, 1) if total else 0,
        "total_pnl": round(sum(pnls), 2),
    }


# ── Gold market hours ─────────────────────────────────────────────────────────
def _gold_market_open() -> bool:
    now = datetime.now(timezone.utc)
    wd  = now.weekday()
    hr  = now.hour
    if wd == 5: return False           # Saturday
    if wd == 6 and hr < 22: return False  # Sunday before 22:00 UTC
    if hr == 21: return False          # daily maintenance
    return True


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(once: bool = False):
    global _last_action, _last_signal_ts, _history, _active_scalp
    _history = _load_hist()

    from xauusd.data           import get_bars, get_price
    from xauusd.scalp_strategy import analyze_scalp

    print("=" * 58)
    print("  ⚡ XAUUSD Scalping Engine  v1")
    print("  Tape: CVD + Absorption + Climax + VWAP + EMA8/21")
    print("  Timeframe: 1m entry | 5m context | 30s polling")
    print(f"  SL: ${4:.0f} | TP1: ${5:.0f} | TP2: ${10:.0f} | Cooldown: {SIGNAL_COOLDOWN//60}min")
    print("  Sessions: London 07-12 UTC | NY 13-17 UTC")
    print("=" * 58)

    while True:
        if not _gold_market_open() and not once:
            now_utc = datetime.now(timezone.utc)
            print(f"\r  Gold market closed — {now_utc.strftime('%a %H:%M UTC')}   ",
                  end="", flush=True)
            time.sleep(30)
            continue

        try:
            # Fetch data
            df_1m = get_bars("1m", "1d")
            df_5m = get_bars("5m", "5d")
            live_px = get_price()

            if df_1m.empty:
                logger.warning("[Scalp] No 1m data")
                time.sleep(POLL_SECS)
                continue

            # Use live price as latest close
            if live_px > 0 and not df_1m.empty:
                df_1m.loc[df_1m.index[-1], "close"] = live_px
                df_1m.loc[df_1m.index[-1], "high"]  = max(df_1m["high"].iloc[-1], live_px)
                df_1m.loc[df_1m.index[-1], "low"]   = min(df_1m["low"].iloc[-1],  live_px)

            # Check active trade
            current_price = live_px if live_px > 0 else float(df_1m["close"].iloc[-1])
            _check_trade(current_price)

            # Run scalp analysis
            sig = analyze_scalp(df_1m, df_5m)

            # Log every tick
            tape_info = f"Tape={sig.tape_bias} Buy={sig.buy_pressure:.0f}% Δ5m={sig.delta_5m:+.0f}"
            if not sig.is_trade():
                logger.info(
                    f"[Scalp] WAIT score={sig.score} | {sig.signal_type} | "
                    f"{tape_info} | VWAPdev={sig.vwap_dev:+.2f}σ | ${current_price:.2f}"
                )
            else:
                logger.info(
                    f"[Scalp] *** {sig.action} score={sig.score} [{sig.signal_type}] "
                    f"entry=${sig.entry:.2f} SL=${sig.sl:.2f} TP2=${sig.tp2:.2f} ***"
                )

            # Cooldown logic
            now_ts  = time.time()
            elapsed = now_ts - _last_signal_ts
            should_alert = (
                sig.is_trade() and
                _active_scalp is None and      # no open trade
                (sig.action != _last_action or elapsed >= SIGNAL_COOLDOWN)
            )

            if should_alert:
                _tg(sig.telegram_html())
                _last_action    = sig.action
                _last_signal_ts = now_ts
                _active_scalp   = {
                    "direction":   sig.action,
                    "entry":       sig.entry,
                    "sl":          sig.sl,
                    "tp1":         sig.tp1,
                    "tp2":         sig.tp2,
                    "tp1_hit":     False,
                    "activated":   False,
                    "signal_type": sig.signal_type,
                    "session":     sig.session,
                    "open_ts":     sig.timestamp.isoformat(),
                }
                stats = _pnl_stats()
                if stats.get("trades", 0) > 0:
                    logger.info(
                        f"[Scalp] Stats: {stats['trades']} trades | "
                        f"{stats['wr']}% WR | PnL {stats['total_pnl']:+.2f} pts"
                    )

            # Cancel active trade if signal reverses
            elif _active_scalp and sig.is_trade() and sig.action != _active_scalp["direction"]:
                t = _active_scalp
                pnl = (current_price - t["entry"]) if t["direction"] == "BUY" \
                       else (t["entry"] - current_price)
                _tg(
                    f"↩️ <b>SCALP CANCELLED — Signal reversed</b>\n"
                    f"Was: {t['direction']} | Now: {sig.action}\n"
                    f"Closed @ ${current_price:.2f} | P&L: {pnl:+.2f} pts\n"
                    f"⏱ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
                )
                _record_outcome(t, current_price, "CANCELLED")
                _active_scalp  = None
                _last_action   = "WAIT"

            # Save signal JSON for dashboard
            try:
                sig_dict = {
                    "action":       sig.action,
                    "signal_type":  sig.signal_type,
                    "entry":        sig.entry,
                    "sl":           sig.sl,
                    "tp1":          sig.tp1,
                    "tp2":          sig.tp2,
                    "tape_bias":    sig.tape_bias,
                    "buy_pressure": sig.buy_pressure,
                    "delta_5m":     sig.delta_5m,
                    "vwap":         sig.vwap,
                    "vwap_dev":     sig.vwap_dev,
                    "score":        sig.score,
                    "session":      sig.session,
                    "reasons":      sig.reasons,
                    "active_trade": _active_scalp,
                    "pnl_stats":    _pnl_stats(),
                    "timestamp":    sig.timestamp.isoformat(),
                }
                with open(SIG_FILE, "w") as f:
                    json.dump(sig_dict, f)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[Scalp] Loop error: {e}", exc_info=True)

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
