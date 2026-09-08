"""
XAUUSD Scalp Backtest
=====================
Walk-forward simulation on historical 1m bars.
Uses analyze_scalp() on a rolling window — no lookahead bias.

Outputs: data/scalp_backtest_results.json

Run:
    python3 backtest/scalp_backtest.py
    python3 backtest/scalp_backtest.py --days 3
"""

from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
import pandas as pd
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
SL_PTS   = 4.0
TP1_PTS  = 5.0
TP2_PTS  = 10.0
ACTIVATION_PTS = 1.5
MIN_BARS = 60   # need at least 60 bars of history for indicators

OUT_FILE = ROOT / "data" / "scalp_backtest_results.json"


# ── Simulate one trade ──────────────────────────────────────────────────────────
def _simulate_trade(direction: str, entry: float, future_bars: pd.DataFrame) -> dict:
    """
    Walk forward on remaining bars after signal bar.
    Returns trade result dict.
    """
    sl  = entry - SL_PTS  if direction == "BUY" else entry + SL_PTS
    tp1 = entry + TP1_PTS if direction == "BUY" else entry - TP1_PTS
    tp2 = entry + TP2_PTS if direction == "BUY" else entry - TP2_PTS

    activated = False
    tp1_hit   = False
    sl_be     = sl  # after TP1, SL moves to entry

    for _, bar in future_bars.iterrows():
        hi = bar["high"]
        lo = bar["low"]

        # Activation check
        if not activated:
            if (direction == "BUY"  and hi >= entry + ACTIVATION_PTS) or \
               (direction == "SELL" and lo <= entry - ACTIVATION_PTS):
                activated = True
            else:
                # SL before activation = CANCELLED
                if (direction == "BUY"  and lo <= sl) or \
                   (direction == "SELL" and hi >= sl):
                    return {"outcome": "CANCELLED", "pnl": 0.0, "sl": sl, "tp1": tp1, "tp2": tp2}
            continue

        # After activation: check SL first (conservative)
        if not tp1_hit:
            if (direction == "BUY"  and lo <= sl_be) or \
               (direction == "SELL" and hi >= sl_be):
                return {"outcome": "SL", "pnl": round(sl_be - entry if direction == "BUY" else entry - sl_be, 2),
                        "sl": sl_be, "tp1": tp1, "tp2": tp2}
            # TP1
            if (direction == "BUY"  and hi >= tp1) or \
               (direction == "SELL" and lo <= tp1):
                tp1_hit = True
                sl_be   = entry   # move SL to breakeven
                pnl_tp1 = abs(tp1 - entry)
        else:
            # After TP1: check breakeven SL
            if (direction == "BUY"  and lo <= sl_be) or \
               (direction == "SELL" and hi >= sl_be):
                return {"outcome": "BE", "pnl": 0.0, "sl": sl_be, "tp1": tp1, "tp2": tp2}
            # TP2
            if (direction == "BUY"  and hi >= tp2) or \
               (direction == "SELL" and lo <= tp2):
                return {"outcome": "TP2", "pnl": round(abs(tp2 - entry), 2),
                        "sl": sl_be, "tp1": tp1, "tp2": tp2}

    # Still open at end of data
    return {"outcome": "OPEN", "pnl": 0.0, "sl": sl_be, "tp1": tp1, "tp2": tp2}


# ── Main backtest ───────────────────────────────────────────────────────────────
def run_backtest(days: int = 5) -> dict:
    from xauusd.data          import get_bars
    from xauusd.scalp_strategy import analyze_scalp

    logger.info(f"[ScalpBT] Fetching {days}d of 1m bars...")
    df_1m = get_bars("1m", f"{days}d")

    if df_1m is None or df_1m.empty:
        logger.error("[ScalpBT] No 1m data returned")
        return {}

    logger.info(f"[ScalpBT] Got {len(df_1m)} bars — running walk-forward simulation...")

    # Also fetch 5m bars for context
    df_5m = get_bars("5m", f"{days + 2}d")

    trades        = []
    equity        = 0.0
    equity_curve  = []
    last_signal   = None
    cooldown_bar  = -999
    in_trade_until_bar = -1  # index after which we can open new trade

    for i in range(MIN_BARS, len(df_1m)):
        # Skip if inside an active trade window
        if i <= in_trade_until_bar:
            continue

        hist_1m = df_1m.iloc[:i]
        ts      = df_1m.index[i]

        # Only run during London (07-12 UTC) or NY (13-17 UTC) sessions
        if hasattr(ts, "hour"):
            h = ts.hour
            if not ((7 <= h < 12) or (13 <= h < 17)):
                continue

        # Skip weekends
        if hasattr(ts, "weekday") and ts.weekday() >= 5:
            continue

        # Get 5m context bars up to this point
        if df_5m is not None and not df_5m.empty:
            hist_5m = df_5m[df_5m.index <= ts]
        else:
            hist_5m = pd.DataFrame()

        try:
            sig = analyze_scalp(hist_1m, hist_5m)
        except Exception as e:
            logger.debug(f"[ScalpBT] analyze_scalp error at bar {i}: {e}")
            continue

        if not sig.is_trade():
            continue

        # Cooldown: skip if same direction within 10 bars
        if last_signal == sig.action and (i - cooldown_bar) < 10:
            continue

        entry = float(df_1m["close"].iloc[i])

        # Simulate trade on future bars (max 60 bars = 1 hour)
        future = df_1m.iloc[i+1 : i+61]
        if future.empty:
            continue

        result = _simulate_trade(sig.action, entry, future)
        outcome = result["outcome"]

        if outcome in ("CANCELLED", "OPEN"):
            continue

        pnl = result["pnl"] if sig.action == "BUY" else -result.get("pnl", 0)
        # For SL on SELL, pnl is already computed correctly in _simulate_trade
        pnl = result["pnl"]
        if outcome == "SL":
            pnl = -SL_PTS
        elif outcome == "BE":
            pnl = 0.0

        equity += pnl
        equity_curve.append(round(equity, 2))

        trades.append({
            "ts":          str(ts),
            "direction":   sig.action,
            "signal_type": sig.signal_type,
            "entry":       round(entry, 2),
            "sl":          round(result["sl"], 2),
            "tp1":         round(result["tp1"], 2),
            "tp2":         round(result["tp2"], 2),
            "outcome":     outcome,
            "pnl_pts":     round(pnl, 2),
            "score":       sig.score,
            "session":     sig.session,
        })

        last_signal    = sig.action
        cooldown_bar   = i
        # Advance past simulated trade window
        in_trade_until_bar = i + len(future)

    # ── Stats ─────────────────────────────────────────────────────────────────
    total  = len(trades)
    wins   = [t for t in trades if t["outcome"] == "TP2"]
    losses = [t for t in trades if t["outcome"] == "SL"]
    bes    = [t for t in trades if t["outcome"] == "BE"]
    pnls   = [t["pnl_pts"] for t in trades]

    avg_win  = round(sum(t["pnl_pts"] for t in wins)  / len(wins),  2) if wins  else 0
    avg_loss = round(sum(t["pnl_pts"] for t in losses) / len(losses), 2) if losses else 0
    wr       = round(len(wins) / total * 100, 1) if total else 0

    results = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "days_backtested": days,
        "total_bars":    len(df_1m),
        "trades":        total,
        "wins":          len(wins),
        "losses":        len(losses),
        "breakevens":    len(bes),
        "wr":            wr,
        "total_pnl":     round(sum(pnls), 2),
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "max_drawdown":  _max_drawdown(equity_curve),
        "equity_curve":  equity_curve,
        "trade_list":    trades,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    logger.info(
        f"[ScalpBT] Done: {total} trades | {wr}% WR | "
        f"PnL {results['total_pnl']:+.2f} pts | "
        f"Saved → {OUT_FILE}"
    )
    return results


def _max_drawdown(equity_curve: list) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5, help="Days of 1m history to backtest")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    r = run_backtest(days=args.days)
    if r:
        print(f"\nBacktest complete:")
        print(f"  Trades      : {r['trades']}")
        print(f"  Win rate    : {r['wr']}%")
        print(f"  Total P&L   : {r['total_pnl']:+.2f} pts")
        print(f"  Avg win     : +{r['avg_win']:.2f} pts")
        print(f"  Avg loss    : {r['avg_loss']:.2f} pts")
        print(f"  Max drawdown: {r['max_drawdown']:.2f} pts")
        print(f"  Output      : {OUT_FILE}")
