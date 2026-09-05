"""
Loop Engine — keeps everything running 24/7
============================================
- Every 60s  : poll tape, run ML agent, print signal card
- Every 30min: store market state in Qdrant vector memory
- Every 30min: record outcomes for states stored 30min ago (for training)
- EOD (15:35) : retrain XGBoost on today's real labeled outcomes
- Continuous  : dashboard + ngrok stay alive, auto-restart if they crash

Run:
    python3 loop_engine.py              # NIFTY
    python3 loop_engine.py BANKNIFTY
"""

import os
import sys
import time
import signal
import subprocess
from datetime import datetime, time as dtime, timedelta
from collections import deque
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

SYMBOL       = sys.argv[1].upper() if len(sys.argv) > 1 else "NIFTY"
POLL_SECS    = 60        # agent decision every 60s
MEMORY_SECS  = 1800      # store vector state every 30min
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
EOD_RETRAIN  = dtime(15, 35)  # retrain after market close

DIVIDER = "=" * 65


# ── Process watchers ──────────────────────────────────────────────────────────

_procs: dict = {}

def _port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _ensure_running(name: str, cmd: list, log_file: str, check_port: int = 0):
    """Start a subprocess if not alive (skips if port already responding)."""
    # If a port check is given and something is already listening → skip
    if check_port and _port_in_use(check_port):
        return
    proc = _procs.get(name)
    if proc and proc.poll() is None:
        return  # already running
    logger.info(f"Starting {name}...")
    f = open(log_file, "a")
    _procs[name] = subprocess.Popen(cmd, stdout=f, stderr=f)
    logger.info(f"{name} PID: {_procs[name].pid}")


def _keep_services():
    """Ensure dashboard + ngrok are always running."""
    _ensure_running(
        "dashboard",
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", "8502", "--server.headless", "true"],
        "/tmp/dashboard.log",
        check_port=8502,
    )
    _ensure_running(
        "ngrok",
        ["ngrok", "http", "8501", "--log=stdout"],
        "/tmp/ngrok.log",
        check_port=4040,
    )


# ── Market hours helpers ──────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:   # Saturday=5, Sunday=6 — NSE closed
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def is_eod() -> bool:
    now = datetime.now().time()
    return EOD_RETRAIN <= now <= dtime(16, 0)


# ── Outcome recorder ─────────────────────────────────────────────────────────

class OutcomeTracker:
    """Tracks stored vector IDs and records outcomes 30min later."""

    def __init__(self, tracker, vector_store):
        self._tracker = tracker
        self._vs      = vector_store
        self._pending = deque()  # (store_time, point_id, spot_at_store)

    def add(self, point_id: str, spot: float):
        if point_id:
            self._pending.append((datetime.now(), point_id, spot))

    def flush_ready(self):
        """Record outcomes for states that are now 30+ min old."""
        now = datetime.now()
        while self._pending:
            store_time, point_id, old_spot = self._pending[0]
            if (now - store_time).total_seconds() < MEMORY_SECS:
                break
            self._pending.popleft()
            try:
                current_spot = self._tracker.get_market_summary().get("spot", old_spot)
                move_pct = (current_spot - old_spot) / old_spot if old_spot else 0
                direction = "UP" if move_pct > 0.001 else "DOWN" if move_pct < -0.001 else "FLAT"
                self._vs.record_outcome(point_id, direction, move_pct)
                logger.info(f"Outcome recorded: {point_id[:8]}... {direction} {move_pct:+.2%}")
            except Exception as e:
                logger.warning(f"Outcome record failed: {e}")


# ── Signal formatting ─────────────────────────────────────────────────────────

def _print_signal(decision):
    print(DIVIDER)
    print(decision.signal_message())
    print(DIVIDER)
    print()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    from core.order_flow.oi_tracker import OITracker
    from core.agent.ml_agent import OpenSourceTradingAgent, get_ml_agent
    from core.memory.vector_store import get_vector_store

    print(DIVIDER)
    print(f"  AI-Trade Loop Engine | {SYMBOL}")
    print(f"  Market: {MARKET_OPEN.strftime('%H:%M')} – {MARKET_CLOSE.strftime('%H:%M')}")
    print(f"  EOD retrain at: {EOD_RETRAIN.strftime('%H:%M')}")
    print(DIVIDER)

    # Init components
    tracker      = OITracker(SYMBOL)
    agent        = get_ml_agent(SYMBOL)
    vector_store = get_vector_store()
    outcome_tracker = OutcomeTracker(tracker, vector_store)

    last_poll       = 0.0
    last_mem_store  = 0.0
    retrained_today = False
    last_signal_action = None

    # Keep services alive
    _keep_services()
    logger.info("Dashboard + ngrok running. Access: http://localhost:4040 for ngrok URL")

    def _shutdown(sig, frame):
        print("\n  Shutting down loop engine...")
        for name, proc in _procs.items():
            proc.terminate()
            logger.info(f"Stopped {name}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        now     = datetime.now()
        now_ts  = time.time()

        # Keep services alive
        _keep_services()

        # ── EOD Retrain (historical backtest + live outcomes) ──────────
        if is_eod() and not retrained_today:
            logger.info("EOD retrain: running historical backtest trainer...")
            try:
                import importlib, backtest_trainer
                importlib.reload(backtest_trainer)
                backtest_trainer.SYMBOL = SYMBOL
                backtest_trainer.main()
                retrained_today = True
                agent._xgb_model._load()
                logger.info("EOD retrain complete — model updated with today's data.")
                # Send Telegram EOD summary
                try:
                    from core.alerts.telegram_bot import send_eod_summary
                    send_eod_summary(SYMBOL, agent._xgb_model.info, vector_store.get_stats())
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"EOD retrain failed: {e}")
                # Fallback: retrain on synthetic only
                try:
                    agent.train(force=True)
                    retrained_today = True
                except Exception as e2:
                    logger.error(f"Fallback retrain failed: {e2}")

        # Reset retrained flag at midnight
        if now.hour == 0 and now.minute < 2:
            retrained_today = False

        # ── Market hours only ─────────────────────────────────────────
        if not is_market_open():
            mins_to_open = None
            if now.time() < MARKET_OPEN:
                opens_at = datetime.combine(now.date(), MARKET_OPEN)
                secs = int((opens_at - now).total_seconds())
                print(f"\r  Market opens in {secs // 3600:02d}h {(secs % 3600) // 60:02d}m {secs % 60:02d}s   ",
                      end="", flush=True)
            time.sleep(10)
            continue

        print()  # clear the countdown line

        # ── Flush outcomes ────────────────────────────────────────────
        outcome_tracker.flush_ready()

        # ── Poll tape ─────────────────────────────────────────────────
        if now_ts - last_poll >= POLL_SECS:
            last_poll = now_ts
            try:
                tracker.tick_tape()
                decision = agent.decide(tracker, symbol=SYMBOL)

                # Compute Greeks for actionable signals
                greeks = {}
                if decision.is_trade():
                    try:
                        from core.options.greeks import greeks_for_decision, greeks_summary
                        greeks = greeks_for_decision(decision)
                        logger.info(f"Greeks: {greeks_summary(greeks)}")
                    except Exception as ge:
                        logger.debug(f"Greeks error: {ge}")

                # SHAP explanation
                if decision.is_trade():
                    try:
                        from core.agent.explainer import get_explainer
                        exp = get_explainer(agent._xgb_model)
                        features = tracker.get_model_features()
                        expl = exp.explain(features, decision.action, top_n=5)
                        logger.info(f"SHAP: {expl['summary_text']}")
                    except Exception as se:
                        logger.debug(f"SHAP error: {se}")

                _print_signal(decision)

                # Save Nifty signal to JSON for XAUUSD dashboard
                try:
                    import json as _json
                    _summary  = tracker.get_market_summary()
                    _features = tracker.get_model_features()
                    _sig_data = {
                        "action":     decision.action,
                        "strike":     getattr(decision, "strike", 0),
                        "entry":      getattr(decision, "entry", 0),
                        "stop_loss":  getattr(decision, "stop_loss", 0),
                        "target":     getattr(decision, "target", 0),
                        "confidence": float(getattr(decision, "confidence", 0)),
                        "spot":       _summary.get("spot", 0),
                        "pcr":        _summary.get("pcr", 0),
                        "expiry":     _summary.get("expiry", ""),
                        "vix":        _features.get("f_vix", 0),
                        "tape_bias":  _features.get("f_tape_bias", 0),
                        "symbol":     SYMBOL,
                        "timestamp":  datetime.now().isoformat(),
                    }
                    with open("/tmp/nifty_signal.json", "w") as _f:
                        _json.dump(_sig_data, _f)
                except Exception as _je:
                    logger.debug(f"Nifty JSON save error: {_je}")

                # Telegram alert on new actionable signal
                if decision.action != last_signal_action and decision.is_trade():
                    logger.info(f"*** NEW SIGNAL: {decision.action} @ strike {decision.strike} ***")
                    try:
                        from core.alerts.telegram_bot import send_signal
                        from core.data.nse_participant import get_participant_data
                        fii_pic = get_participant_data().get_full_picture()
                        send_signal(decision, greeks=greeks, fii_summary=fii_pic)
                    except Exception as te:
                        logger.debug(f"Telegram error: {te}")
                last_signal_action = decision.action

            except Exception as e:
                logger.error(f"Poll error: {e}")

        # ── Store vector memory every 30min ───────────────────────────
        if now_ts - last_mem_store >= MEMORY_SECS:
            last_mem_store = now_ts
            try:
                features     = tracker.get_model_features()
                tape_summary = tracker.tape_reader.get_flow_summary()
                from core.data.nse_participant import get_participant_data
                pic = get_participant_data().get_full_picture()

                point_id = vector_store.store_state(
                    features    = features,
                    symbol      = SYMBOL,
                    tape_bias   = tape_summary.get("bias", "NEUTRAL"),
                    fii_bias    = pic.get("fno", {}).get("fii_bias", "NEUTRAL"),
                    smart_money = pic.get("smart_money_bias", "NEUTRAL"),
                )
                spot = features.get("f_spot", 0)
                outcome_tracker.add(point_id, spot)
                stats = vector_store.get_stats()
                logger.info(f"Vector state stored. Total in memory: {stats.get('total_states', 0)}")
            except Exception as e:
                logger.error(f"Memory store error: {e}")

        time.sleep(5)


if __name__ == "__main__":
    run()
