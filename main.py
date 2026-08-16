"""
AI-Trade - Main Orchestrator
=============================
Wires together all components:
  - NSE Data Scraper
  - Order Flow Tracker
  - Multiple Strategies (intraday, swing, positional, F&O)
  - Broker (paper or live)
  - Risk Manager
  - Database
  - Scheduler
  - Dashboard (via separate process)

Run modes:
  python main.py live        → Start live/paper trading loop
  python main.py backtest    → Run backtest on all strategies
  python main.py dashboard   → Launch Streamlit dashboard only
  python main.py scan        → One-shot market scan
"""
import sys
import time
import argparse
import threading
from datetime import datetime
from pathlib import Path

from loguru import logger

# ── Setup paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
# Our root MUST be first so config/ resolves to ours, not vendors/ai-trader/config/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ai-trader is appended (not inserted at 0) so it only fills gaps
_ai_trader_path = str(ROOT / "vendors" / "ai-trader")
if _ai_trader_path not in sys.path:
    sys.path.append(_ai_trader_path)

# ── Setup logging ─────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(
    LOG_DIR / "trading_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {name} | {message}",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from core.data.db import init_db, DBManager
from core.data.nse_scraper import get_scraper
from core.data.historical import fetch_historical, add_indicators, get_nifty_data
from core.order_flow.oi_tracker import OITracker
from core.brokers.paper import PaperBroker
from core.risk.manager import RiskManager
from core.backtest.engine import Backtester, BacktestResult
from config.settings import config


def get_broker():
    """Get broker based on config."""
    broker_name = config.get("brokers", {}).get("default", "paper")
    if broker_name == "zerodha":
        try:
            from core.brokers.zerodha import ZerodhaBroker
            return ZerodhaBroker()
        except Exception as e:
            logger.error(f"Zerodha connection failed: {e}. Falling back to paper.")
    logger.info("Using Paper Broker")
    return PaperBroker()


def get_strategies(broker, risk_mgr):
    """Initialize all enabled strategies."""
    strategies = []
    cfg = config.get("strategies", {})

    if cfg.get("scalp_momentum", {}).get("enabled", False):
        from core.strategies.intraday.scalp_momentum import ScalpMomentumStrategy
        strategies.append(ScalpMomentumStrategy(broker=broker))
        logger.info("Loaded: ScalpMomentum")

    if cfg.get("orb", {}).get("enabled", False):
        from core.strategies.intraday.orb import ORBStrategy
        strategies.append(ORBStrategy(broker=broker))
        logger.info("Loaded: ORB")

    if cfg.get("swing_trend", {}).get("enabled", False):
        from core.strategies.swing.trend_swing import SwingTrendStrategy
        strategies.append(SwingTrendStrategy(broker=broker))
        logger.info("Loaded: SwingTrend")

    if cfg.get("long_term", {}).get("enabled", False):
        from core.strategies.positional.long_term import LongTermStrategy
        strategies.append(LongTermStrategy(broker=broker))
        logger.info("Loaded: LongTerm")

    if cfg.get("order_flow_fno", {}).get("enabled", True):
        from core.strategies.fno.order_flow_fno import OrderFlowFNOStrategy
        strategies.append(OrderFlowFNOStrategy(broker=broker))
        logger.info("Loaded: OrderFlowFNO")

    return strategies


# ── Mode: Live / Paper Trading Loop ──────────────────────────────────────────

def run_live():
    """Main live trading loop."""
    logger.info("=" * 60)
    logger.info("  AI-TRADE | Starting Trading Loop")
    logger.info("=" * 60)

    # Init
    init_db()
    broker = get_broker()
    risk_mgr = RiskManager(broker)
    strategies = get_strategies(broker, risk_mgr)
    scraper = get_scraper()
    oi_tracker = OITracker("NIFTY")

    logger.info(f"Active strategies: {[s.name for s in strategies]}")
    logger.info(f"Broker: {type(broker).__name__}")
    logger.info(f"Capital: ₹{config['risk']['default_capital']:,}")

    indices = config.get("market", {}).get("indices", ["NIFTY"])
    timeframes = {
        "intraday": "5min",
        "swing": "1d",
        "positional": "1wk",
    }

    db = DBManager()
    scan_count = 0
    risk_mgr.reset_daily()

    while True:
        try:
            now = datetime.now()

            # Market hours check
            if not scraper.is_market_open():
                logger.info("Market closed. Waiting...")
                time.sleep(60)
                continue

            scan_count += 1
            logger.info(f"[Scan #{scan_count}] {now.strftime('%H:%M:%S')}")

            for symbol in indices:
                # Fetch intraday data
                data_5m = fetch_historical(symbol, "5min", days=2)
                if not data_5m.empty:
                    data_5m = add_indicators(data_5m)

                data_1d = fetch_historical(symbol, "1d", days=365)
                if not data_1d.empty:
                    data_1d = add_indicators(data_1d)

                # Run strategies
                for strategy in strategies:
                    try:
                        if strategy.name in ("scalp_momentum", "orb"):
                            signals = strategy.generate_signals(data_5m, symbol=symbol)
                        elif strategy.name == "swing_trend":
                            signals = strategy.generate_signals(data_1d, symbol=symbol)
                        elif strategy.name == "order_flow_fno":
                            signals = strategy.generate_signals(data_1d, symbol=symbol)
                        else:
                            signals = []

                        for sig in signals:
                            # Risk check
                            approval = risk_mgr.approve_trade(sig)
                            if approval["approved"]:
                                logger.info(f"SIGNAL → {sig.strategy} | {sig.symbol} {sig.direction} | {sig.notes}")
                                # Execute
                                results = strategy.execute_signals([sig])
                                if results:
                                    risk_mgr.on_trade_opened(sig.symbol)
                                # Save signal to DB
                                db.save_signal({
                                    "strategy": sig.strategy,
                                    "symbol": sig.symbol,
                                    "direction": sig.direction,
                                    "signal_type": sig.signal_type,
                                    "price": sig.price,
                                    "target": sig.target,
                                    "stop_loss": sig.stop_loss,
                                    "notes": sig.notes,
                                    "executed": len(results) > 0,
                                })
                            else:
                                logger.debug(f"Signal rejected: {approval['reason']}")

                    except Exception as e:
                        logger.error(f"Strategy {strategy.name} error: {e}")

            # Wait for next scan (60 seconds)
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Shutting down trading loop...")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(30)

    db.close()
    logger.info("Trading loop stopped.")


# ── Mode: Backtest ────────────────────────────────────────────────────────────

def run_backtest(strategy_name: str = "all", days: int = 365):
    """Run backtest for specified strategy."""
    logger.info(f"Starting backtest | Strategy: {strategy_name} | Days: {days}")
    init_db()

    results = {}

    strategies_to_test = []

    if strategy_name in ("all", "swing"):
        from core.strategies.swing.trend_swing import SwingTrendStrategy
        strategies_to_test.append(("NIFTY", "1d", SwingTrendStrategy()))

    if strategy_name in ("all", "long_term"):
        from core.strategies.positional.long_term import LongTermStrategy
        strategies_to_test.append(("NIFTY", "1wk", LongTermStrategy()))

    if strategy_name in ("all", "scalp"):
        from core.strategies.intraday.scalp_momentum import ScalpMomentumStrategy
        strategies_to_test.append(("NIFTY", "5min", ScalpMomentumStrategy()))

    for symbol, tf, strat in strategies_to_test:
        logger.info(f"Backtesting {strat.name} on {symbol} ({tf})...")
        bt = Backtester(
            strategy=strat,
            symbol=symbol,
            timeframe=tf,
            days=days,
            initial_capital=500000,
        )
        result = bt.run()
        results[strat.name] = result

        print(f"\n{'='*50}")
        print(f"  {strat.name.upper()} on {symbol} ({tf})")
        print(f"{'='*50}")
        for k, v in result.summary().items():
            print(f"  {k:20s}: {v}")

        # Export CSV
        out_path = ROOT / "data" / f"backtest_{strat.name}_{symbol}.csv"
        result.export_csv(str(out_path))

    return results


# ── Mode: Market Scan ─────────────────────────────────────────────────────────

def run_scan():
    """One-shot market scan - show current signals without trading."""
    logger.info("Running market scan...")

    sym_map = {
        "NIFTY": ("NIFTY 50", "^NSEI"),
        "BANKNIFTY": ("NIFTY BANK", "^NSEBANK"),
    }

    for symbol, (nse_sym, yf_sym) in sym_map.items():
        print(f"\n{'='*60}")
        print(f"  {symbol} Market State")
        print(f"{'='*60}")

        # Try yfinance for price (works globally)
        try:
            import yfinance as yf
            ticker = yf.Ticker(yf_sym)
            info = ticker.fast_info
            ltp = getattr(info, "last_price", 0) or 0
            prev = getattr(info, "previous_close", 0) or 0
            chg = ((ltp - prev) / prev * 100) if prev else 0
            print(f"  LTP      : ₹{ltp:,.2f}")
            print(f"  Change   : {chg:+.2f}%")
            print(f"  High     : ₹{getattr(info, 'day_high', 0):,.2f}")
            print(f"  Low      : ₹{getattr(info, 'day_low', 0):,.2f}")
        except Exception as e:
            print(f"  Price data: unavailable ({e})")

        # Try NSE live data (only works from Indian IP/VPN)
        scraper = get_scraper()
        pcr_data = scraper.get_pcr_data(symbol)
        if pcr_data and pcr_data.get("pcr_oi"):
            print(f"  PCR (OI) : {pcr_data['pcr_oi']:.3f}")
            print(f"  PCR (Vol): {pcr_data['pcr_vol']:.3f}")
            max_pain = scraper.get_max_pain(symbol)
            if max_pain:
                print(f"  Max Pain : {max_pain:.0f}")
            vix = scraper.get_vix()
            if vix:
                print(f"  India VIX: {vix:.2f}")
        else:
            print(f"  [Option chain requires Indian IP or Zerodha API]")

    # Strategy signals from historical data (works globally via yfinance)
    print(f"\n{'='*60}")
    print("  Strategy Signals (from daily chart)")
    print(f"{'='*60}")
    from core.strategies.swing.trend_swing import SwingTrendStrategy
    from core.strategies.positional.long_term import LongTermStrategy

    for symbol in ["NIFTY", "BANKNIFTY"]:
        data = fetch_historical(symbol, "1d", 365)
        if data.empty:
            continue
        data = add_indicators(data)
        latest = data.iloc[-1]

        print(f"\n  {symbol}:")
        print(f"    EMA21: {latest.get('ema21', 0):.1f}  EMA50: {latest.get('ema50', 0):.1f}")
        print(f"    RSI: {latest.get('rsi', 0):.1f}  ATR: {latest.get('atr', 0):.1f}")

        for strat in [SwingTrendStrategy(), LongTermStrategy()]:
            try:
                sigs = strat.generate_signals(data, symbol=symbol)
                for s in sigs:
                    arrow = "▲" if s.direction == "BUY" else "▼"
                    print(f"    {arrow} [{strat.name}] {s.direction} | {s.notes[:80]}")
            except Exception:
                pass

    print(f"\n{'='*60}")
    print("  Note: Run 'python3 main.py dashboard' for full UI")
    print(f"{'='*60}")


# ── Mode: Dashboard ───────────────────────────────────────────────────────────

def run_dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    dashboard_path = ROOT / "dashboard" / "app.py"
    port = config.get("dashboard", {}).get("port", 8501)
    logger.info(f"Launching dashboard at http://localhost:{port}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", "false",
    ])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI-Trade: Nifty Algo Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  live        Start live/paper trading loop
  backtest    Run backtest on strategies
  dashboard   Launch Streamlit dashboard
  scan        One-shot market scan (no trades)

Examples:
  python main.py scan
  python main.py backtest --strategy swing --days 500
  python main.py live
  python main.py dashboard
        """
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="dashboard",
        choices=["live", "backtest", "dashboard", "scan"],
        help="Run mode (default: dashboard)"
    )
    parser.add_argument("--strategy", default="all", help="Strategy to backtest")
    parser.add_argument("--days", type=int, default=365, help="Days of data for backtest")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════╗
║       AI-TRADE | Nifty Trading System    ║
║  Strategies: Intraday + Swing + F&O      ║
╚══════════════════════════════════════════╝
""")

    if args.mode == "live":
        run_live()
    elif args.mode == "backtest":
        run_backtest(args.strategy, args.days)
    elif args.mode == "scan":
        run_scan()
    elif args.mode == "dashboard":
        run_dashboard()


if __name__ == "__main__":
    main()
