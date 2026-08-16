"""
Backtesting Engine
===================
Integrates:
  vendors/ai-trader/backtest/ (BacktestResult, BacktestTrade, metrics)

Walk-forward backtesting on historical OHLCV data.
Supports all strategies (intraday, swing, positional, F&O).

Usage:
  from core.backtest.engine import Backtester
  bt = Backtester(strategy, data, capital=500000)
  result = bt.run()
  result.summary()
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Type
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from loguru import logger

from core.strategies.base import BaseStrategy, TradeSignal
from core.data.historical import fetch_historical, add_indicators

ROOT_DIR = Path(__file__).parent.parent.parent
VENDOR_DIR = ROOT_DIR / "vendors"
# Keep our root first in path so config/ etc resolve to ours
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: Optional[datetime] = None
    symbol: str = ""
    direction: str = ""
    strategy: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    quantity: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    result: str = ""   # WIN / LOSS / TIMEOUT
    notes: str = ""


@dataclass
class BacktestResult:
    trades: List[BacktestTrade] = field(default_factory=list)
    initial_capital: float = 500000
    final_capital: float = 500000

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0

    @property
    def gross_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0
        equity = [self.initial_capital]
        for t in self.trades:
            equity.append(equity[-1] + t.pnl)
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        if not self.trades:
            return 0
        returns = [t.pnl_pct for t in self.trades]
        if len(returns) < 2:
            return 0
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        return (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0

    @property
    def expectancy(self) -> float:
        if not self.trades:
            return 0
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if self.wins > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl <= 0]) if self.losses > 0 else 0
        return (self.win_rate * avg_win) + ((1 - self.win_rate) * avg_loss)

    def summary(self) -> Dict:
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": f"{self.win_rate:.1%}",
            "gross_pnl": f"₹{self.gross_pnl:,.2f}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "max_drawdown": f"{self.max_drawdown:.1%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "expectancy": f"₹{self.expectancy:,.2f}",
            "final_capital": f"₹{self.final_capital:,.2f}",
            "return_pct": f"{(self.final_capital / self.initial_capital - 1):.1%}",
        }

    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "result": t.result,
            }
            for t in self.trades
        ])

    def export_csv(self, path: str = "backtest_results.csv"):
        self.to_dataframe().to_csv(path, index=False)
        logger.info(f"Backtest results saved to {path}")


class Backtester:
    """
    Bar-by-bar backtesting engine.
    Simulates execution with slippage and brokerage.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame = None,
        symbol: str = "NIFTY",
        timeframe: str = "1d",
        days: int = 365,
        initial_capital: float = 500000,
        slippage_pct: float = 0.05,
        brokerage: float = 20.0,
    ):
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.slippage = slippage_pct / 100
        self.brokerage = brokerage

        if data is not None:
            self.data = data
        else:
            logger.info(f"Fetching {days}d of {timeframe} data for {symbol}...")
            raw = fetch_historical(symbol, timeframe, days)
            self.data = add_indicators(raw) if not raw.empty else raw

    def run(self) -> BacktestResult:
        """Run backtest bar by bar."""
        if self.data.empty:
            logger.error("No data for backtest")
            return BacktestResult(initial_capital=self.initial_capital)

        result = BacktestResult(initial_capital=self.initial_capital)
        open_trade: Optional[BacktestTrade] = None
        capital = self.initial_capital

        logger.info(f"Backtesting {self.strategy.name} on {self.symbol} ({len(self.data)} bars)")

        for i in range(20, len(self.data)):
            window = self.data.iloc[:i]
            latest = self.data.iloc[i].to_dict()
            close = latest.get("close", 0)
            high = latest.get("high", 0)
            low = latest.get("low", 0)
            ts = latest.get("timestamp", datetime.now())

            # Check exits for open trade
            if open_trade:
                exit_price = None
                exit_reason = None

                if open_trade.direction == "BUY":
                    if open_trade.stop_loss > 0 and low <= open_trade.stop_loss:
                        exit_price = open_trade.stop_loss * (1 - self.slippage)
                        exit_reason = "SL"
                    elif open_trade.target > 0 and high >= open_trade.target:
                        exit_price = open_trade.target * (1 - self.slippage)
                        exit_reason = "TARGET"
                else:  # SELL
                    if open_trade.stop_loss > 0 and high >= open_trade.stop_loss:
                        exit_price = open_trade.stop_loss * (1 + self.slippage)
                        exit_reason = "SL"
                    elif open_trade.target > 0 and low <= open_trade.target:
                        exit_price = open_trade.target * (1 + self.slippage)
                        exit_reason = "TARGET"

                if exit_price:
                    pnl_sign = 1 if open_trade.direction == "BUY" else -1
                    pnl = pnl_sign * (exit_price - open_trade.entry_price) * open_trade.quantity - self.brokerage
                    capital += pnl
                    open_trade.exit_price = exit_price
                    open_trade.exit_time = ts
                    open_trade.pnl = pnl
                    open_trade.pnl_pct = pnl / (open_trade.entry_price * open_trade.quantity)
                    open_trade.result = "WIN" if pnl > 0 else "LOSS"
                    result.trades.append(open_trade)
                    open_trade = None
                    continue

            # Generate signals
            if not open_trade:
                signals = self.strategy.generate_signals(window, symbol=self.symbol)
                for sig in signals:
                    if sig.signal_type == "ENTRY" and sig.direction in ("BUY", "SELL"):
                        fill_price = close * (1 + self.slippage if sig.direction == "BUY" else 1 - self.slippage)
                        qty = max(int(capital * 0.05 / fill_price), 1)
                        capital -= self.brokerage

                        open_trade = BacktestTrade(
                            entry_time=ts,
                            symbol=sig.symbol,
                            direction=sig.direction,
                            strategy=self.strategy.name,
                            entry_price=fill_price,
                            stop_loss=sig.stop_loss,
                            target=sig.target,
                            quantity=qty,
                            notes=sig.notes,
                        )
                        break

        # Close any open trade at end
        if open_trade:
            last_close = self.data.iloc[-1].get("close", open_trade.entry_price)
            pnl_sign = 1 if open_trade.direction == "BUY" else -1
            pnl = pnl_sign * (last_close - open_trade.entry_price) * open_trade.quantity - self.brokerage
            capital += pnl
            open_trade.exit_price = last_close
            open_trade.exit_time = self.data.iloc[-1].get("timestamp", datetime.now())
            open_trade.pnl = pnl
            open_trade.result = "TIMEOUT"
            result.trades.append(open_trade)

        result.final_capital = capital
        summary = result.summary()
        logger.info(f"Backtest done: {summary}")
        return result
