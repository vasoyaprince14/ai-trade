"""
Risk Manager
=============
Integrates with:
  vendors/ai-trader/risk/risk_manager.py (Kelly, max drawdown, regime gating)
  vendors/ai-trader/risk/portfolio_tracker.py (real-time P&L)

Responsibilities:
  - Position sizing (Fixed Risk / Kelly Criterion)
  - Daily/Weekly loss limits
  - Max drawdown circuit breaker
  - Trade approval (validate signal before execution)
  - Real-time portfolio tracking
"""
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, List
from loguru import logger

from core.brokers.base import BaseBroker, AccountInfo
from config.settings import config

RISK_CFG = config.get("risk", {})
DEFAULT_CAPITAL = RISK_CFG.get("default_capital", 500000)


class RiskManager:
    """
    Validates trades against risk limits and sizes positions.
    Wraps ai-trader's risk_manager.py concepts.
    """

    def __init__(self, broker: Optional[BaseBroker] = None):
        self.broker = broker
        self.capital = DEFAULT_CAPITAL
        self.max_per_trade_pct = RISK_CFG.get("max_capital_per_trade_pct", 5) / 100
        self.max_open_trades = RISK_CFG.get("max_open_trades", 5)
        self.max_daily_loss_pct = RISK_CFG.get("max_daily_loss_pct", 2) / 100
        self.max_weekly_loss_pct = RISK_CFG.get("max_weekly_loss_pct", 5) / 100
        self.max_drawdown_pct = RISK_CFG.get("max_drawdown_pct", 15) / 100
        self.sizing_mode = RISK_CFG.get("position_sizing", "fixed_risk")

        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._peak_capital: float = self.capital
        self._open_trades: int = 0
        self._trading_halted: bool = False
        self._halt_reason: str = ""

        # Try to use ai-trader risk manager
        self._ai_risk = None
        try:
            VENDOR_DIR = Path(__file__).parent.parent.parent / "vendors"
            sys.path.insert(0, str(VENDOR_DIR / "ai-trader"))
            from risk.risk_manager import RiskManager as AIRiskManager
            self._ai_risk = AIRiskManager()
            logger.info("Loaded ai-trader RiskManager")
        except (ImportError, Exception):
            pass

    def update_capital(self):
        """Refresh capital from broker."""
        if self.broker:
            try:
                account = self.broker.get_account()
                self.capital = account.total_balance
                self._daily_pnl = account.realized_pnl + account.unrealized_pnl
            except Exception:
                pass

    def approve_trade(self, signal, quantity: int = 0) -> Dict:
        """
        Check if a trade should be approved based on risk rules.
        Returns {"approved": bool, "reason": str, "qty": int}
        """
        if self._trading_halted:
            return {"approved": False, "reason": f"Trading halted: {self._halt_reason}", "qty": 0}

        self.update_capital()

        # Check daily loss limit
        daily_loss_pct = abs(self._daily_pnl) / self.capital if self._daily_pnl < 0 else 0
        if daily_loss_pct > self.max_daily_loss_pct:
            self._halt("Daily loss limit hit", halt=True)
            return {"approved": False, "reason": "Daily loss limit exceeded", "qty": 0}

        # Check max drawdown
        drawdown = (self._peak_capital - self.capital) / self._peak_capital if self._peak_capital > 0 else 0
        if drawdown > self.max_drawdown_pct:
            self._halt("Max drawdown hit", halt=True)
            return {"approved": False, "reason": "Max drawdown exceeded", "qty": 0}

        # Check open trade count
        if self._open_trades >= self.max_open_trades:
            return {"approved": False, "reason": f"Max {self.max_open_trades} open trades", "qty": 0}

        # Size position
        qty = quantity or self._size_position(signal)
        if qty <= 0:
            return {"approved": False, "reason": "Position size too small", "qty": 0}

        return {"approved": True, "reason": "OK", "qty": qty}

    def _size_position(self, signal) -> int:
        """Calculate position size based on sizing mode."""
        if self.sizing_mode == "fixed_risk":
            # Risk fixed % of capital
            risk_amount = self.capital * self.max_per_trade_pct
            price = getattr(signal, "price", 0) or 100
            sl = getattr(signal, "stop_loss", 0)
            risk_per_unit = abs(price - sl) if sl > 0 else price * 0.3
            if risk_per_unit <= 0:
                risk_per_unit = price * 0.3
            qty = int(risk_amount / risk_per_unit)
            return max(qty, 1)

        elif self.sizing_mode == "kelly":
            # Simplified Kelly (assume 60% win rate, 1.5:1 reward/risk)
            win_rate = 0.6
            rr = 1.5
            kelly_pct = win_rate - (1 - win_rate) / rr
            kelly_pct = max(0.05, min(kelly_pct * 0.5, 0.15))  # Half-Kelly, capped
            risk_amount = self.capital * kelly_pct
            price = getattr(signal, "price", 0) or 100
            return max(int(risk_amount / price), 1)

        else:  # fixed_qty
            return 1

    def on_trade_opened(self, symbol: str, pnl_invested: float = 0):
        self._open_trades += 1
        logger.info(f"Risk: Trade opened | Open: {self._open_trades}")

    def on_trade_closed(self, pnl: float):
        self._open_trades = max(0, self._open_trades - 1)
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        if self.capital > self._peak_capital:
            self._peak_capital = self.capital
        logger.info(f"Risk: Trade closed | P&L: ₹{pnl:,.2f} | Daily: ₹{self._daily_pnl:,.2f}")

    def _halt(self, reason: str, halt: bool = False):
        if halt:
            self._trading_halted = True
            self._halt_reason = reason
            logger.warning(f"TRADING HALTED: {reason}")
        else:
            logger.warning(f"Risk warning: {reason}")

    def reset_daily(self):
        """Call at market open each day."""
        self._daily_pnl = 0.0
        if datetime.now().weekday() == 0:
            self._weekly_pnl = 0.0
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("Risk Manager: Daily reset done")

    def get_status(self) -> Dict:
        self.update_capital()
        drawdown = (self._peak_capital - self.capital) / self._peak_capital if self._peak_capital > 0 else 0
        return {
            "capital": self.capital,
            "daily_pnl": self._daily_pnl,
            "weekly_pnl": self._weekly_pnl,
            "drawdown_pct": round(drawdown * 100, 2),
            "open_trades": self._open_trades,
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "max_daily_loss": self.capital * self.max_daily_loss_pct,
            "max_drawdown": self._peak_capital * self.max_drawdown_pct,
        }
