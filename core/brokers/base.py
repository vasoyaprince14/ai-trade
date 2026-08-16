"""
Abstract broker interface.
All brokers (Zerodha, Angel, Paper) implement this.
Adapted from vendors/ai-trader/broker/base_adapter.py
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_MARKET = "SL-M"


class ProductType(str, Enum):
    NRML = "NRML"    # Carry-forward (F&O overnight)
    MIS = "MIS"      # Intraday (auto-square-off 15:20 IST)
    CNC = "CNC"      # Cash and Carry (equity)


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: int
    exchange: str = "NFO"
    order_type: OrderType = OrderType.MARKET
    product: ProductType = ProductType.MIS
    price: float = 0.0
    trigger_price: float = 0.0
    tag: str = ""


@dataclass
class OrderResponse:
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_qty: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    raw: Dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    quantity: int         # positive = long, negative = short
    avg_price: float
    current_price: float = 0.0
    product: ProductType = ProductType.MIS
    exchange: str = "NFO"

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_price) * self.quantity


@dataclass
class AccountInfo:
    available_margin: float = 0.0
    used_margin: float = 0.0
    total_balance: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


class BaseBroker(ABC):
    """Abstract broker. All implementations must follow this interface."""

    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderResponse:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_positions(self) -> List[Position]:
        ...

    @abstractmethod
    def get_account(self) -> AccountInfo:
        ...

    @abstractmethod
    def get_ltp(self, symbol: str, exchange: str = "NFO") -> float:
        """Get last traded price."""
        ...

    def buy(self, symbol: str, qty: int, price: float = 0.0,
            product: ProductType = ProductType.MIS,
            order_type: OrderType = OrderType.MARKET,
            tag: str = "") -> OrderResponse:
        return self.place_order(OrderRequest(
            symbol=symbol, side=OrderSide.BUY, quantity=qty,
            price=price, product=product, order_type=order_type, tag=tag
        ))

    def sell(self, symbol: str, qty: int, price: float = 0.0,
             product: ProductType = ProductType.MIS,
             order_type: OrderType = OrderType.MARKET,
             tag: str = "") -> OrderResponse:
        return self.place_order(OrderRequest(
            symbol=symbol, side=OrderSide.SELL, quantity=qty,
            price=price, product=product, order_type=order_type, tag=tag
        ))

    def sell_short(self, symbol: str, qty: int, price: float = 0.0,
                   product: ProductType = ProductType.NRML,
                   tag: str = "") -> OrderResponse:
        """Sell to open (short selling / option writing)."""
        return self.place_order(OrderRequest(
            symbol=symbol, side=OrderSide.SELL, quantity=qty,
            price=price, product=product,
            order_type=OrderType.MARKET if price == 0 else OrderType.LIMIT,
            tag=tag
        ))
