"""
Zerodha Kite Connect Broker Adapter
====================================
Wraps kiteconnect library for live trading.
Requires API key + access token (generate daily or use auto-login with TOTP).

Reference:
  - vendors/ai-trader/broker/zerodha_adapter.py
  - vendors/algo-strategies/short-straddle/ (for symbol formatting)
  - kiteconnect docs: https://kite.trade/docs/connect/v3/
"""
from datetime import datetime
from typing import List, Optional
from loguru import logger

from core.brokers.base import (
    BaseBroker, OrderRequest, OrderResponse, Position,
    AccountInfo, OrderSide, OrderStatus, OrderType, ProductType
)
from config.settings import settings


class ZerodhaBroker(BaseBroker):
    """
    Live trading via Zerodha Kite Connect API.
    Set ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN in .env
    """

    def __init__(self):
        self._kite = None
        self._connected = False
        self._connect()

    def _connect(self):
        try:
            from kiteconnect import KiteConnect
            api_key = settings.ZERODHA_API_KEY
            access_token = settings.ZERODHA_ACCESS_TOKEN

            if not api_key:
                logger.error("ZERODHA_API_KEY not set in .env")
                return

            self._kite = KiteConnect(api_key=api_key)
            if access_token:
                self._kite.set_access_token(access_token)
                # Verify by fetching profile
                try:
                    profile = self._kite.profile()
                    logger.info(f"Zerodha connected: {profile.get('user_name', 'Unknown')}")
                    self._connected = True
                except Exception as e:
                    logger.error(f"Access token invalid: {e}")
            else:
                logger.warning("ZERODHA_ACCESS_TOKEN not set. Use generate_session() to login.")
        except ImportError:
            logger.error("kiteconnect not installed. Run: pip install kiteconnect")

    def generate_login_url(self) -> str:
        """Get Zerodha login URL to obtain request token."""
        if self._kite:
            return self._kite.login_url()
        return ""

    def generate_session(self, request_token: str) -> str:
        """Exchange request_token for access_token. Save to .env."""
        if not self._kite:
            return ""
        data = self._kite.generate_session(request_token, settings.ZERODHA_API_SECRET)
        access_token = data.get("access_token", "")
        self._kite.set_access_token(access_token)
        self._connected = True
        logger.info(f"Session generated. Add ZERODHA_ACCESS_TOKEN={access_token} to .env")
        return access_token

    def _check(self):
        if not self._connected or not self._kite:
            raise RuntimeError("Zerodha not connected. Check API keys in .env")

    def place_order(self, req: OrderRequest) -> OrderResponse:
        self._check()
        try:
            kite_order_type = {
                OrderType.MARKET: self._kite.ORDER_TYPE_MARKET,
                OrderType.LIMIT: self._kite.ORDER_TYPE_LIMIT,
                OrderType.SL: self._kite.ORDER_TYPE_SL,
                OrderType.SL_MARKET: self._kite.ORDER_TYPE_SLM,
            }.get(req.order_type, self._kite.ORDER_TYPE_MARKET)

            kite_product = {
                ProductType.NRML: self._kite.PRODUCT_NRML,
                ProductType.MIS: self._kite.PRODUCT_MIS,
                ProductType.CNC: self._kite.PRODUCT_CNC,
            }.get(req.product, self._kite.PRODUCT_MIS)

            kite_side = (
                self._kite.TRANSACTION_TYPE_BUY
                if req.side == OrderSide.BUY
                else self._kite.TRANSACTION_TYPE_SELL
            )

            order_id = self._kite.place_order(
                tradingsymbol=req.symbol,
                exchange=req.exchange,
                transaction_type=kite_side,
                quantity=req.quantity,
                order_type=kite_order_type,
                product=kite_product,
                price=req.price if req.order_type == OrderType.LIMIT else None,
                trigger_price=req.trigger_price if req.trigger_price > 0 else None,
                tag=req.tag[:20] if req.tag else None,
                variety=self._kite.VARIETY_REGULAR,
            )
            logger.info(f"[LIVE] Order placed: {order_id} | {req.side.value} {req.quantity}x {req.symbol}")
            return OrderResponse(
                order_id=str(order_id),
                status=OrderStatus.OPEN,
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return OrderResponse(status=OrderStatus.REJECTED, message=str(e))

    def cancel_order(self, order_id: str) -> bool:
        self._check()
        try:
            self._kite.cancel_order(variety=self._kite.VARIETY_REGULAR, order_id=order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def get_positions(self) -> List[Position]:
        self._check()
        try:
            raw = self._kite.positions()
            positions = []
            for p in raw.get("net", []):
                if p.get("quantity", 0) != 0:
                    positions.append(Position(
                        symbol=p["tradingsymbol"],
                        quantity=p["quantity"],
                        avg_price=p["average_price"],
                        current_price=p.get("last_price", p["average_price"]),
                        exchange=p.get("exchange", "NFO"),
                    ))
            return positions
        except Exception as e:
            logger.error(f"Get positions failed: {e}")
            return []

    def get_account(self) -> AccountInfo:
        self._check()
        try:
            margins = self._kite.margins()
            equity = margins.get("equity", {})
            return AccountInfo(
                available_margin=equity.get("available", {}).get("live_balance", 0),
                used_margin=equity.get("utilised", {}).get("debits", 0),
                total_balance=equity.get("net", 0),
            )
        except Exception as e:
            logger.error(f"Get margins failed: {e}")
            return AccountInfo()

    def get_ltp(self, symbol: str, exchange: str = "NFO") -> float:
        self._check()
        try:
            key = f"{exchange}:{symbol}"
            data = self._kite.ltp([key])
            return data.get(key, {}).get("last_price", 0.0)
        except Exception as e:
            logger.error(f"LTP fetch failed: {e}")
            return 0.0

    @staticmethod
    def format_option_symbol(index: str, expiry_date, strike: float, option_type: str) -> str:
        """
        Format option trading symbol for Zerodha.
        Example: NIFTY26041323800PE
        Format: {INDEX}{YY}{MMM}{DD}{STRIKE}{TYPE}
        """
        month_map = {
            1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
            7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
        }
        # Weekly options: NIFTY + YY + M(1 char for months Oct-Dec) + DD + strike + type
        # Monthly: NIFTY + YY + MMM + strike + type
        # Simplified format (adjust per actual NSE symbol):
        yy = str(expiry_date.year)[-2:]
        mon = month_map[expiry_date.month][:3]
        dd = f"{expiry_date.day:02d}"
        strike_str = str(int(strike))
        return f"{index}{yy}{dd}{mon}{strike_str}{option_type}"
