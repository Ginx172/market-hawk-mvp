"""
SCRIPT NAME: ib_adapter.py
====================================
Execution Location: market-hawk-mvp/executor/broker_adapters/
Purpose: Interactive Brokers adapter — paper trading via TWS/Gateway + ib_insync.
Creation Date: 2026-03-07

Credentials from environment variables via broker_auth.py:
    - MH_IB_HOST (default: 127.0.0.1)
    - MH_IB_PORT (default: 7497 — paper trading)
    - MH_IB_CLIENT_ID (default: 1)

Features:
    - Full BrokerAdapter implementation with ib_insync
    - Reconnect logic with exponential backoff (max 5 retries)
    - RBAC check on submit_order via @requires_permission(Permission.TRADE)
    - Live tick streaming via ib_insync reqMktData
"""

import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, List, Optional

from config.rbac import Permission, requires_permission
from executor.broker_adapter import (
    AccountInfo,
    BrokerAdapter,
    BrokerFactory,
    OrderResult,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
    PositionInfo,
)
from executor.broker_auth import BrokerAuthManager

logger = logging.getLogger("market_hawk.broker.ib")

# ib_insync imports — guarded
try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, Trade
    _HAS_IB = True
except ImportError:
    IB = None  # type: ignore[assignment,misc]
    _HAS_IB = False
    logger.debug("ib_insync not installed — IBAdapter unavailable")


# ============================================================
# IB STATE MAPPING
# ============================================================

_IB_STATE_MAP = {
    "PendingSubmit": OrderState.PENDING,
    "PendingCancel": OrderState.PENDING,
    "PreSubmitted": OrderState.ACCEPTED,
    "Submitted": OrderState.ACCEPTED,
    "ApiPending": OrderState.PENDING,
    "ApiCancelled": OrderState.CANCELLED,
    "Cancelled": OrderState.CANCELLED,
    "Filled": OrderState.FILLED,
    "Inactive": OrderState.REJECTED,
}


def _map_ib_state(status: str) -> OrderState:
    """Map IB order status string to our OrderState enum."""
    return _IB_STATE_MAP.get(status, OrderState.PENDING)


# ============================================================
# IB ADAPTER
# ============================================================

class IBAdapter(BrokerAdapter):
    """Interactive Brokers adapter via ib_insync.

    Connects to TWS or IB Gateway. Default port 7497 = paper trading.

    Attributes:
        _ib: ib_insync.IB connection instance.
        _auth: BrokerAuthManager for credentials.
        _connected: Whether we have a live connection.
        _max_retries: Max reconnect attempts.
        _base_delay: Initial backoff delay in seconds.
    """

    _REQUEST_TIMEOUT = 30  # seconds
    _MAX_RETRIES = 5
    _BASE_DELAY = 1.0  # seconds (exponential backoff base)

    def __init__(self) -> None:
        if not _HAS_IB:
            raise ImportError(
                "ib_insync is required for IBAdapter. "
                "Install with: pip install ib_insync~=0.9.0"
            )
        self._auth = BrokerAuthManager()
        self._ib: Optional[IB] = None
        self._connected = False

    # ---------- CONNECTION ----------

    def connect(self) -> bool:
        """Connect to TWS/Gateway with exponential backoff retry.

        Returns:
            True if connection succeeded.
        """
        creds = self._auth.get_credentials("ib")
        if creds is None:
            logger.error("IB credentials not configured (set MH_IB_HOST)")
            return False

        self._ib = IB()
        delay = self._BASE_DELAY

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                self._ib.connect(
                    host=creds.host,
                    port=creds.port,
                    clientId=creds.client_id,
                    timeout=self._REQUEST_TIMEOUT,
                    readonly=False,
                )
                self._connected = True
                logger.info(
                    "Connected to IB TWS/Gateway at %s:%d (client %d, attempt %d)",
                    creds.host, creds.port, creds.client_id, attempt,
                )
                return True
            except Exception:
                logger.warning(
                    "IB connection attempt %d/%d failed (retry in %.1fs)",
                    attempt, self._MAX_RETRIES, delay,
                    exc_info=True,
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)  # cap la 60s

        logger.error("IB connection failed after %d attempts", self._MAX_RETRIES)
        self._connected = False
        return False

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
        self._ib = None
        self._connected = False
        logger.info("Disconnected from IB")

    def _ensure_connected(self) -> IB:
        """Return the IB instance or raise if not connected."""
        if not self._connected or self._ib is None or not self._ib.isConnected():
            raise ConnectionError("Not connected to IB. Call connect() first.")
        return self._ib

    def _reconnect_if_needed(self) -> IB:
        """Reconnect if the connection dropped, then return IB instance."""
        if self._ib is not None and self._ib.isConnected():
            return self._ib
        logger.warning("IB connection lost — attempting reconnect")
        if self.connect():
            return self._ib  # type: ignore[return-value]
        raise ConnectionError("IB reconnect failed")

    # ---------- ACCOUNT & POSITIONS ----------

    def get_account(self) -> AccountInfo:
        """Fetch IB account information.

        Returns:
            AccountInfo with cash, equity, buying_power.
        """
        ib = self._reconnect_if_needed()
        ib.reqAccountSummary()
        # ib_insync caches account values after reqAccountSummary
        summaries = ib.accountSummary()

        values = {}
        for item in summaries:
            values[item.tag] = item.value

        return AccountInfo(
            cash=Decimal(str(values.get("TotalCashValue", "0"))),
            equity=Decimal(str(values.get("NetLiquidation", "0"))),
            buying_power=Decimal(str(values.get("BuyingPower", "0"))),
            currency=values.get("Currency", "USD"),
        )

    def get_positions(self) -> List[PositionInfo]:
        """Fetch all open IB positions.

        Returns:
            List of PositionInfo for each position.
        """
        ib = self._reconnect_if_needed()
        positions = ib.positions()
        result: List[PositionInfo] = []

        for pos in positions:
            qty = Decimal(str(pos.position))
            avg_cost = Decimal(str(pos.avgCost))
            # ib_insync nu da current_price direct in positions(),
            # trebuie reqMktData, dar folosim avgCost ca fallback
            result.append(PositionInfo(
                symbol=pos.contract.symbol,
                qty=abs(qty),
                avg_entry_price=avg_cost,
                current_price=avg_cost,  # actualizat la stream_quotes
                market_value=abs(qty) * avg_cost,
                unrealized_pnl=Decimal("0"),  # se actualizeaza via PnL
                side="long" if qty > 0 else "short",
            ))

        return result

    # ---------- ORDERS ----------

    @requires_permission(Permission.TRADE)
    def submit_order(
        self,
        symbol: str,
        qty: Decimal,
        side: OrderSide,
        order_type: OrderType,
        limit_price: Optional[Decimal] = None,
    ) -> OrderResult:
        """Submit an order to Interactive Brokers.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL').
            qty: Number of shares.
            side: BUY or SELL.
            order_type: MARKET, LIMIT, etc.
            limit_price: Required for LIMIT / STOP_LIMIT orders.

        Returns:
            OrderResult with IB-assigned order_id.

        Raises:
            PermissionError: If current role lacks TRADE permission.
            ValueError: If limit_price missing for limit orders.
        """
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and limit_price is None:
            raise ValueError(f"limit_price required for {order_type.value} orders")

        ib = self._reconnect_if_needed()
        contract = Stock(symbol.upper(), "SMART", "USD")
        ib.qualifyContracts(contract)

        action = "BUY" if side == OrderSide.BUY else "SELL"

        if order_type == OrderType.MARKET:
            order = MarketOrder(action, float(qty))
        elif order_type == OrderType.LIMIT:
            order = LimitOrder(action, float(qty), float(limit_price))  # type: ignore[arg-type]
        elif order_type == OrderType.STOP:
            order = StopOrder(action, float(qty), float(limit_price))  # type: ignore[arg-type]
        else:
            # STOP_LIMIT — ib_insync does not have a dedicated class,
            # build via Order()
            from ib_insync import Order as IBOrder
            order = IBOrder(
                action=action,
                totalQuantity=float(qty),
                orderType="STP LMT",
                lmtPrice=float(limit_price),  # type: ignore[arg-type]
                auxPrice=float(limit_price),  # type: ignore[arg-type]
            )

        logger.info("Submitting IB order: %s %s %s @ %s", action, qty, symbol, order_type.value)
        trade: Trade = ib.placeOrder(contract, order)
        ib.sleep(0.5)  # Permite IB sa proceseze

        return OrderResult(
            order_id=str(trade.order.orderId),
            symbol=symbol.upper(),
            qty=qty,
            side=side,
            order_type=order_type,
            state=_map_ib_state(trade.orderStatus.status),
            filled_qty=Decimal(str(trade.orderStatus.filled)),
            filled_avg_price=(
                Decimal(str(trade.orderStatus.avgFillPrice))
                if trade.orderStatus.avgFillPrice else None
            ),
            message=f"IB order {trade.order.orderId} placed",
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open IB order.

        Args:
            order_id: IB order ID (as string).

        Returns:
            True if cancellation was accepted.
        """
        ib = self._reconnect_if_needed()
        try:
            # Gasim trade-ul dupa orderId
            target_id = int(order_id)
            for trade in ib.openTrades():
                if trade.order.orderId == target_id:
                    ib.cancelOrder(trade.order)
                    ib.sleep(0.5)
                    logger.info("Cancelled IB order %s", order_id)
                    return True
            logger.warning("IB order %s not found in open trades", order_id)
            return False
        except Exception:
            logger.exception("Failed to cancel IB order %s", order_id)
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get status of an IB order.

        Args:
            order_id: IB order ID (as string).

        Returns:
            OrderStatus with current state and fill info.

        Raises:
            ValueError: If order_id not found.
        """
        ib = self._reconnect_if_needed()
        target_id = int(order_id)

        # Cauta in toate trade-urile (open + completed)
        for trade in ib.trades():
            if trade.order.orderId == target_id:
                return OrderStatus(
                    order_id=order_id,
                    state=_map_ib_state(trade.orderStatus.status),
                    filled_qty=Decimal(str(trade.orderStatus.filled)),
                    filled_avg_price=(
                        Decimal(str(trade.orderStatus.avgFillPrice))
                        if trade.orderStatus.avgFillPrice else None
                    ),
                )

        raise ValueError(f"IB order {order_id} not found")

    # ---------- STREAMING ----------

    def stream_quotes(
        self,
        symbols: List[str],
        callback: Callable[[str, Decimal, datetime], None],
    ) -> None:
        """Stream live tick data from IB via reqMktData.

        Subscribes to real-time market data and calls callback
        on each price update. Runs in a background thread.

        Args:
            symbols: List of ticker symbols.
            callback: Function called with (symbol, price, timestamp).
        """
        ib = self._reconnect_if_needed()

        contracts = []
        for sym in symbols:
            contract = Stock(sym.upper(), "SMART", "USD")
            ib.qualifyContracts(contract)
            contracts.append(contract)

        def _on_tick(ticker: object) -> None:
            try:
                price = getattr(ticker, "last", None) or getattr(ticker, "close", None)
                if price is not None and price > 0:
                    callback(
                        ticker.contract.symbol,
                        Decimal(str(price)),
                        datetime.now(timezone.utc),
                    )
            except Exception:
                logger.exception("IB tick callback error")

        for contract in contracts:
            ticker = ib.reqMktData(contract, "", False, False)
            ticker.updateEvent += _on_tick

        logger.info("Started IB quote stream for: %s", symbols)

        def _run_loop() -> None:
            try:
                ib.run()
            except Exception:
                logger.exception("IB event loop error")

        stream_thread = threading.Thread(
            target=_run_loop, daemon=True, name="ib-stream"
        )
        stream_thread.start()


# ============================================================
# AUTO-REGISTER WITH FACTORY
# ============================================================

if _HAS_IB:
    BrokerFactory.register("ib", IBAdapter)
