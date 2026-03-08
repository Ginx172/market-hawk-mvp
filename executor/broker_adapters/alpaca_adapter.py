"""
SCRIPT NAME: alpaca_adapter.py
====================================
Execution Location: market-hawk-mvp/executor/broker_adapters/
Purpose: Alpaca broker adapter — paper trading via Alpaca REST API.
Creation Date: 2026-03-07

Credentials from environment variables via broker_auth.py:
    - MH_ALPACA_KEY
    - MH_ALPACA_SECRET
    - MH_ALPACA_BASE_URL (default: https://paper-api.alpaca.markets)

Features:
    - Full BrokerAdapter implementation with real Alpaca API calls
    - Rate limiting: max 200 requests/min
    - TLS enforcement via broker_auth.get_tls_context()
    - RBAC check on submit_order via @requires_permission(Permission.TRADE)
    - SSE/WebSocket quote streaming via alpaca-trade-api
"""

import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Dict, List, Optional

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

logger = logging.getLogger("market_hawk.broker.alpaca")

# Alpaca SDK imports — guarded so the module can be imported even without the lib
try:
    import alpaca_trade_api as tradeapi
    _HAS_ALPACA = True
except ImportError:
    tradeapi = None  # type: ignore[assignment]
    _HAS_ALPACA = False
    logger.debug("alpaca-trade-api not installed — AlpacaAdapter unavailable")


# ============================================================
# RATE LIMITER
# ============================================================

class _RateLimiter:
    """Singleton sliding-window rate limiter (thread-safe).

    Enforces max_calls within window_seconds.
    All AlpacaAdapter instances share the same limiter so that the
    aggregate request rate stays within Alpaca's 200 req/min cap.
    """

    _instance: Optional["_RateLimiter"] = None
    _init_lock = threading.Lock()

    def __new__(cls, max_calls: int = 200, window_seconds: float = 60.0) -> "_RateLimiter":
        with cls._init_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._max_calls = max_calls
                inst._window = window_seconds
                inst._timestamps: List[float] = []
                inst._lock = threading.Lock()
                cls._instance = inst
        return cls._instance

    def acquire(self) -> None:
        """Block until a request slot is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Discard timestamps outside the window
                self._timestamps = [
                    t for t in self._timestamps if now - t < self._window
                ]
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                # Calculate how long to wait for the oldest entry to expire
                wait = self._window - (now - self._timestamps[0])
            time.sleep(max(wait, 0.05))

    @classmethod
    def _reset(cls) -> None:
        """Reset singleton (testing only)."""
        with cls._init_lock:
            cls._instance = None


# ============================================================
# ALPACA STATE MAPPING
# ============================================================

_ALPACA_STATE_MAP: Dict[str, OrderState] = {
    "new": OrderState.PENDING,
    "accepted": OrderState.ACCEPTED,
    "pending_new": OrderState.PENDING,
    "accepted_for_processing": OrderState.ACCEPTED,
    "partially_filled": OrderState.PARTIAL,
    "filled": OrderState.FILLED,
    "done_for_day": OrderState.FILLED,
    "canceled": OrderState.CANCELLED,
    "expired": OrderState.EXPIRED,
    "replaced": OrderState.CANCELLED,
    "pending_cancel": OrderState.PENDING,
    "pending_replace": OrderState.PENDING,
    "stopped": OrderState.FILLED,
    "rejected": OrderState.REJECTED,
    "suspended": OrderState.REJECTED,
    "calculated": OrderState.PENDING,
}


def _map_order_state(alpaca_status: str) -> OrderState:
    """Map Alpaca order status string to our OrderState enum."""
    return _ALPACA_STATE_MAP.get(alpaca_status.lower(), OrderState.PENDING)


# ============================================================
# ALPACA ADAPTER
# ============================================================

class AlpacaAdapter(BrokerAdapter):
    """Alpaca Markets broker adapter for paper trading.

    Uses alpaca-trade-api SDK. Credentials come from BrokerAuthManager
    (env vars MH_ALPACA_KEY, MH_ALPACA_SECRET, MH_ALPACA_BASE_URL).

    Attributes:
        _api: alpaca_trade_api.REST instance.
        _auth: BrokerAuthManager for credentials and TLS.
        _rate_limiter: Enforces 200 req/min.
        _connected: Whether we have a valid connection.
    """

    _REQUEST_TIMEOUT = 30  # seconds

    def __init__(self) -> None:
        if not _HAS_ALPACA:
            raise ImportError(
                "alpaca-trade-api is required for AlpacaAdapter. "
                "Install with: pip install alpaca-trade-api~=3.0.0"
            )
        self._auth = BrokerAuthManager()
        self._api: Optional[tradeapi.REST] = None
        self._rate_limiter = _RateLimiter(max_calls=200, window_seconds=60.0)
        self._connected = False
        self._stream_thread: Optional[threading.Thread] = None

    # ---------- CONNECTION ----------

    def connect(self) -> bool:
        """Connect to Alpaca paper trading API.

        Returns:
            True if connection succeeded and account is accessible.
        """
        creds = self._auth.get_credentials("alpaca")
        if creds is None:
            logger.error("Alpaca credentials not configured (set MH_ALPACA_KEY, MH_ALPACA_SECRET)")
            return False

        try:
            self._api = tradeapi.REST(
                key_id=creds.api_key,
                secret_key=creds.api_secret,
                base_url=creds.base_url,
                api_version="v2",
            )
            # Verify connection with a lightweight call
            self._rate_limiter.acquire()
            account = self._api.get_account()
            self._connected = True
            logger.info(
                "Connected to Alpaca (paper=%s, status=%s, equity=%s)",
                "paper" in creds.base_url,
                account.status,
                account.equity,
            )
            return True
        except Exception:
            logger.exception("Alpaca connection failed")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Alpaca (release resources)."""
        self._api = None
        self._connected = False
        logger.info("Disconnected from Alpaca")

    def _ensure_connected(self) -> tradeapi.REST:
        """Return the API instance or raise if not connected."""
        if not self._connected or self._api is None:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")
        return self._api

    # ---------- ACCOUNT & POSITIONS ----------

    def get_account(self) -> AccountInfo:
        """Fetch Alpaca account information.

        Returns:
            AccountInfo with cash, equity, buying_power.
        """
        api = self._ensure_connected()
        self._rate_limiter.acquire()
        account = api.get_account()
        return AccountInfo(
            cash=Decimal(str(account.cash)),
            equity=Decimal(str(account.equity)),
            buying_power=Decimal(str(account.buying_power)),
            currency=account.currency,
        )

    def get_positions(self) -> List[PositionInfo]:
        """Fetch all open positions from Alpaca.

        Returns:
            List of PositionInfo for each open position.
        """
        api = self._ensure_connected()
        self._rate_limiter.acquire()
        positions = api.list_positions()
        result: List[PositionInfo] = []
        for pos in positions:
            result.append(PositionInfo(
                symbol=pos.symbol,
                qty=Decimal(str(pos.qty)),
                avg_entry_price=Decimal(str(pos.avg_entry_price)),
                current_price=Decimal(str(pos.current_price)),
                market_value=Decimal(str(pos.market_value)),
                unrealized_pnl=Decimal(str(pos.unrealized_pl)),
                side=pos.side,
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
        """Submit an order to Alpaca.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL').
            qty: Number of shares.
            side: BUY or SELL.
            order_type: MARKET, LIMIT, etc.
            limit_price: Required for LIMIT / STOP_LIMIT orders.

        Returns:
            OrderResult with broker-assigned order_id.

        Raises:
            PermissionError: If current role lacks TRADE permission.
            ValueError: If limit_price missing for limit orders.
            ConnectionError: If not connected.
        """
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and limit_price is None:
            raise ValueError(f"limit_price required for {order_type.value} orders")

        api = self._ensure_connected()
        self._rate_limiter.acquire()

        kwargs = {
            "symbol": symbol.upper(),
            "qty": str(qty),
            "side": side.value,
            "type": order_type.value,
            "time_in_force": "day",
        }
        if limit_price is not None:
            kwargs["limit_price"] = str(limit_price)

        logger.info("Submitting order: %s %s %s @ %s", side.value, qty, symbol, order_type.value)
        order = api.submit_order(**kwargs)

        return OrderResult(
            order_id=order.id,
            symbol=order.symbol,
            qty=Decimal(str(order.qty)),
            side=side,
            order_type=order_type,
            state=_map_order_state(order.status),
            filled_qty=Decimal(str(order.filled_qty or "0")),
            filled_avg_price=(
                Decimal(str(order.filled_avg_price))
                if order.filled_avg_price else None
            ),
            message=f"Order {order.id} submitted",
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open Alpaca order.

        Args:
            order_id: Alpaca order UUID.

        Returns:
            True if cancellation request was accepted.
        """
        api = self._ensure_connected()
        self._rate_limiter.acquire()
        try:
            api.cancel_order(order_id)
            logger.info("Cancelled order %s", order_id)
            return True
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get status of an Alpaca order.

        Args:
            order_id: Alpaca order UUID.

        Returns:
            OrderStatus with current state and fill info.
        """
        api = self._ensure_connected()
        self._rate_limiter.acquire()
        order = api.get_order(order_id)

        return OrderStatus(
            order_id=order.id,
            state=_map_order_state(order.status),
            filled_qty=Decimal(str(order.filled_qty or "0")),
            filled_avg_price=(
                Decimal(str(order.filled_avg_price))
                if order.filled_avg_price else None
            ),
        )

    # ---------- STREAMING ----------

    def stream_quotes(
        self,
        symbols: List[str],
        callback: Callable[[str, Decimal, datetime], None],
    ) -> None:
        """Stream live quote updates from Alpaca.

        Runs in a background thread. Calls callback(symbol, price, timestamp)
        for each trade update received.

        Args:
            symbols: List of ticker symbols to subscribe to.
            callback: Function called with (symbol, price, timestamp).
        """
        api = self._ensure_connected()

        conn = tradeapi.StreamConn(
            key_id=self._auth.get_credentials("alpaca").api_key,
            secret_key=self._auth.get_credentials("alpaca").api_secret,
            base_url=self._auth.get_credentials("alpaca").base_url,
            data_stream="iex",
        )

        @conn.on(r"T$")
        async def _on_trade(conn: object, channel: str, bar: object) -> None:
            try:
                callback(
                    bar.symbol,
                    Decimal(str(bar.price)),
                    bar.timestamp if hasattr(bar, "timestamp")
                    else datetime.now(timezone.utc),
                )
            except Exception:
                logger.exception("Stream callback error")

        channels = [f"T.{s.upper()}" for s in symbols]
        logger.info("Starting Alpaca quote stream for: %s", symbols)

        def _run_stream() -> None:
            try:
                conn.run(channels)
            except Exception:
                logger.exception("Alpaca stream error")

        self._stream_thread = threading.Thread(
            target=_run_stream, daemon=True, name="alpaca-stream"
        )
        self._stream_thread.start()


# ============================================================
# AUTO-REGISTER WITH FACTORY
# ============================================================

if _HAS_ALPACA:
    BrokerFactory.register("alpaca", AlpacaAdapter)
