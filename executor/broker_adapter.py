"""
SCRIPT NAME: broker_adapter.py
====================================
Execution Location: market-hawk-mvp/executor/
Purpose: Broker Integration — Adapter Pattern for paper trading.
         Abstract base class, common dataclasses, and BrokerFactory.
Creation Date: 2026-03-07

Provides:
    - BrokerAdapter (ABC): interface for all broker integrations
    - AccountInfo, PositionInfo, OrderResult, OrderStatus: shared data models
    - BrokerFactory: creates the right adapter from config / env var

Usage:
    from executor.broker_adapter import BrokerFactory

    broker = BrokerFactory.create_broker("alpaca")
    broker.connect()
    account = broker.get_account()
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, unique
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("market_hawk.broker_adapter")


# ============================================================
# ENUMS
# ============================================================

@unique
class OrderSide(Enum):
    """Trade direction."""
    BUY = "buy"
    SELL = "sell"


@unique
class OrderType(Enum):
    """Order execution type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@unique
class OrderState(Enum):
    """Lifecycle state of an order."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIAL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


# ============================================================
# DATACLASSES
# ============================================================

@dataclass(frozen=True)
class AccountInfo:
    """Broker account snapshot."""
    cash: Decimal
    equity: Decimal
    buying_power: Decimal
    currency: str = "USD"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class PositionInfo:
    """A single open position."""
    symbol: str
    qty: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    side: str = "long"


@dataclass(frozen=True)
class OrderResult:
    """Result returned after submitting an order."""
    order_id: str
    symbol: str
    qty: Decimal
    side: OrderSide
    order_type: OrderType
    state: OrderState
    filled_qty: Decimal = Decimal("0")
    filled_avg_price: Optional[Decimal] = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""


@dataclass(frozen=True)
class OrderStatus:
    """Current status of an existing order."""
    order_id: str
    state: OrderState
    filled_qty: Decimal = Decimal("0")
    filled_avg_price: Optional[Decimal] = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================
# ABSTRACT BASE CLASS
# ============================================================

class BrokerAdapter(ABC):
    """Abstract broker interface — all broker integrations implement this.

    Subclasses must implement every abstract method. The adapter pattern
    lets the rest of the system be broker-agnostic.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker.

        Returns:
            True if connection succeeded.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully close the broker connection."""

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Fetch current account information.

        Returns:
            AccountInfo with cash, equity, buying_power.
        """

    @abstractmethod
    def get_positions(self) -> List[PositionInfo]:
        """Fetch all open positions.

        Returns:
            List of PositionInfo for each position.
        """

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: Decimal,
        side: OrderSide,
        order_type: OrderType,
        limit_price: Optional[Decimal] = None,
    ) -> OrderResult:
        """Submit a new order to the broker.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL').
            qty: Number of shares.
            side: BUY or SELL.
            order_type: MARKET, LIMIT, etc.
            limit_price: Required for LIMIT and STOP_LIMIT orders.

        Returns:
            OrderResult with order_id and initial state.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Broker-assigned order identifier.

        Returns:
            True if cancellation was accepted.
        """

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Query the current status of an order.

        Args:
            order_id: Broker-assigned order identifier.

        Returns:
            OrderStatus with current state and fill info.
        """

    @abstractmethod
    def stream_quotes(
        self,
        symbols: List[str],
        callback: Callable[[str, Decimal, datetime], None],
    ) -> None:
        """Stream live price updates for a list of symbols.

        Args:
            symbols: List of ticker symbols to subscribe to.
            callback: Called with (symbol, price, timestamp) on each update.
        """


# ============================================================
# BROKER FACTORY
# ============================================================

class BrokerFactory:
    """Factory for creating broker adapter instances.

    Usage:
        broker = BrokerFactory.create_broker("alpaca")
        broker = BrokerFactory.create_broker()  # reads MH_DEFAULT_BROKER
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: type) -> None:
        """Register a broker adapter class under a name.

        Args:
            name: Lowercase broker identifier (e.g., 'alpaca').
            adapter_cls: Class that extends BrokerAdapter.
        """
        cls._registry[name.lower()] = adapter_cls
        logger.debug("Registered broker adapter: %s -> %s", name, adapter_cls.__name__)

    @classmethod
    def create_broker(cls, name: Optional[str] = None) -> BrokerAdapter:
        """Create and return a broker adapter instance.

        Args:
            name: Broker identifier ('alpaca', 'ib'). If None, reads
                  MH_DEFAULT_BROKER env var (default: 'alpaca').

        Returns:
            A BrokerAdapter instance.

        Raises:
            ValueError: If the broker name is not recognized.
        """
        if name is None:
            name = os.environ.get("MH_DEFAULT_BROKER", "alpaca")

        name_lower = name.lower()

        # Lazy-import adapters so they register themselves
        if not cls._registry:
            _import_adapters()

        if name_lower not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "(none)"
            raise ValueError(
                f"Unknown broker {name!r}. Available: {available}"
            )

        adapter_cls = cls._registry[name_lower]
        logger.info("Creating broker adapter: %s (%s)", name_lower, adapter_cls.__name__)
        return adapter_cls()


def _import_adapters() -> None:
    """Lazy-import adapter modules so they auto-register with BrokerFactory."""
    try:
        from executor.broker_adapters import alpaca_adapter  # noqa: F401
    except ImportError:
        logger.debug("alpaca_adapter not available (missing dependencies?)")

    try:
        from executor.broker_adapters import ib_adapter  # noqa: F401
    except ImportError:
        logger.debug("ib_adapter not available (missing dependencies?)")
