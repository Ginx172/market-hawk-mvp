"""Market Hawk MVP — executor module.

Phase 4: Broker Integration with Adapter Pattern.

Exports:
    BrokerAdapter, BrokerFactory — core adapter interface and factory
    AccountInfo, PositionInfo, OrderResult, OrderStatus — data models
    OrderSide, OrderType, OrderState — enums
    BrokerAuthManager — credential management
"""

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

__all__ = [
    "AccountInfo",
    "BrokerAdapter",
    "BrokerAuthManager",
    "BrokerFactory",
    "OrderResult",
    "OrderSide",
    "OrderState",
    "OrderStatus",
    "OrderType",
    "PositionInfo",
]
