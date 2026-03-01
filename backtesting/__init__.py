"""
Market Hawk 3 — Backtesting Engine Package
==========================================
Enterprise-grade backtesting with local historical data support.
"""

from backtesting.engine import BacktestEngine
from backtesting.data_loader import HistoricalDataLoader
from backtesting.strategies import StrategyBase, MACrossover, RSIMeanReversion, AgentConsensusStrategy
from backtesting.report import BacktestReport
from backtesting.alerts import AlertManager, AlertRule, AlertLevel, AlertChannel, attach_alerts_to_engine

__all__ = [
    "BacktestEngine",
    "HistoricalDataLoader",
    "StrategyBase",
    "MACrossover",
    "RSIMeanReversion",
    "AgentConsensusStrategy",
    "BacktestReport",
    "AlertManager",
    "AlertRule",
    "AlertLevel",
    "AlertChannel",
    "attach_alerts_to_engine",
]
