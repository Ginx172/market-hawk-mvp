"""
Shared pytest fixtures for Market Hawk 3 test suite.

Provides mock data, sample DataFrames, and reusable test objects
so individual test modules stay focused on logic validation.
"""

import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

D = Decimal


# ============================================================
# SAMPLE OHLCV DATA
# ============================================================

@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with 500 bars of trending data."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2025-01-01", periods=n, freq="h")

    # Simulate uptrend with noise
    base_price = 100.0
    returns = np.random.normal(0.0002, 0.01, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_p = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1000, 100000, n).astype(float)

    df = pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)

    return df


@pytest.fixture
def small_ohlcv_df() -> pd.DataFrame:
    """Minimal 10-bar DataFrame for unit tests that need deterministic data."""
    dates = pd.date_range("2025-06-01 09:00", periods=10, freq="h")
    return pd.DataFrame({
        "Open":   [100, 101, 102, 101, 103, 104, 103, 105, 106, 107],
        "High":   [101, 102, 103, 103, 104, 105, 105, 106, 107, 108],
        "Low":    [ 99, 100, 101, 100, 102, 103, 102, 104, 105, 106],
        "Close":  [101, 102, 101, 103, 104, 103, 105, 106, 107, 108],
        "Volume": [1000, 1200, 900, 1100, 1500, 1300, 1400, 1600, 1700, 1800],
    }, index=dates)


# ============================================================
# MOCK BRAIN DECISION
# ============================================================

@dataclass
class MockBrainDecision:
    """Lightweight mock for brain.orchestrator.BrainDecision."""
    action: str = "HOLD"
    consensus_score: float = 0.0
    approved: bool = False
    position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    agent_votes: List[Dict] = None
    reasoning: str = ""

    def __post_init__(self):
        if self.agent_votes is None:
            self.agent_votes = []


@pytest.fixture
def mock_buy_decision() -> MockBrainDecision:
    """A mock BUY decision with strong consensus."""
    return MockBrainDecision(
        action="BUY",
        consensus_score=0.75,
        approved=True,
        position_size=0.02,
        stop_loss=0.02,
        take_profit=0.04,
        reasoning="Strong bullish consensus from 3 agents",
    )


@pytest.fixture
def mock_sell_decision() -> MockBrainDecision:
    """A mock SELL decision."""
    return MockBrainDecision(
        action="SELL",
        consensus_score=-0.68,
        approved=True,
        position_size=0.02,
        stop_loss=0.02,
        take_profit=0.04,
        reasoning="Bearish consensus",
    )


@pytest.fixture
def mock_hold_decision() -> MockBrainDecision:
    """A mock HOLD decision (no action)."""
    return MockBrainDecision(
        action="HOLD",
        consensus_score=0.15,
        approved=False,
    )


# ============================================================
# DECIMAL HELPERS
# ============================================================

@pytest.fixture
def decimal_price() -> Decimal:
    """A sample Decimal price for tests."""
    return D("150.25")


@pytest.fixture
def decimal_capital() -> Decimal:
    """Standard $100K starting capital as Decimal."""
    return D("100000")
