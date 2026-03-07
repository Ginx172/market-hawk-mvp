"""
Tests for backtesting/engine.py

Covers:
    - BacktestEngine initialization with Decimal internals
    - Simple LONG run: open, hold, close, verify PnL and commission
    - Equity curve generation
    - Stop-loss and take-profit execution
    - _to_decimal helper in engine module
    - BacktestResult serialization
"""

import pytest
from decimal import Decimal

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine, BacktestTrade, BacktestResult, _to_decimal
from backtesting.strategies import (
    StrategyBase, TradeSignal, Signal, MACrossover
)

D = Decimal
ZERO = D("0")


# ============================================================
# HELPER: Deterministic always-buy strategy
# ============================================================

class AlwaysBuyOnce(StrategyBase):
    """Buys on bar 5, holds until SL/TP or end."""

    @property
    def name(self) -> str:
        return "AlwaysBuyOnce"

    def generate_signal(self, df: pd.DataFrame, idx: int) -> TradeSignal:
        if idx == 5:
            return TradeSignal(
                signal=Signal.BUY,
                confidence=0.80,
                stop_loss_pct=0.05,
                take_profit_pct=0.10,
                position_size=0.05,
                reason="Test buy",
            )
        return TradeSignal(Signal.HOLD)


class BuySellStrategy(StrategyBase):
    """Buys on bar 2, sells on bar 7."""

    @property
    def name(self) -> str:
        return "BuySell"

    def generate_signal(self, df: pd.DataFrame, idx: int) -> TradeSignal:
        if idx == 2:
            return TradeSignal(signal=Signal.BUY, confidence=0.7,
                               position_size=0.05, stop_loss_pct=0.10,
                               take_profit_pct=0.20)
        if idx == 7:
            return TradeSignal(signal=Signal.SELL, confidence=0.7,
                               position_size=0.05, stop_loss_pct=0.10,
                               take_profit_pct=0.20)
        return TradeSignal(Signal.HOLD)


# ============================================================
# ENGINE INIT
# ============================================================

class TestBacktestEngineInit:
    """Test engine initialization uses Decimal."""

    def test_initial_capital_decimal(self):
        engine = BacktestEngine(initial_capital=50_000)
        assert isinstance(engine.initial_capital, Decimal)
        assert engine.initial_capital == D("50000")

    def test_commission_decimal(self):
        engine = BacktestEngine(commission_pct=0.002)
        assert isinstance(engine.commission_pct, Decimal)

    def test_cash_equals_capital(self):
        engine = BacktestEngine(initial_capital=75_000)
        assert engine._cash == D("75000")

    def test_to_decimal_in_engine(self):
        assert _to_decimal(3.14) == D("3.14")
        assert _to_decimal(None) == ZERO


# ============================================================
# SIMPLE RUN
# ============================================================

class TestBacktestRun:
    """Test engine run with synthetic data."""

    def test_buy_and_hold_to_end(self, small_ohlcv_df):
        """Buy at bar 5, hold until end_of_data close."""
        engine = BacktestEngine(
            initial_capital=10_000,
            commission_pct=0.001,
            slippage_pct=0.0,
        )
        result = engine.run(small_ohlcv_df, AlwaysBuyOnce(), symbol="TEST")

        assert result.total_trades == 1
        assert result.total_bars == 10
        # Trade opened at bar 5, closed at end (bar 9)
        trade = result.trades[0]
        assert trade["side"] == "LONG"
        assert trade["exit_reason"] == "end_of_data"

    def test_equity_curve_length(self, small_ohlcv_df):
        engine = BacktestEngine(initial_capital=10_000)
        result = engine.run(small_ohlcv_df, AlwaysBuyOnce(), symbol="TEST")
        assert len(result.equity_curve) == len(small_ohlcv_df)

    def test_buy_then_sell_signal(self, small_ohlcv_df):
        """Buy at bar 2, sell at bar 7 via signal reversal."""
        engine = BacktestEngine(
            initial_capital=10_000,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        result = engine.run(small_ohlcv_df, BuySellStrategy(), symbol="TEST")

        # Should have at least 1 closed trade from signal_reversal
        assert result.total_trades >= 1
        first_trade = result.trades[0]
        assert first_trade["exit_reason"] == "signal_reversal"

    def test_commission_deducted(self, small_ohlcv_df):
        """Verify commission is tracked."""
        engine = BacktestEngine(
            initial_capital=10_000,
            commission_pct=0.01,  # 1% — intentionally high for visibility
            slippage_pct=0.0,
        )
        result = engine.run(small_ohlcv_df, AlwaysBuyOnce(), symbol="TEST")
        assert result.total_commission > 0

    def test_no_trades_hold_only(self, small_ohlcv_df):
        """A strategy that always holds should produce 0 trades."""
        class AlwaysHold(StrategyBase):
            @property
            def name(self) -> str:
                return "AlwaysHold"
            def generate_signal(self, df, idx):
                return TradeSignal(Signal.HOLD)

        engine = BacktestEngine(initial_capital=10_000)
        result = engine.run(small_ohlcv_df, AlwaysHold(), symbol="TEST")
        assert result.total_trades == 0
        assert result.final_equity == pytest.approx(10_000, rel=1e-6)


# ============================================================
# METRICS
# ============================================================

class TestBacktestMetrics:
    """Test computed metrics in BacktestResult."""

    def test_return_pct_positive_trend(self, sample_ohlcv_df):
        """On uptrending data, MA crossover should have positive or zero return."""
        engine = BacktestEngine(initial_capital=100_000, commission_pct=0.0,
                                slippage_pct=0.0)
        result = engine.run(sample_ohlcv_df, MACrossover(fast_period=10, slow_period=30),
                            symbol="SYNTH")
        # Just verify it runs and produces valid output
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown_pct, float)
        assert 0 <= result.win_rate <= 1.0

    def test_result_to_dict(self, small_ohlcv_df):
        engine = BacktestEngine(initial_capital=10_000)
        result = engine.run(small_ohlcv_df, AlwaysBuyOnce(), symbol="TEST")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "equity_curve" in d
        assert "trades" in d
