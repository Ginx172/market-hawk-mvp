"""
Tests for agents/risk_manager/kelly_criterion.py

Covers:
    - Kelly Criterion formula correctness
    - Edge cases: win_rate=0, win_rate=1, avg_loss=0
    - Position sizing output
    - Risk approval logic
    - Integration with RiskConfig defaults
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass


# ============================================================
# MOCK CONFIG (avoid loading real config/settings.py)
# ============================================================

@dataclass
class MockRiskConfig:
    """Minimal mock matching config.settings.RiskConfig fields."""
    max_position_pct: float = 0.02
    max_portfolio_risk: float = 0.06
    max_drawdown_pct: float = 0.15
    kelly_fraction: float = 0.25
    max_correlated_positions: int = 3
    default_stop_loss_pct: float = 0.02
    default_take_profit_pct: float = 0.04


def _make_risk_manager(config=None):
    """Instantiate RiskManager with mock config to avoid importing settings."""
    from agents.risk_manager.kelly_criterion import RiskManager
    rm = RiskManager.__new__(RiskManager)
    rm.config = config or MockRiskConfig()
    return rm


# ============================================================
# KELLY FORMULA
# ============================================================

class TestKellyFormula:
    """Test the internal _kelly_criterion calculation."""

    def test_standard_case(self):
        rm = _make_risk_manager()
        # win_rate=0.6, avg_win=0.04, avg_loss=0.02 -> b=2
        # kelly_full = (2*0.6 - 0.4) / 2 = 0.4
        # with quarter-kelly: 0.4 * 0.25 = 0.10
        result = rm._kelly_criterion(0.6, 0.04, 0.02)
        assert result == pytest.approx(0.10, abs=1e-6)

    def test_high_win_rate(self):
        rm = _make_risk_manager()
        # win_rate=0.9, avg_win=0.03, avg_loss=0.01 -> b=3
        # kelly_full = (3*0.9 - 0.1) / 3 = 2.6/3 = 0.8667
        # quarter-kelly: 0.8667 * 0.25 = 0.2167
        result = rm._kelly_criterion(0.9, 0.03, 0.01)
        assert result == pytest.approx(0.2167, abs=1e-3)

    def test_win_rate_zero(self):
        rm = _make_risk_manager()
        # win_rate=0: kelly_full = (b*0 - 1)/b = -1/b < 0 -> max(0, ...) = 0
        result = rm._kelly_criterion(0.0, 0.04, 0.02)
        assert result == 0.0

    def test_win_rate_one(self):
        rm = _make_risk_manager()
        # win_rate=1: kelly_full = (b*1 - 0)/b = 1.0
        # quarter-kelly: 1.0 * 0.25 = 0.25
        result = rm._kelly_criterion(1.0, 0.04, 0.02)
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_avg_loss_zero(self):
        rm = _make_risk_manager()
        # Division by zero guard -> return 0.0
        result = rm._kelly_criterion(0.6, 0.04, 0.0)
        assert result == 0.0

    def test_breakeven_system(self):
        rm = _make_risk_manager()
        # win_rate=0.5, avg_win=0.02, avg_loss=0.02 -> b=1
        # kelly_full = (1*0.5 - 0.5)/1 = 0 -> quarter-kelly = 0
        result = rm._kelly_criterion(0.5, 0.02, 0.02)
        assert result == 0.0

    def test_losing_system_returns_zero(self):
        rm = _make_risk_manager()
        # win_rate=0.3, avg_win=0.02, avg_loss=0.03 -> b=0.667
        # kelly_full = (0.667*0.3 - 0.7)/0.667 = (0.2 - 0.7)/0.667 = -0.75 < 0
        result = rm._kelly_criterion(0.3, 0.02, 0.03)
        assert result == 0.0


# ============================================================
# EVALUATE (RISK GATING)
# ============================================================

class TestEvaluate:
    """Test the evaluate() method that gates trading signals."""

    def test_hold_always_approved(self):
        rm = _make_risk_manager()
        result = rm.evaluate({"action": "HOLD", "consensus": 0.0})
        assert result["approved"] is True
        assert result["position_size"] == 0

    def test_buy_strong_consensus_approved(self):
        rm = _make_risk_manager()
        result = rm.evaluate({"action": "BUY", "consensus": 0.80})
        assert result["approved"] is True
        assert result["position_size"] > 0
        assert result["stop_loss"] > 0
        assert result["take_profit"] > 0

    def test_position_size_capped_at_max(self):
        rm = _make_risk_manager(MockRiskConfig(max_position_pct=0.01))
        result = rm.evaluate({"action": "BUY", "consensus": 0.99})
        assert result["position_size"] <= 0.01

    def test_weak_consensus_rejected(self):
        rm = _make_risk_manager()
        # Very weak consensus should produce tiny position -> rejected
        result = rm.evaluate({"action": "BUY", "consensus": 0.001})
        assert result["approved"] is False

    def test_adaptive_sl_wider_on_low_consensus(self):
        rm = _make_risk_manager()
        low = rm.evaluate({"action": "BUY", "consensus": 0.30})
        high = rm.evaluate({"action": "BUY", "consensus": 0.90})
        # Lower consensus -> wider stop loss (more conservative)
        assert low["stop_loss"] > high["stop_loss"]

    def test_adaptive_tp_wider_on_high_consensus(self):
        rm = _make_risk_manager()
        low = rm.evaluate({"action": "BUY", "consensus": 0.30})
        high = rm.evaluate({"action": "BUY", "consensus": 0.90})
        # Higher consensus -> wider take profit (more ambitious)
        assert high["take_profit"] > low["take_profit"]
