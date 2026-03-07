"""
Tests for trading/paper_trader.py

Covers:
    - Position dataclass: update_price, should_stop_loss, should_take_profit
    - Portfolio properties: equity, total_pnl, win_rate, max_drawdown
    - _to_decimal helper: conversion safety from float, str, int, None
    - ClosedTrade PnL calculations
"""

import pytest
from decimal import Decimal

D = Decimal
ZERO = D("0")


# ============================================================
# _to_decimal HELPER
# ============================================================

class TestToDecimal:
    """Test the _to_decimal conversion helper."""

    def test_from_float(self):
        from trading.paper_trader import _to_decimal
        result = _to_decimal(3.14)
        assert isinstance(result, Decimal)
        # Via str() avoids float imprecision
        assert result == D("3.14")

    def test_from_int(self):
        from trading.paper_trader import _to_decimal
        assert _to_decimal(42) == D("42")

    def test_from_string(self):
        from trading.paper_trader import _to_decimal
        assert _to_decimal("99.99") == D("99.99")

    def test_from_none(self):
        from trading.paper_trader import _to_decimal
        assert _to_decimal(None) == ZERO

    def test_from_decimal_passthrough(self):
        from trading.paper_trader import _to_decimal
        val = D("123.456")
        assert _to_decimal(val) is val

    def test_from_numpy_float(self):
        import numpy as np
        from trading.paper_trader import _to_decimal
        result = _to_decimal(np.float64(100.5))
        assert isinstance(result, Decimal)


# ============================================================
# POSITION
# ============================================================

class TestPosition:
    """Test Position dataclass methods."""

    def _make_position(self, side="LONG", entry_price="100.00",
                       quantity="10", stop_loss=0.02, take_profit=0.04):
        from trading.paper_trader import Position
        return Position(
            symbol="AAPL",
            side=side,
            entry_price=D(entry_price),
            quantity=D(quantity),
            entry_time="2025-01-01T10:00:00",
            stop_loss=stop_loss,
            take_profit=take_profit,
            consensus_at_entry=0.75,
            current_price=D(entry_price),
        )

    def test_update_price_long_profit(self):
        pos = self._make_position(side="LONG", entry_price="100.00", quantity="10")
        pos.update_price(D("110.00"))
        assert pos.unrealized_pnl == D("100.00")  # (110-100)*10
        assert pos.unrealized_pnl_pct == pytest.approx(0.10)

    def test_update_price_long_loss(self):
        pos = self._make_position(side="LONG", entry_price="100.00", quantity="10")
        pos.update_price(D("95.00"))
        assert pos.unrealized_pnl == D("-50.00")  # (95-100)*10
        assert pos.unrealized_pnl_pct == pytest.approx(-0.05)

    def test_update_price_short_profit(self):
        pos = self._make_position(side="SHORT", entry_price="100.00", quantity="10")
        pos.update_price(D("90.00"))
        assert pos.unrealized_pnl == D("100.00")  # (100-90)*10
        assert pos.unrealized_pnl_pct == pytest.approx(0.10)

    def test_should_stop_loss_long(self):
        pos = self._make_position(side="LONG", entry_price="100.00", stop_loss=0.02)
        # SL at 98.00 (2% below entry)
        assert pos.should_stop_loss(D("97.99")) is True
        assert pos.should_stop_loss(D("98.01")) is False

    def test_should_stop_loss_short(self):
        pos = self._make_position(side="SHORT", entry_price="100.00", stop_loss=0.02)
        # SL at 102.00 (2% above entry)
        assert pos.should_stop_loss(D("102.01")) is True
        assert pos.should_stop_loss(D("101.99")) is False

    def test_should_take_profit_long(self):
        pos = self._make_position(side="LONG", entry_price="100.00", take_profit=0.04)
        # TP at 104.00 (4% above entry)
        assert pos.should_take_profit(D("104.01")) is True
        assert pos.should_take_profit(D("103.99")) is False

    def test_should_take_profit_short(self):
        pos = self._make_position(side="SHORT", entry_price="100.00", take_profit=0.04)
        # TP at 96.00 (4% below entry)
        assert pos.should_take_profit(D("95.99")) is True
        assert pos.should_take_profit(D("96.01")) is False

    def test_zero_entry_price_no_crash(self):
        pos = self._make_position(entry_price="0")
        pos.update_price(D("10.00"))  # Should not raise ZeroDivisionError
        assert pos.unrealized_pnl_pct == 0.0


# ============================================================
# PORTFOLIO
# ============================================================

class TestPortfolio:
    """Test Portfolio properties and calculations."""

    def _make_portfolio(self):
        from trading.paper_trader import Portfolio
        return Portfolio()

    def test_initial_state(self):
        p = self._make_portfolio()
        assert p.cash == D("100000")
        assert p.equity == D("100000")
        assert p.total_pnl == ZERO
        assert p.total_pnl_pct == 0.0
        assert p.win_rate == 0.0
        assert p.max_drawdown == 0.0

    def test_equity_with_position(self):
        from trading.paper_trader import Portfolio, Position
        p = Portfolio(cash=D("90000"))
        p.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG", entry_price=D("100"),
            quantity=D("100"), entry_time="2025-01-01",
            stop_loss=0.02, take_profit=0.04, consensus_at_entry=0.7,
            current_price=D("105"), unrealized_pnl=D("500"),
        )
        assert p.equity == D("90500")

    def test_win_rate(self):
        from trading.paper_trader import Portfolio, ClosedTrade
        p = self._make_portfolio()
        p.closed_trades.append(ClosedTrade(
            symbol="AAPL", side="LONG", entry_price=D("100"),
            exit_price=D("110"), quantity=D("10"), entry_time="",
            exit_time="", pnl=D("100"), pnl_pct=0.10, exit_reason="take_profit",
        ))
        p.closed_trades.append(ClosedTrade(
            symbol="MSFT", side="LONG", entry_price=D("200"),
            exit_price=D("190"), quantity=D("5"), entry_time="",
            exit_time="", pnl=D("-50"), pnl_pct=-0.05, exit_reason="stop_loss",
        ))
        assert p.win_rate == pytest.approx(0.5)

    def test_realized_pnl(self):
        from trading.paper_trader import Portfolio, ClosedTrade
        p = self._make_portfolio()
        p.closed_trades.append(ClosedTrade(
            symbol="A", side="LONG", entry_price=D("10"),
            exit_price=D("15"), quantity=D("10"), entry_time="",
            exit_time="", pnl=D("50"), pnl_pct=0.50, exit_reason="tp",
        ))
        p.closed_trades.append(ClosedTrade(
            symbol="B", side="LONG", entry_price=D("20"),
            exit_price=D("18"), quantity=D("10"), entry_time="",
            exit_time="", pnl=D("-20"), pnl_pct=-0.10, exit_reason="sl",
        ))
        assert p.realized_pnl == D("30")

    def test_max_drawdown_at_peak(self):
        p = self._make_portfolio()
        # Cash == peak equity, no drawdown
        assert p.max_drawdown == 0.0

    def test_max_drawdown_below_peak(self):
        p = self._make_portfolio()
        p.peak_equity = D("110000")
        p.cash = D("99000")
        # DD = (110000 - 99000) / 110000 = 0.1
        assert p.max_drawdown == pytest.approx(0.1, abs=1e-6)
