#!/usr/bin/env python3
"""
SCRIPT NAME: engine.py
====================================
Execution Location: K:\\_DEV_MVP_2026\\Market_Hawk_3\\backtesting\\
Purpose: Core backtesting engine — runs strategies on historical data,
         tracks positions, calculates P&L, generates comprehensive reports.
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01

All money/price calculations use decimal.Decimal for precision.
Ratios, percentages, and statistical metrics remain float.
"""

import gc
import json
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from backtesting.strategies import StrategyBase, TradeSignal, Signal
from backtesting.data_loader import HistoricalDataLoader
from config.settings import RISK_CONFIG

logger = logging.getLogger("market_hawk.backtest.engine")

# Decimal helpers (same pattern as trading/paper_trader.py)
D = Decimal
ZERO = D("0")
ONE = D("1")


def _to_decimal(value: Any) -> Decimal:
    """Convert a numeric value to Decimal safely via string representation."""
    if isinstance(value, Decimal):
        return value
    if value is None:
        return ZERO
    return D(str(value))


# ============================================================
# COMMISSION MODEL
# ============================================================

class CommissionModel:
    """Configurable commission model for backtesting.

    Supports three modes:
        - "percentage": commission = trade_value * pct  (default)
        - "per_share": commission = quantity * per_share_cost
        - "tiered": per_share with volume tiers

    All modes enforce a minimum commission floor.
    """

    def __init__(self,
                 mode: str = "percentage",
                 commission_pct: float = 0.001,
                 per_share: float = 0.005,
                 min_commission: float = 1.0,
                 tiers: Optional[List[Tuple[int, float]]] = None):
        """
        Args:
            mode: "percentage", "per_share", or "tiered".
            commission_pct: Rate for percentage mode (0.001 = 0.1%).
            per_share: Cost per share for per_share/tiered modes.
            min_commission: Minimum commission per trade ($1 default).
            tiers: For tiered mode, list of (qty_threshold, per_share_rate)
                   sorted ascending. E.g. [(0, 0.005), (500, 0.004), (1000, 0.003)].
        """
        self.mode = mode
        self.commission_pct = _to_decimal(commission_pct)
        self.per_share = _to_decimal(per_share)
        self.min_commission = _to_decimal(min_commission)
        self.tiers = tiers or [(0, 0.005), (500, 0.004), (1000, 0.003)]

    def calculate(self, trade_value: Decimal, quantity: Decimal) -> Decimal:
        """Calculate commission for a trade.

        Args:
            trade_value: Total dollar value of the trade.
            quantity: Number of shares/units.

        Returns:
            Commission amount (Decimal), at least min_commission.
        """
        if self.mode == "per_share":
            commission = quantity * self.per_share
        elif self.mode == "tiered":
            commission = self._tiered_commission(quantity)
        else:
            commission = trade_value * self.commission_pct

        return max(commission, self.min_commission)

    def _tiered_commission(self, quantity: Decimal) -> Decimal:
        """Calculate tiered per-share commission."""
        qty_float = float(quantity)
        rate = _to_decimal(self.tiers[0][1])
        for threshold, tier_rate in self.tiers:
            if qty_float >= threshold:
                rate = _to_decimal(tier_rate)
        return quantity * rate


# ============================================================
# SLIPPAGE MODEL
# ============================================================

class SlippageModel:
    """Variable slippage model based on volume impact.

    slippage = base_slippage * (1 + volume_impact)
    volume_impact = order_size / avg_daily_volume

    Clamped to [min_slippage, max_slippage].
    Falls back to base_slippage when volume data is unavailable.
    """

    def __init__(self,
                 base_slippage_pct: float = 0.0005,
                 min_slippage_pct: float = 0.0001,
                 max_slippage_pct: float = 0.005):
        """
        Args:
            base_slippage_pct: Base slippage fraction (0.0005 = 0.05%).
            min_slippage_pct: Floor (0.0001 = 0.01%).
            max_slippage_pct: Cap (0.005 = 0.5%).
        """
        self.base = _to_decimal(base_slippage_pct)
        self.min_slip = _to_decimal(min_slippage_pct)
        self.max_slip = _to_decimal(max_slippage_pct)

    def calculate(self, order_value: Decimal,
                  avg_daily_volume_value: Optional[Decimal] = None) -> Decimal:
        """Calculate slippage fraction for a trade.

        Args:
            order_value: Dollar value of the order.
            avg_daily_volume_value: Average daily volume in dollar terms.
                                   If None, returns base slippage.

        Returns:
            Slippage fraction (Decimal), clamped to [min, max].
        """
        if avg_daily_volume_value is None or avg_daily_volume_value <= ZERO:
            return max(self.min_slip, min(self.base, self.max_slip))

        volume_impact = order_value / avg_daily_volume_value
        slippage = self.base * (ONE + volume_impact)
        return max(self.min_slip, min(slippage, self.max_slip))


# ============================================================
# TRADE RECORD
# ============================================================

@dataclass
class BacktestTrade:
    """A completed backtesting trade."""
    trade_id: int
    symbol: str
    side: str              # LONG or SHORT
    entry_idx: int
    exit_idx: int
    entry_time: str
    exit_time: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    pnl_pct: float         # Ratio, not money
    commission: Decimal
    exit_reason: str       # stop_loss, take_profit, signal_reversal, end_of_data
    hold_bars: int
    signal_confidence: float
    signal_reason: str


# ============================================================
# BACKTEST RESULTS
# ============================================================

@dataclass
class BacktestResult:
    """Complete backtesting results."""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    total_bars: int
    initial_capital: float      # float for JSON/reporting compatibility
    final_equity: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_hold_bars: float
    total_commission: float
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    execution_time_sec: float = 0.0

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Truncate curves for JSON serialization
        if len(d["equity_curve"]) > 5000:
            step = len(d["equity_curve"]) // 5000
            d["equity_curve"] = d["equity_curve"][::step]
            d["drawdown_curve"] = d["drawdown_curve"][::step]
        return d


# ============================================================
# BACKTEST ENGINE
# ============================================================

class BacktestEngine:
    """
    Enterprise-Grade Backtesting Engine for Market Hawk 3.

    Features:
        - Pluggable strategies via StrategyBase interface
        - Commission modeling (default 0.1% per trade)
        - Slippage simulation (configurable)
        - Position sizing (fixed fraction or Kelly)
        - Stop-loss / take-profit execution
        - Detailed trade log and equity curve
        - Performance metrics: Sharpe, Sortino, Calmar, max DD
        - Monthly return breakdown
        - Progress bar with ETA

    Usage:
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(
            df=historical_data,
            strategy=MACrossover(fast=20, slow=50),
            symbol="BTCUSDT",
            timeframe="1h"
        )
        result.sharpe_ratio, result.win_rate, etc.
    """

    def __init__(self,
                 initial_capital: float = 100_000.0,
                 commission_pct: float = 0.001,    # 0.1% per trade
                 slippage_pct: float = 0.0005,     # 0.05% slippage
                 max_position_pct: float = 0.10,   # Max 10% per position
                 allow_short: bool = True,
                 allow_pyramiding: bool = False,
                 max_drawdown_pct: Optional[float] = None,
                 commission_model: Optional[CommissionModel] = None,
                 slippage_model: Optional[SlippageModel] = None):
        """
        Args:
            initial_capital: Starting cash
            commission_pct: Commission as fraction (0.001 = 0.1%).
                Used only if commission_model is None (backward compatible).
            slippage_pct: Slippage as fraction.
                Used only if slippage_model is None (backward compatible).
            max_position_pct: Max fraction of equity per trade
            allow_short: Enable short selling
            allow_pyramiding: Allow multiple entries in same direction
            max_drawdown_pct: Max drawdown before circuit breaker (default from RiskConfig)
            commission_model: CommissionModel instance. Overrides commission_pct.
            slippage_model: SlippageModel instance. Overrides slippage_pct.
        """
        self.initial_capital = _to_decimal(initial_capital)
        self.commission_pct = _to_decimal(commission_pct)
        self.slippage_pct = _to_decimal(slippage_pct)
        self.max_position_pct = _to_decimal(max_position_pct)
        self.allow_short = allow_short
        self.allow_pyramiding = allow_pyramiding
        self.max_drawdown_pct = max_drawdown_pct if max_drawdown_pct is not None else RISK_CONFIG.max_drawdown_pct

        self._commission_model = commission_model or CommissionModel(
            mode="percentage", commission_pct=commission_pct,
        )
        self._slippage_model = slippage_model or SlippageModel(
            base_slippage_pct=slippage_pct,
        )

        # State
        self._cash: Decimal = self.initial_capital
        self._position: Optional[Dict] = None  # Current open position
        self._trades: List[BacktestTrade] = []
        self._trade_counter = 0
        self._equity_curve: List[float] = []
        self._halted: bool = False

    # ============================================================
    # MAIN RUN METHOD
    # ============================================================

    def run(self, df: pd.DataFrame, strategy: StrategyBase,
            symbol: str = "UNKNOWN", timeframe: str = "1h",
            progress_callback=None) -> BacktestResult:
        """
        Execute a full backtest.

        Args:
            df: Historical OHLCV DataFrame with DatetimeIndex
            strategy: Strategy instance implementing StrategyBase
            symbol: Symbol name for reporting
            timeframe: Timeframe for reporting
            progress_callback: Optional callable(pct, msg) for progress updates

        Returns:
            BacktestResult with all metrics
        """
        t0 = time.time()

        logger.info("=" * 60)
        logger.info("BACKTEST: %s on %s (%s) — %d bars",
                     strategy.name, symbol, timeframe, len(df))
        logger.info("=" * 60)

        # Reset state
        self._cash = self.initial_capital
        self._position = None
        self._trades = []
        self._trade_counter = 0
        self._equity_curve = []
        self._halted = False
        _peak_equity = float(self.initial_capital)

        # Pre-compute indicators
        logger.info("Pre-computing indicators...")
        df = strategy.on_init(df.copy())

        total_bars = len(df)
        report_interval = max(total_bars // 20, 100)

        # ---- Main loop ----
        for idx in range(total_bars):
            current_price = _to_decimal(df["Close"].iloc[idx])

            # 1. Check SL/TP on open position
            if self._position is not None:
                exit_signal = self._check_exit(df, idx, current_price)
                if exit_signal:
                    self._close_position(df, idx, current_price, exit_signal)

            # 2. Generate strategy signal (skip if halted by circuit breaker)
            if self._halted:
                equity = self._calculate_equity(current_price)
                equity_f = float(equity)
                self._equity_curve.append(equity_f)
                if equity_f > _peak_equity:
                    _peak_equity = equity_f
                if idx % report_interval == 0 and idx > 0:
                    pct = idx / total_bars * 100
                    logger.info("  [%5.1f%%] bar %s/%s — HALTED (circuit breaker)",
                                pct, f"{idx:,}", f"{total_bars:,}")
                if idx % 100_000 == 0 and idx > 0:
                    gc.collect()
                continue

            signal = strategy.generate_signal(df, idx)

            # 3. Process signal
            if signal.signal == Signal.BUY and self._position is None:
                self._open_position(df, idx, current_price, "LONG", signal)
            elif signal.signal == Signal.SELL:
                if self._position is not None and self._position["side"] == "LONG":
                    self._close_position(df, idx, current_price, "signal_reversal")
                if self._position is None and self.allow_short:
                    self._open_position(df, idx, current_price, "SHORT", signal)
            elif signal.signal == Signal.BUY and self._position is not None:
                if self._position["side"] == "SHORT":
                    self._close_position(df, idx, current_price, "signal_reversal")
                    self._open_position(df, idx, current_price, "LONG", signal)

            # 4. Track equity (as float for numpy-based metrics later)
            equity = self._calculate_equity(current_price)
            equity_f = float(equity)
            self._equity_curve.append(equity_f)

            # 4b. Max drawdown circuit breaker
            if equity_f > _peak_equity:
                _peak_equity = equity_f
            if _peak_equity > 0:
                current_dd = (_peak_equity - equity_f) / _peak_equity
                if current_dd > self.max_drawdown_pct and not self._halted:
                    logger.warning(
                        "CIRCUIT BREAKER: max drawdown exceeded (%.2f%% > %.2f%%) at bar %d",
                        current_dd * 100, self.max_drawdown_pct * 100, idx)
                    if self._position is not None:
                        self._close_position(df, idx, current_price, "circuit_breaker")
                        self._equity_curve[-1] = float(self._calculate_equity(current_price))
                    self._halted = True

            # 5. Progress reporting
            if idx % report_interval == 0 and idx > 0:
                pct = idx / total_bars * 100
                elapsed = time.time() - t0
                eta = (elapsed / idx) * (total_bars - idx)
                msg = (f"  [{pct:5.1f}%] bar {idx:,}/{total_bars:,} | "
                       f"equity: ${equity:,.0f} | trades: {len(self._trades)} | "
                       f"ETA: {eta:.0f}s")
                logger.info(msg)
                if progress_callback:
                    progress_callback(pct, msg)

            # 6. Memory management (every 100K bars)
            if idx % 100_000 == 0 and idx > 0:
                gc.collect()

        # Close any remaining position at end
        if self._position is not None:
            last_price = _to_decimal(df["Close"].iloc[-1])
            self._close_position(df, len(df) - 1, last_price, "end_of_data")
            self._equity_curve[-1] = float(self._calculate_equity(last_price))

        # Calculate metrics
        elapsed = time.time() - t0
        result = self._compute_metrics(df, strategy.name, symbol, timeframe, elapsed)

        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE in %.1fs", elapsed)
        logger.info("  Return: %+.2f%% | Win rate: %.1f%% | Sharpe: %.2f | Max DD: %.2f%%",
                     result.total_return_pct * 100,
                     result.win_rate * 100,
                     result.sharpe_ratio,
                     result.max_drawdown_pct * 100)
        logger.info("=" * 60)

        gc.collect()
        return result

    # ============================================================
    # MULTI-SYMBOL RUN
    # ============================================================

    def run_portfolio(self, data: Dict[str, pd.DataFrame],
                      strategy: StrategyBase,
                      timeframe: str = "1h") -> Dict[str, BacktestResult]:
        """
        Run backtest across multiple symbols.

        Args:
            data: Dict of symbol -> DataFrame
            strategy: Strategy to apply to all symbols

        Returns:
            Dict of symbol -> BacktestResult
        """
        results = {}
        total = len(data)

        for i, (symbol, df) in enumerate(data.items(), 1):
            logger.info("[%d/%d] Backtesting %s...", i, total, symbol)
            try:
                # Fresh strategy instance for each symbol
                import copy
                strat_copy = copy.deepcopy(strategy)
                result = self.run(df, strat_copy, symbol, timeframe)
                results[symbol] = result
            except Exception:
                logger.exception("Failed on %s", symbol)
            gc.collect()

        return results

    # ============================================================
    # POSITION MANAGEMENT
    # ============================================================

    def _open_position(self, df: pd.DataFrame, idx: int,
                       price: Decimal, side: str, signal: TradeSignal) -> None:
        """Open a new position."""
        # Position sizing
        equity = self._calculate_equity(price)
        pos_size = _to_decimal(signal.position_size)
        max_pos = self.max_position_pct
        trade_value = equity * min(pos_size, max_pos)

        # Calculate slippage (variable, volume-aware)
        avg_vol_value = self._estimate_avg_volume_value(df, idx, price)
        slippage_pct = self._slippage_model.calculate(trade_value, avg_vol_value)

        if side == "LONG":
            fill_price = price * (ONE + slippage_pct)
        else:
            fill_price = price * (ONE - slippage_pct)

        # Calculate commission (model-based)
        quantity = trade_value / fill_price if fill_price > ZERO else ZERO
        commission = self._commission_model.calculate(trade_value, quantity)
        trade_value -= commission
        quantity = trade_value / fill_price if fill_price > ZERO else ZERO

        if quantity <= ZERO:
            return

        # Deduct cash for LONG
        if side == "LONG":
            self._cash -= (trade_value + commission)

        self._position = {
            "side": side,
            "entry_price": fill_price,
            "quantity": quantity,
            "entry_idx": idx,
            "entry_time": str(df.index[idx]) if hasattr(df.index[idx], 'strftime') else str(idx),
            "stop_loss_pct": signal.stop_loss_pct,
            "take_profit_pct": signal.take_profit_pct,
            "commission_entry": commission,
            "signal_confidence": signal.confidence,
            "signal_reason": signal.reason,
        }

    def _close_position(self, df: pd.DataFrame, idx: int,
                        price: Decimal, reason: str) -> None:
        """Close the current position."""
        if self._position is None:
            return

        pos = self._position
        side = pos["side"]

        # Apply slippage (variable, volume-aware)
        exit_value_est = price * pos["quantity"]
        avg_vol_value = self._estimate_avg_volume_value(df, idx, price)
        slippage_pct = self._slippage_model.calculate(exit_value_est, avg_vol_value)

        if side == "LONG":
            fill_price = price * (ONE - slippage_pct)
        else:
            fill_price = price * (ONE + slippage_pct)

        # Calculate P&L
        if side == "LONG":
            pnl = (fill_price - pos["entry_price"]) * pos["quantity"]
        else:
            pnl = (pos["entry_price"] - fill_price) * pos["quantity"]

        # Commission on exit (model-based)
        exit_value = fill_price * pos["quantity"]
        commission_exit = self._commission_model.calculate(exit_value, pos["quantity"])
        pnl -= commission_exit
        total_commission = pos["commission_entry"] + commission_exit

        entry_value = pos["entry_price"] * pos["quantity"]
        pnl_pct = float(pnl / entry_value) if entry_value > ZERO else 0.0

        # Return cash
        if side == "LONG":
            self._cash += exit_value - commission_exit
        else:
            self._cash += pnl - commission_exit  # For short, just add the P&L

        self._trade_counter += 1
        trade = BacktestTrade(
            trade_id=self._trade_counter,
            symbol="",  # Set by run()
            side=side,
            entry_idx=pos["entry_idx"],
            exit_idx=idx,
            entry_time=pos["entry_time"],
            exit_time=str(df.index[idx]) if hasattr(df.index[idx], 'strftime') else str(idx),
            entry_price=pos["entry_price"],
            exit_price=fill_price,
            quantity=pos["quantity"],
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=total_commission,
            exit_reason=reason,
            hold_bars=idx - pos["entry_idx"],
            signal_confidence=pos["signal_confidence"],
            signal_reason=pos["signal_reason"],
        )
        self._trades.append(trade)
        self._position = None

    def _check_exit(self, df: pd.DataFrame, idx: int,
                    price: Decimal) -> Optional[str]:
        """Check if position should be closed (SL/TP)."""
        if self._position is None:
            return None

        pos = self._position
        entry = pos["entry_price"]
        sl = _to_decimal(pos["stop_loss_pct"])
        tp = _to_decimal(pos["take_profit_pct"])

        if pos["side"] == "LONG":
            if price <= entry * (ONE - sl):
                return "stop_loss"
            if price >= entry * (ONE + tp):
                return "take_profit"
        else:  # SHORT
            if price >= entry * (ONE + sl):
                return "stop_loss"
            if price <= entry * (ONE - tp):
                return "take_profit"

        return None

    @staticmethod
    def _estimate_avg_volume_value(df: pd.DataFrame, idx: int,
                                   current_price: Decimal) -> Optional[Decimal]:
        """Estimate average daily dollar volume from recent bars.

        Uses a 20-bar rolling average of Volume * Close.
        Returns None if volume data is unavailable.
        """
        if "Volume" not in df.columns:
            return None
        lookback = min(20, idx)
        if lookback < 1:
            return None
        vol_slice = df["Volume"].iloc[max(0, idx - lookback):idx]
        if vol_slice.empty or vol_slice.sum() == 0:
            return None
        avg_volume = float(vol_slice.mean())
        return _to_decimal(avg_volume) * current_price

    def _calculate_equity(self, current_price: Decimal) -> Decimal:
        """Calculate current total equity."""
        equity = self._cash
        if self._position is not None:
            pos = self._position
            if pos["side"] == "LONG":
                equity += current_price * pos["quantity"]
            else:
                equity += (pos["entry_price"] - current_price) * pos["quantity"]
        return equity

    # ============================================================
    # METRICS COMPUTATION
    # ============================================================

    def _compute_metrics(self, df: pd.DataFrame, strategy_name: str,
                         symbol: str, timeframe: str,
                         elapsed: float) -> BacktestResult:
        """Compute all performance metrics from trades and equity curve."""

        equity = np.array(self._equity_curve) if self._equity_curve else np.array([float(self.initial_capital)])
        final_equity = equity[-1] if len(equity) > 0 else float(self.initial_capital)

        # Basic stats
        total_trades = len(self._trades)
        wins = [t for t in self._trades if t.pnl > ZERO]
        losses = [t for t in self._trades if t.pnl <= ZERO]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0

        # Profit factor
        gross_profit = sum((t.pnl for t in wins), ZERO) if wins else ZERO
        gross_loss = abs(sum((t.pnl for t in losses), ZERO)) if losses else ONE
        profit_factor = float(gross_profit / gross_loss) if gross_loss > ZERO else float("inf")

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Max drawdown duration
        dd_duration = 0
        max_dd_duration = 0
        for i in range(len(drawdown)):
            if drawdown[i] < 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        # Returns series for Sharpe/Sortino
        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
        returns = returns[np.isfinite(returns)]

        # Sharpe ratio — auto-detect actual data frequency from index
        bars_per_year_map = {"1m": 252 * 390, "5m": 252 * 78, "15m": 252 * 26,
                             "1h": 252 * 6.5, "4h": 252 * 1.625,
                             "1d": 252, "1wk": 52, "1mo": 12}

        # Auto-detect from median bar spacing if DatetimeIndex
        detected_tf = timeframe
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 10:
            median_gap = df.index.to_series().diff().median()
            if median_gap is not None and pd.notna(median_gap):
                gap_hours = median_gap.total_seconds() / 3600
                if gap_hours < 0.1:       detected_tf = "1m"
                elif gap_hours < 0.2:     detected_tf = "5m"
                elif gap_hours < 0.5:     detected_tf = "15m"
                elif gap_hours < 2:       detected_tf = "1h"
                elif gap_hours < 8:       detected_tf = "4h"
                elif gap_hours < 48:      detected_tf = "1d"
                elif gap_hours < 240:     detected_tf = "1wk"
                else:                     detected_tf = "1mo"
                if detected_tf != timeframe:
                    logger.info("Auto-detected timeframe: %s (requested: %s)",
                                detected_tf, timeframe)

        annual_factor = bars_per_year_map.get(detected_tf, 252)
        mean_r = np.mean(returns) if len(returns) > 0 else 0
        std_r = np.std(returns) if len(returns) > 0 else 1
        sharpe = (mean_r / std_r) * np.sqrt(annual_factor) if std_r > 0 else 0

        # Sortino (downside deviation only)
        neg_returns = returns[returns < 0]
        downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 1
        sortino = (mean_r / downside_std) * np.sqrt(annual_factor) if downside_std > 0 else 0

        # Calmar (return / max DD)
        initial_f = float(self.initial_capital)
        total_return = (final_equity - initial_f) / initial_f
        calmar = total_return / max_dd if max_dd > 0 else 0

        # Average hold time
        avg_hold = np.mean([t.hold_bars for t in self._trades]) if self._trades else 0

        # Total commission
        total_comm = sum((t.commission for t in self._trades), ZERO)

        # Monthly returns (if DatetimeIndex)
        monthly = {}
        if isinstance(df.index, pd.DatetimeIndex) and len(equity) > 0:
            eq_series = pd.Series(equity, index=df.index[:len(equity)])
            monthly_eq = eq_series.resample("ME").last()
            monthly_ret = monthly_eq.pct_change().dropna()
            for dt, ret in monthly_ret.items():
                monthly[dt.strftime("%Y-%m")] = round(float(ret), 6)

        # Build result
        start_date = str(df.index[0]) if len(df) > 0 else ""
        end_date = str(df.index[-1]) if len(df) > 0 else ""

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            total_bars=len(df),
            initial_capital=initial_f,
            final_equity=float(final_equity),
            total_return_pct=float(total_return),
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=float(win_rate),
            avg_win_pct=float(avg_win),
            avg_loss_pct=float(avg_loss),
            profit_factor=float(min(profit_factor, 999)),
            max_drawdown_pct=float(max_dd),
            max_drawdown_duration_bars=max_dd_duration,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            avg_hold_bars=float(avg_hold),
            total_commission=float(total_comm),
            equity_curve=equity.tolist(),
            drawdown_curve=drawdown.tolist(),
            trades=[asdict(t) for t in self._trades],
            monthly_returns=monthly,
            execution_time_sec=elapsed,
        )

    # ============================================================
    # CONVENIENCE: Quick backtest with data loading
    # ============================================================

    def quick_test(self, symbol: str, strategy: StrategyBase,
                   timeframe: str = "1h",
                   start: Optional[str] = None,
                   end: Optional[str] = None) -> Optional[BacktestResult]:
        """
        Quick backtest: load data + run strategy in one call.

        Usage:
            engine = BacktestEngine()
            result = engine.quick_test("AAPL", MACrossover(), timeframe="1h")
        """
        loader = HistoricalDataLoader()
        df = loader.load(symbol, timeframe, start, end)
        if df is None or df.empty:
            logger.error("No data available for %s", symbol)
            return None
        return self.run(df, strategy, symbol, timeframe)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    for noisy in ["httpx", "httpcore", "urllib3", "yfinance", "peewee"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from backtesting.strategies import MACrossover, RSIMeanReversion, BollingerBreakout

    print("\n" + "=" * 60)
    print("BACKTEST ENGINE — Quick Test")
    print("=" * 60)

    engine = BacktestEngine(initial_capital=100_000)

    # Test with multiple strategies on AAPL
    strategies = [
        MACrossover(fast_period=20, slow_period=50),
        RSIMeanReversion(period=14),
        BollingerBreakout(period=20),
    ]

    for strat in strategies:
        print(f"\n--- {strat.name} ---")
        result = engine.quick_test("AAPL", strat, timeframe="1h")
        if result:
            print(f"  Return:    {result.total_return_pct:+.2%}")
            print(f"  Trades:    {result.total_trades}")
            print(f"  Win Rate:  {result.win_rate:.1%}")
            print(f"  Sharpe:    {result.sharpe_ratio:.2f}")
            print(f"  Max DD:    {result.max_drawdown_pct:.2%}")
            print(f"  P/F:       {result.profit_factor:.2f}")
        else:
            print("  No result")

    print("\nBacktest engine test complete")
