"""
SCRIPT NAME: walk_forward.py
====================================
Execution Location: market-hawk-mvp/ml/
Purpose: Anchored Walk-Forward Optimization for backtesting strategies.
         Divides data into rolling train/test windows, backtests each,
         and aggregates out-of-sample performance.
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-07

Provides:
    - WalkForwardOptimizer: anchored walk-forward with BacktestEngine integration
    - WalkForwardResult / WindowResult: structured results with stability scoring

Usage:
    from backtesting.engine import BacktestEngine
    from backtesting.strategies import MACrossover

    optimizer = WalkForwardOptimizer(n_windows=5, train_pct=0.6, step_pct=0.1)
    result = optimizer.run(df, MACrossover(), symbol="AAPL", timeframe="1h")
    print(result.aggregate_sharpe, result.stability_score, result.is_stable)
"""

import copy
import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.strategies import StrategyBase

logger = logging.getLogger("market_hawk.ml.walk_forward")


# ============================================================
# RESULT DATA STRUCTURES
# ============================================================

@dataclass
class WindowResult:
    """Result of a single walk-forward window."""
    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_bars: int
    test_bars: int
    # Out-of-sample metrics (test period)
    sharpe: float
    return_pct: float
    win_rate: float
    total_trades: int
    max_drawdown_pct: float
    profit_factor: float
    # In-sample metrics (train period) for overfit comparison
    train_sharpe: float = 0.0
    train_return_pct: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward optimization results."""
    strategy_name: str
    symbol: str
    timeframe: str
    n_windows: int
    total_bars: int
    window_results: List[WindowResult] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def aggregate_sharpe(self) -> float:
        """Mean out-of-sample Sharpe across windows."""
        sharpes = [w.sharpe for w in self.window_results]
        return float(np.mean(sharpes)) if sharpes else 0.0

    @property
    def aggregate_return(self) -> float:
        """Mean out-of-sample return across windows."""
        returns = [w.return_pct for w in self.window_results]
        return float(np.mean(returns)) if returns else 0.0

    @property
    def aggregate_win_rate(self) -> float:
        """Mean out-of-sample win rate across windows."""
        rates = [w.win_rate for w in self.window_results]
        return float(np.mean(rates)) if rates else 0.0

    @property
    def stability_score(self) -> float:
        """1 - coefficient_of_variation of out-of-sample returns.

        1.0 = perfectly stable, 0.0 = high variance, negative = very unstable.
        """
        returns = [w.return_pct for w in self.window_results]
        if len(returns) < 2:
            return 0.0
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        if mean_r == 0:
            return 0.0
        cv = abs(std_r / mean_r)
        return float(max(1.0 - cv, -1.0))

    @property
    def is_stable(self) -> bool:
        """True if stability_score > 0.5 (returns are relatively consistent)."""
        return self.stability_score > 0.5

    @property
    def overfit_score(self) -> float:
        """Mean ratio of in-sample Sharpe to out-of-sample Sharpe.

        > 2.0 suggests heavy overfitting.
        """
        ratios = []
        for w in self.window_results:
            if w.sharpe != 0:
                ratios.append(w.train_sharpe / w.sharpe if w.sharpe != 0 else float("inf"))
        if not ratios:
            return 0.0
        # Cap infinite values
        ratios = [min(r, 10.0) for r in ratios]
        return float(np.mean(ratios))

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Walk-Forward: {self.strategy_name} on {self.symbol} ({self.timeframe})",
            f"  Windows: {self.n_windows} | Total bars: {self.total_bars} | Time: {self.elapsed_sec:.1f}s",
            f"  OOS Sharpe:  {self.aggregate_sharpe:+.3f}",
            f"  OOS Return:  {self.aggregate_return:+.4%}",
            f"  OOS WinRate: {self.aggregate_win_rate:.1%}",
            f"  Stability:   {self.stability_score:.3f}"
            f" ({'STABLE' if self.is_stable else 'UNSTABLE'})",
            f"  Overfit:     {self.overfit_score:.2f}"
            f" ({'WARNING' if self.overfit_score > 2.0 else 'OK'})",
            "",
        ]
        for w in self.window_results:
            lines.append(
                f"  Window {w.window_idx}: "
                f"train[{w.train_start}..{w.train_end}] ({w.train_bars}) "
                f"-> test[{w.test_start}..{w.test_end}] ({w.test_bars}) | "
                f"Sharpe={w.sharpe:+.3f} Ret={w.return_pct:+.4%} "
                f"WR={w.win_rate:.1%} Trades={w.total_trades} "
                f"DD={w.max_drawdown_pct:.2%}"
            )

        return "\n".join(lines)


# ============================================================
# WALK-FORWARD OPTIMIZER
# ============================================================

class WalkForwardOptimizer:
    """Anchored Walk-Forward Optimization using BacktestEngine.

    Divides the data into overlapping windows:
      - Training set starts at the beginning and grows (anchored)
      - Test set is the next segment after training
      - Window advances by step_pct of total data each iteration

    For each window:
      1. Copy the strategy (fresh state)
      2. Backtest on training period (in-sample metrics)
      3. Backtest on test period (out-of-sample metrics)
      4. Record both for overfit comparison

    Args:
        n_windows: Number of walk-forward windows (default 5).
        train_pct: Fraction of data for the first training window (default 0.6).
        step_pct: Fraction of data to advance per window (default 0.1).
        initial_capital: Starting capital for each window's backtest.
        commission_pct: Commission per trade.
        slippage_pct: Slippage per trade.
    """

    def __init__(self,
                 n_windows: int = 5,
                 train_pct: float = 0.6,
                 step_pct: float = 0.1,
                 initial_capital: float = 100_000.0,
                 commission_pct: float = 0.001,
                 slippage_pct: float = 0.0005) -> None:
        if n_windows < 2:
            raise ValueError("n_windows must be >= 2")
        if not (0.1 <= train_pct <= 0.9):
            raise ValueError("train_pct must be between 0.1 and 0.9")
        if not (0.01 <= step_pct <= 0.5):
            raise ValueError("step_pct must be between 0.01 and 0.5")

        self.n_windows = n_windows
        self.train_pct = train_pct
        self.step_pct = step_pct
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(self,
            df: pd.DataFrame,
            strategy: StrategyBase,
            symbol: str = "UNKNOWN",
            timeframe: str = "1h") -> WalkForwardResult:
        """Execute anchored walk-forward optimization.

        Args:
            df: Historical OHLCV DataFrame with DatetimeIndex.
            strategy: Strategy instance (deep-copied for each window).
            symbol: Symbol name for reporting.
            timeframe: Timeframe string.

        Returns:
            WalkForwardResult with per-window and aggregate metrics.
        """
        t0 = time.time()
        n = len(df)

        train_size_0 = int(n * self.train_pct)
        step_size = max(1, int(n * self.step_pct))

        logger.info("=" * 60)
        logger.info("WALK-FORWARD: %s on %s (%s)", strategy.name, symbol, timeframe)
        logger.info("  Total bars: %d | Windows: %d | Train0: %d | Step: %d",
                     n, self.n_windows, train_size_0, step_size)
        logger.info("=" * 60)

        window_results: List[WindowResult] = []

        for win_idx in range(self.n_windows):
            train_end = train_size_0 + win_idx * step_size
            test_start = train_end
            test_end = min(test_start + step_size, n)

            if train_end >= n or test_start >= n or test_end <= test_start:
                logger.info("  Window %d: not enough data, stopping", win_idx)
                break

            df_train = df.iloc[:train_end].copy()
            df_test = df.iloc[test_start:test_end].copy()

            if len(df_test) < 10:
                logger.info("  Window %d: test set too small (%d bars), skipping",
                            win_idx, len(df_test))
                continue

            logger.info("  Window %d/%d: train[0:%d] (%d bars) -> test[%d:%d] (%d bars)",
                         win_idx + 1, self.n_windows,
                         train_end, len(df_train),
                         test_start, test_end, len(df_test))

            # Fresh engine and strategy copy for each window
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission_pct=self.commission_pct,
                slippage_pct=self.slippage_pct,
                max_drawdown_pct=1.0,  # Disable circuit breaker for WF analysis
            )

            # In-sample backtest (training period)
            strat_train = copy.deepcopy(strategy)
            train_result = engine.run(df_train, strat_train, symbol, timeframe)

            # Out-of-sample backtest (test period)
            strat_test = copy.deepcopy(strategy)
            engine_test = BacktestEngine(
                initial_capital=self.initial_capital,
                commission_pct=self.commission_pct,
                slippage_pct=self.slippage_pct,
                max_drawdown_pct=1.0,
            )
            test_result = engine_test.run(df_test, strat_test, symbol, timeframe)

            # Date labels
            def _date_label(dataframe: pd.DataFrame, pos: int) -> str:
                idx_val = dataframe.index[pos]
                if hasattr(idx_val, "strftime"):
                    return idx_val.strftime("%Y-%m-%d")
                return str(idx_val)

            wr = WindowResult(
                window_idx=win_idx,
                train_start=_date_label(df_train, 0),
                train_end=_date_label(df_train, -1),
                test_start=_date_label(df_test, 0),
                test_end=_date_label(df_test, -1),
                train_bars=len(df_train),
                test_bars=len(df_test),
                sharpe=test_result.sharpe_ratio,
                return_pct=test_result.total_return_pct,
                win_rate=test_result.win_rate,
                total_trades=test_result.total_trades,
                max_drawdown_pct=test_result.max_drawdown_pct,
                profit_factor=test_result.profit_factor,
                train_sharpe=train_result.sharpe_ratio,
                train_return_pct=train_result.total_return_pct,
            )
            window_results.append(wr)

            logger.info("    OOS: Sharpe=%+.3f Return=%+.4f%% Trades=%d WR=%.1f%%",
                         test_result.sharpe_ratio,
                         test_result.total_return_pct * 100,
                         test_result.total_trades,
                         test_result.win_rate * 100)

            gc.collect()

        elapsed = time.time() - t0

        result = WalkForwardResult(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=timeframe,
            n_windows=len(window_results),
            total_bars=n,
            window_results=window_results,
            elapsed_sec=elapsed,
        )

        logger.info("=" * 60)
        logger.info("WALK-FORWARD COMPLETE in %.1fs", elapsed)
        logger.info("  OOS Sharpe: %+.3f | OOS Return: %+.4f%% | Stability: %.3f (%s)",
                     result.aggregate_sharpe,
                     result.aggregate_return * 100,
                     result.stability_score,
                     "STABLE" if result.is_stable else "UNSTABLE")
        if result.overfit_score > 2.0:
            logger.warning("  OVERFITTING WARNING: overfit_score=%.2f", result.overfit_score)
        logger.info("=" * 60)

        return result


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for noisy in ["market_hawk.backtest.engine"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from backtesting.strategies import MACrossover, RSIMeanReversion

    print("\n" + "=" * 60)
    print("WALK-FORWARD OPTIMIZATION — Demo")
    print("=" * 60)

    # Generate synthetic trending data
    np.random.seed(42)
    n = 2000
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    returns = np.random.normal(0.0001, 0.008, n)
    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.004, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.004, n)))
    open_p = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1000, 50000, n).astype(float)

    df = pd.DataFrame({
        "Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)

    # Walk-forward with MA Crossover
    print("\n--- MA Crossover Walk-Forward ---")
    optimizer = WalkForwardOptimizer(n_windows=5, train_pct=0.5, step_pct=0.1)
    result = optimizer.run(df, MACrossover(fast_period=10, slow_period=30),
                           symbol="SYNTH", timeframe="1h")
    print("\n" + result.summary())

    # Walk-forward with RSI Mean Reversion
    print("\n--- RSI Mean Reversion Walk-Forward ---")
    result2 = optimizer.run(df, RSIMeanReversion(period=14),
                            symbol="SYNTH", timeframe="1h")
    print("\n" + result2.summary())

    print("\nDone.")
