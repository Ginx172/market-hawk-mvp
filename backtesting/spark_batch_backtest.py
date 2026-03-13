"""
Spark-based batch backtesting for Market Hawk MVP.

Distributes N-symbol × M-strategy combinations across Spark partitions
so that each CPU core runs independent ``BacktestEngine`` instances in
parallel.  The engine and strategy objects themselves remain pandas-based;
Spark is used exclusively for task distribution.

Hardware target: Intel i7-9700F (8 cores), 64 GB DDR4.
"""
import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

logger = logging.getLogger("market_hawk.spark.backtest")

# Strategy name → class mapping used by worker partitions
_STRATEGY_REGISTRY: Dict[str, str] = {
    "MACrossover": "backtesting.strategies.MACrossover",
    "RSIMeanReversion": "backtesting.strategies.RSIMeanReversion",
    "BollingerBreakout": "backtesting.strategies.BollingerBreakout",
}


def _import_class(dotted_path: str) -> Any:
    """Import a class from a dotted module path.

    Args:
        dotted_path: e.g. ``"backtesting.strategies.MACrossover"``.

    Returns:
        The class object.
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _run_single_backtest(
    symbol: str,
    strategy_name: str,
    strategy_kwargs: Dict[str, Any],
    initial_capital: float,
    commission_pct: float,
    slippage_pct: float,
    timeframe: str,
) -> Dict[str, Any]:
    """Run one backtest and return a result dictionary.

    Designed to be called inside a Spark partition so it must be
    self-contained (no captured closures with non-picklable objects).

    Args:
        symbol:           Ticker symbol (e.g. ``"AAPL"``).
        strategy_name:    Name key from ``_STRATEGY_REGISTRY``.
        strategy_kwargs:  Keyword arguments forwarded to the strategy
                          constructor.
        initial_capital:  Starting portfolio value in USD.
        commission_pct:   Commission rate (0.001 = 0.1 %).
        slippage_pct:     Slippage rate (0.0005 = 0.05 %).
        timeframe:        Data timeframe string (e.g. ``"1d"``).

    Returns:
        Dictionary with backtest metrics plus ``symbol`` and
        ``strategy_name`` keys.  On failure, returns a dict with
        ``error`` key set.
    """
    try:
        from backtesting.engine import BacktestEngine, CommissionModel
        from backtesting.data_loader import HistoricalDataLoader

        # Resolve strategy class
        dotted = _STRATEGY_REGISTRY.get(strategy_name)
        if dotted is None:
            raise ValueError(f"Unknown strategy: {strategy_name!r}")
        strategy_cls = _import_class(dotted)
        strategy = strategy_cls(**strategy_kwargs)

        loader = HistoricalDataLoader()
        df = loader.load(symbol, timeframe=timeframe)

        if df is None or df.empty:
            logger.warning(
                "No data for %s/%s — skipping", symbol, strategy_name
            )
            return {
                "symbol": symbol,
                "strategy_name": strategy_name,
                "error": "no_data",
            }

        commission = CommissionModel(
            mode="percentage",
            commission_pct=commission_pct,
        )
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            slippage_pct=slippage_pct,
        )
        result = engine.run(df, strategy)

        return {
            "symbol": symbol,
            "strategy_name": strategy_name,
            "total_return_pct": float(result.total_return_pct),
            "win_rate": float(result.win_rate),
            "sharpe_ratio": float(result.sharpe_ratio),
            "max_drawdown_pct": float(result.max_drawdown_pct),
            "profit_factor": float(result.profit_factor),
            "total_trades": int(result.total_trades),
            "sortino_ratio": float(result.sortino_ratio),
            "calmar_ratio": float(result.calmar_ratio),
            "avg_hold_bars": float(result.avg_hold_bars),
            "total_commission": float(result.total_commission),
            "error": None,
        }
    except (SystemExit, KeyboardInterrupt):
        # Allow graceful shutdown signals to propagate
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Backtest failed for %s/%s: %s", symbol, strategy_name, exc
        )
        return {
            "symbol": symbol,
            "strategy_name": strategy_name,
            "error": str(exc),
        }
    finally:
        gc.collect()


def _partition_runner(
    rows: Iterator,
    initial_capital: float,
    commission_pct: float,
    slippage_pct: float,
    timeframe: str,
) -> Iterator[Dict[str, Any]]:
    """Process a Spark partition of (symbol, strategy_name, kwargs) tuples.

    Args:
        rows:            Iterator over partition rows (each row is a dict
                         with keys ``symbol``, ``strategy_name``,
                         ``strategy_kwargs``).
        initial_capital: Starting capital in USD.
        commission_pct:  Commission rate.
        slippage_pct:    Slippage rate.
        timeframe:       Data timeframe.

    Yields:
        Result dicts from :func:`_run_single_backtest`.
    """
    for row in rows:
        result = _run_single_backtest(
            symbol=row["symbol"],
            strategy_name=row["strategy_name"],
            strategy_kwargs=row["strategy_kwargs"],
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            timeframe=timeframe,
        )
        yield result
        gc.collect()


class SparkBatchBacktester:
    """Distribute backtests across Spark partitions.

    Each Spark task runs one ``(symbol, strategy)`` pair independently
    using the existing :class:`~backtesting.engine.BacktestEngine` and
    :class:`~backtesting.data_loader.HistoricalDataLoader`.  Results are
    collected to the driver as a pandas DataFrame.

    Args:
        initial_capital: Starting portfolio value in USD
                         (default: 100 000).
        commission_pct:  Commission rate (default: 0.001 = 0.1 %).
        slippage_pct:    Slippage rate (default: 0.0005 = 0.05 %).

    Example::

        backtester = SparkBatchBacktester()
        results_df = backtester.run_batch(
            symbols=["AAPL", "MSFT"],
            strategies=["MACrossover", "RSIMeanReversion"],
        )
        backtester.generate_summary_report(results_df, "reports/batch.md")
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    # ------------------------------------------------------------------

    def run_batch(
        self,
        symbols: List[str],
        strategies: List[str],
        timeframe: str = "1d",
        strategy_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        data_loader: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Run all symbol × strategy combinations in parallel using Spark.

        Args:
            symbols:         List of ticker symbols.
            strategies:      List of strategy name strings.  Must be keys
                             in :data:`_STRATEGY_REGISTRY`.
            timeframe:       Data timeframe (e.g. ``"1d"``, ``"1h"``).
            strategy_kwargs: Optional dict mapping strategy name →
                             constructor kwargs dict.
            data_loader:     Ignored (reserved for future use; each
                             partition creates its own loader).

        Returns:
            pandas DataFrame with one row per (symbol, strategy) pair and
            columns:

            * ``symbol``
            * ``strategy_name``
            * ``total_return_pct``
            * ``win_rate``
            * ``sharpe_ratio``
            * ``max_drawdown_pct``
            * ``profit_factor``
            * ``total_trades``
            * ``sortino_ratio``
            * ``calmar_ratio``
            * ``avg_hold_bars``
            * ``total_commission``
            * ``error``
        """
        from config.spark_config import get_or_create_spark_session

        spark = get_or_create_spark_session()
        strategy_kwargs = strategy_kwargs or {}

        # Build (symbol, strategy_name, kwargs) combinations
        combinations = [
            {
                "symbol": sym,
                "strategy_name": strat,
                "strategy_kwargs": strategy_kwargs.get(strat, {}),
            }
            for sym in symbols
            for strat in strategies
        ]

        num_partitions = min(len(combinations), 8)
        logger.info(
            "Running batch: %d symbol(s) × %d strategy(ies) = %d jobs, "
            "%d Spark partitions",
            len(symbols), len(strategies), len(combinations), num_partitions,
        )

        t0 = time.time()

        # Capture primitive values for serialisation into closures
        _cap_ic = self.initial_capital
        _cap_cp = self.commission_pct
        _cap_sp = self.slippage_pct
        _cap_tf = timeframe

        rdd = spark.sparkContext.parallelize(combinations, numSlices=num_partitions)

        results_raw: List[Dict[str, Any]] = (
            rdd
            .mapPartitions(
                lambda rows: _partition_runner(
                    rows,
                    initial_capital=_cap_ic,
                    commission_pct=_cap_cp,
                    slippage_pct=_cap_sp,
                    timeframe=_cap_tf,
                )
            )
            .collect()
        )

        elapsed = time.time() - t0
        logger.info(
            "Batch complete: %d results in %.1fs", len(results_raw), elapsed
        )

        results_df = pd.DataFrame(results_raw)

        # Ensure all expected columns exist even when some jobs failed
        expected_cols = [
            "symbol", "strategy_name",
            "total_return_pct", "win_rate", "sharpe_ratio",
            "max_drawdown_pct", "profit_factor", "total_trades",
            "sortino_ratio", "calmar_ratio", "avg_hold_bars",
            "total_commission", "error",
        ]
        for col in expected_cols:
            if col not in results_df.columns:
                results_df[col] = None

        gc.collect()
        return results_df[expected_cols]

    # ------------------------------------------------------------------

    def generate_summary_report(
        self, results_df: pd.DataFrame, output_path: str
    ) -> None:
        """Write a Markdown summary table to *output_path*.

        Args:
            results_df:  DataFrame returned by :meth:`run_batch`.
            output_path: File path for the ``.md`` report.
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        successful = results_df[results_df["error"].isna()].copy()
        failed = results_df[~results_df["error"].isna()].copy()

        lines = [
            "# Batch Backtest Summary Report",
            "",
            f"**Total jobs:** {len(results_df)}  "
            f"| **Successful:** {len(successful)}  "
            f"| **Failed:** {len(failed)}",
            "",
        ]

        if not successful.empty:
            lines += [
                "## Results",
                "",
                "| Symbol | Strategy | Return% | Win Rate | Sharpe | "
                "Max DD% | Profit Factor | Trades | Sortino | Calmar |",
                "|--------|----------|---------|----------|--------|"
                "---------|---------------|--------|---------|--------|",
            ]
            for _, row in successful.iterrows():
                lines.append(
                    f"| {row['symbol']} "
                    f"| {row['strategy_name']} "
                    f"| {row.get('total_return_pct', 0):.2f} "
                    f"| {row.get('win_rate', 0):.2%} "
                    f"| {row.get('sharpe_ratio', 0):.3f} "
                    f"| {row.get('max_drawdown_pct', 0):.2f} "
                    f"| {row.get('profit_factor', 0):.3f} "
                    f"| {int(row.get('total_trades', 0))} "
                    f"| {row.get('sortino_ratio', 0):.3f} "
                    f"| {row.get('calmar_ratio', 0):.3f} |"
                )
            lines.append("")

        if not failed.empty:
            lines += [
                "## Failed Jobs",
                "",
                "| Symbol | Strategy | Error |",
                "|--------|----------|-------|",
            ]
            for _, row in failed.iterrows():
                lines.append(
                    f"| {row['symbol']} "
                    f"| {row['strategy_name']} "
                    f"| {row.get('error', 'unknown')} |"
                )
            lines.append("")

        output_path_obj.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Batch report written to %s", output_path)
