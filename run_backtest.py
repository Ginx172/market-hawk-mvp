#!/usr/bin/env python3
"""
SCRIPT NAME: run_backtest.py
====================================
Execution Location: K:\\_DEV_MVP_2026\\Market_Hawk_3\\
Purpose: Quick-launch script to run backtests from the command line.
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01

Usage:
    # Single symbol, single strategy
    python run_backtest.py --symbol AAPL --strategy ma_cross --timeframe 1h

    # Multiple symbols
    python run_backtest.py --symbols AAPL,NVDA,MSFT,BTCUSDT --strategy rsi

    # All built-in strategies on one symbol
    python run_backtest.py --symbol BTCUSDT --strategy all

    # With date range
    python run_backtest.py --symbol AAPL --strategy ma_cross --start 2024-01-01 --end 2024-12-31

    # Scan available local data
    python run_backtest.py --scan

    # Generate HTML report
    python run_backtest.py --symbol AAPL --strategy all --html
"""

import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from backtesting.engine import BacktestEngine
from backtesting.data_loader import HistoricalDataLoader
from backtesting.strategies import (
    MACrossover, RSIMeanReversion, BollingerBreakout,
    MACDHistogram, AgentConsensusStrategy
)
from backtesting.report import BacktestReport


# ============================================================
# STRATEGY REGISTRY
# ============================================================

STRATEGIES = {
    "ma_cross":   lambda: MACrossover(fast_period=20, slow_period=50),
    "ma_fast":    lambda: MACrossover(fast_period=10, slow_period=30, sl_pct=0.015, tp_pct=0.03),
    "rsi":        lambda: RSIMeanReversion(period=14),
    "rsi_tight":  lambda: RSIMeanReversion(period=14, oversold=25, overbought=75),
    "bb":         lambda: BollingerBreakout(period=20, num_std=2.0),
    "macd":       lambda: MACDHistogram(fast=12, slow=26, signal=9),
    "consensus":  lambda: AgentConsensusStrategy(consensus_threshold=0.30),
}


def main():
    parser = argparse.ArgumentParser(
        description="Market Hawk 3 — Backtesting Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  ma_cross    EMA 20/50 Crossover (Golden/Death Cross)
  ma_fast     EMA 10/30 Crossover (faster signals)
  rsi         RSI-14 Mean Reversion (30/70 levels)
  rsi_tight   RSI-14 with tighter 25/75 thresholds
  bb          Bollinger Bands Breakout (20-period, 2 std)
  macd        MACD Histogram Crossover (12/26/9)
  consensus   Agent Consensus (multi-indicator vote)
  all         Run ALL strategies for comparison
        """)

    parser.add_argument("--symbol", type=str, help="Single symbol (e.g., AAPL)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--strategy", type=str, default="ma_cross",
                        help="Strategy name or 'all'")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Candle timeframe (1m/5m/15m/1h/4h/1d)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Starting capital (default: 100000)")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Commission fraction (default: 0.001 = 0.1%%)")
    parser.add_argument("--scan", action="store_true",
                        help="Scan and list available local data")
    parser.add_argument("--rescan", action="store_true",
                        help="Force re-scan G: drive (ignore cache)")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML report")
    parser.add_argument("--json", action="store_true",
                        help="Save JSON results")

    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    for noisy in ["httpx", "httpcore", "urllib3", "yfinance", "peewee",
                  "chromadb"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ---- Scan mode ----
    if args.scan:
        loader = HistoricalDataLoader()
        index = loader.scan_available_data(force_rescan=args.rescan)
        print(f"\n📂 Found {len(index)} data files:\n")
        for key, info in sorted(index.items())[:50]:
            print(f"  {key:<40s} {info['size_mb']:>8.1f} MB  [{info['format']}]")
        if len(index) > 50:
            print(f"\n  ... and {len(index) - 50} more files")
        syms = loader.list_symbols()
        print(f"\n  Unique symbols: {len(syms)}")
        return

    # ---- Backtest mode ----
    if not args.symbol and not args.symbols:
        parser.error("Specify --symbol or --symbols (or use --scan)")

    symbols = []
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    elif args.symbol:
        symbols = [args.symbol]

    # Build strategy list
    if args.strategy == "all":
        strat_list = [(name, factory()) for name, factory in STRATEGIES.items()]
    elif args.strategy in STRATEGIES:
        strat_list = [(args.strategy, STRATEGIES[args.strategy]())]
    else:
        print(f"❌ Unknown strategy: {args.strategy}")
        print(f"   Available: {', '.join(STRATEGIES.keys())}, all")
        return

    # Load data
    loader = HistoricalDataLoader()
    if args.rescan:
        loader.scan_available_data(force_rescan=True)
    data = loader.load_multiple(symbols, args.timeframe, args.start, args.end)

    if not data:
        print("❌ No data loaded. Try --scan to see available files.")
        return

    # Run backtests
    engine = BacktestEngine(
        initial_capital=args.capital,
        commission_pct=args.commission,
    )

    report = BacktestReport()

    for strat_name, strategy in strat_list:
        for symbol, df in data.items():
            import copy
            strat = copy.deepcopy(strategy)
            result = engine.run(df, strat, symbol, args.timeframe)
            report.add_result(result)

    # Output
    report.print_summary()

    if args.html:
        path = report.save_html()
        print(f"\n📄 HTML report: {path}")

    if args.json:
        path = report.save_json()
        print(f"\n📄 JSON report: {path}")


if __name__ == "__main__":
    main()
