#!/usr/bin/env python3
"""
SCRIPT NAME: report.py
====================================
Execution Location: K:\\_DEV_MVP_2026\\Market_Hawk_3\\backtesting\\
Purpose: Generate professional HTML & JSON backtest reports with
         equity curves, drawdown charts, trade analysis, and comparisons.
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01
"""

import json
import logging
from html import escape as html_escape
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from backtesting.engine import BacktestResult

logger = logging.getLogger("market_hawk.backtest.report")


class BacktestReport:
    """
    Generate comprehensive backtest reports.

    Outputs:
        - JSON: Machine-readable results for dashboard integration
        - HTML: Standalone report with embedded charts (Plotly)
        - Console: Quick summary table
    """

    REPORTS_DIR = Path("logs/backtest_reports")

    def __init__(self, results: List[BacktestResult] = None):
        self.results = results or []
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def add_result(self, result: BacktestResult):
        """Add a backtest result for reporting."""
        self.results.append(result)

    # ============================================================
    # CONSOLE REPORT
    # ============================================================

    def print_summary(self):
        """Print a formatted summary table to console."""
        if not self.results:
            print("No results to display.")
            return

        print("\n" + "=" * 100)
        print("  MARKET HAWK 3 — BACKTEST SUMMARY REPORT")
        print("  Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 100)

        # Header
        print(f"\n  {'Strategy':<35s} {'Symbol':<10s} {'Return':>8s} "
              f"{'Trades':>7s} {'WinRate':>8s} {'Sharpe':>7s} "
              f"{'MaxDD':>7s} {'P/F':>6s} {'Time':>6s}")
        print("  " + "─" * 96)

        for r in self.results:
            ret_color = "+" if r.total_return_pct >= 0 else ""
            print(f"  {r.strategy_name:<35s} {r.symbol:<10s} "
                  f"{ret_color}{r.total_return_pct*100:>7.2f}% "
                  f"{r.total_trades:>7d} {r.win_rate*100:>7.1f}% "
                  f"{r.sharpe_ratio:>7.2f} {r.max_drawdown_pct*100:>6.2f}% "
                  f"{r.profit_factor:>6.2f} {r.execution_time_sec:>5.1f}s")

        print("  " + "─" * 96)

        # Best performer
        if len(self.results) > 1:
            best = max(self.results, key=lambda r: r.sharpe_ratio)
            print(f"\n  🏆 Best Sharpe: {best.strategy_name} on {best.symbol} "
                  f"(Sharpe={best.sharpe_ratio:.2f}, Return={best.total_return_pct*100:+.2f}%)")

        print("=" * 100)

    # ============================================================
    # JSON REPORT
    # ============================================================

    def save_json(self, filename: Optional[str] = None) -> str:
        """Save results to JSON file."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{ts}.json"

        filepath = self.REPORTS_DIR / filename

        report = {
            "generated_at": datetime.now().isoformat(),
            "total_strategies": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("JSON report saved: %s", filepath)
        return str(filepath)

    # ============================================================
    # HTML REPORT (Standalone with Plotly CDN)
    # ============================================================

    def save_html(self, filename: Optional[str] = None) -> str:
        """Generate a standalone HTML report with interactive charts."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{ts}.html"

        filepath = self.REPORTS_DIR / filename

        html = self._build_html()

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("HTML report saved: %s", filepath)
        return str(filepath)

    def _build_html(self) -> str:
        """Build the full HTML report."""
        rows_html = ""
        charts_html = ""

        for i, r in enumerate(self.results):
            safe_strategy = html_escape(str(r.strategy_name))
            safe_symbol = html_escape(str(r.symbol))
            safe_start = html_escape(str(r.start_date)[:10])
            safe_end = html_escape(str(r.end_date)[:10])

            ret_class = "positive" if r.total_return_pct >= 0 else "negative"
            rows_html += f"""
            <tr>
                <td>{safe_strategy}</td>
                <td>{safe_symbol}</td>
                <td>{safe_start} → {safe_end}</td>
                <td>{r.total_bars:,}</td>
                <td class="{ret_class}">{r.total_return_pct*100:+.2f}%</td>
                <td>{r.total_trades}</td>
                <td>{r.win_rate*100:.1f}%</td>
                <td>{r.sharpe_ratio:.2f}</td>
                <td>{r.sortino_ratio:.2f}</td>
                <td>{r.max_drawdown_pct*100:.2f}%</td>
                <td>{r.profit_factor:.2f}</td>
                <td>${r.final_equity:,.0f}</td>
            </tr>"""

            # Equity curve chart data
            eq_data = r.equity_curve
            dd_data = r.drawdown_curve
            # Downsample for performance
            if len(eq_data) > 2000:
                step = len(eq_data) // 2000
                eq_data = eq_data[::step]
                dd_data = dd_data[::step]

            charts_html += f"""
            <div class="chart-container">
                <h3>{safe_strategy} — {safe_symbol}</h3>
                <div id="equity_{i}" style="height:300px;"></div>
                <div id="drawdown_{i}" style="height:200px;"></div>
                <script>
                    Plotly.newPlot('equity_{i}', [{{
                        y: {json.dumps(eq_data[:2000])},
                        type: 'scatter', mode: 'lines',
                        name: 'Equity',
                        line: {{color: '{("#22c55e" if r.total_return_pct >= 0 else "#ef4444")}'}}
                    }}], {{
                        title: 'Equity Curve',
                        yaxis: {{title: 'USD', tickformat: '$,.0f'}},
                        margin: {{l:80, r:30, t:40, b:40}},
                        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
                        font: {{color: '#e0e0e0'}}
                    }});
                    Plotly.newPlot('drawdown_{i}', [{{
                        y: {json.dumps([d*100 for d in dd_data[:2000]])},
                        type: 'scatter', mode: 'lines', fill: 'tozeroy',
                        name: 'Drawdown',
                        line: {{color: '#ef4444'}},
                        fillcolor: 'rgba(239,68,68,0.2)'
                    }}], {{
                        title: 'Drawdown',
                        yaxis: {{title: '%', tickformat: '.1f'}},
                        margin: {{l:80, r:30, t:40, b:40}},
                        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
                        font: {{color: '#e0e0e0'}}
                    }});
                </script>
            </div>
            """

            # Monthly returns heatmap
            if r.monthly_returns:
                months = list(r.monthly_returns.keys())
                values = [v * 100 for v in r.monthly_returns.values()]
                charts_html += f"""
                <div class="chart-container">
                    <h3>Monthly Returns — {safe_strategy}</h3>
                    <div id="monthly_{i}" style="height:200px;"></div>
                    <script>
                        Plotly.newPlot('monthly_{i}', [{{
                            x: {json.dumps(months)},
                            y: {json.dumps(values)},
                            type: 'bar',
                            marker: {{color: {json.dumps(values)}.map(v => v >= 0 ? '#22c55e' : '#ef4444')}}
                        }}], {{
                            title: 'Monthly Returns (%)',
                            margin: {{l:60, r:30, t:40, b:60}},
                            paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
                            font: {{color: '#e0e0e0'}}
                        }});
                    </script>
                </div>
                """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Market Hawk 3 — Backtest Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #0f0f23; color: #e0e0e0; padding: 20px; }}
        h1 {{ text-align: center; color: #ffd700; margin: 20px 0; font-size: 28px; }}
        h2 {{ color: #87ceeb; margin: 20px 0 10px; }}
        h3 {{ color: #ffd700; margin: 10px 0; }}
        .meta {{ text-align: center; color: #888; margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #1a1a3e; color: #ffd700; padding: 12px 8px; text-align: right; }}
        th:first-child, th:nth-child(2), th:nth-child(3) {{ text-align: left; }}
        td {{ padding: 10px 8px; text-align: right; border-bottom: 1px solid #2a2a4e; }}
        td:first-child, td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
        tr:hover {{ background: #1a1a3e; }}
        .positive {{ color: #22c55e; font-weight: bold; }}
        .negative {{ color: #ef4444; font-weight: bold; }}
        .chart-container {{ background: #1a1a2e; border-radius: 8px; padding: 15px; margin: 20px 0; }}
        .footer {{ text-align: center; color: #555; margin-top: 40px; padding: 20px; }}
    </style>
</head>
<body>
    <h1>🦅 Market Hawk 3 — Backtest Report</h1>
    <p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
       Strategies: {len(self.results)} | 
       Hardware: Intel i7-9700F + GTX 1070 8GB + 64GB DDR4</p>

    <h2>📊 Strategy Comparison</h2>
    <table>
        <tr>
            <th>Strategy</th><th>Symbol</th><th>Period</th><th>Bars</th>
            <th>Return</th><th>Trades</th><th>Win Rate</th>
            <th>Sharpe</th><th>Sortino</th><th>Max DD</th>
            <th>P/F</th><th>Final Equity</th>
        </tr>
        {rows_html}
    </table>

    <h2>📈 Equity Curves & Drawdown</h2>
    {charts_html}

    <div class="footer">
        <p>Market Hawk 3 Backtesting Engine — Production-Grade AI Trading System</p>
        <p>⚠️ Past performance does not guarantee future results. Paper trading only.</p>
    </div>
</body>
</html>"""


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    print("BacktestReport module — import and use with BacktestResult objects.")
    print("Example:")
    print("  report = BacktestReport(results)")
    print("  report.print_summary()")
    print("  report.save_html()")
    print("  report.save_json()")
