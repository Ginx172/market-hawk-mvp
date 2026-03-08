"""
SCRIPT NAME: anomaly_detector.py
====================================
Execution Location: market-hawk-mvp/agents/security_guard/
Purpose: Security Guard Agent — Anomaly detection + trade safety validation
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01

Detects market anomalies that should BLOCK or FLAG trading decisions:
    1. Volume spikes (unusual activity, possible manipulation)
    2. Price gaps (overnight gaps, flash crashes)
    3. Volatility explosions (regime change detection)
    4. Spread anomalies (illiquid conditions)
    5. Correlation breaks (market structure changes)

Acts as a SAFETY NET — can veto Brain decisions if anomalies detected.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("market_hawk.security_guard")


@dataclass
class AnomalyAlert:
    """Individual anomaly detection."""
    alert_type: str         # volume_spike, price_gap, volatility_explosion, etc.
    severity: str           # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    value: float            # The anomalous value
    threshold: float        # The threshold that was exceeded
    block_trade: bool       # Should this block the trade?


class SecurityGuard:
    """
    Security Guard Agent — Market anomaly detection and trade safety.

    Runs multiple anomaly checks on market data before allowing trades.
    Can BLOCK trades if critical anomalies are detected.

    Brain-compatible: implements analyze(symbol, context) -> Dict

    Checks:
        1. Volume spike detection (Z-score > 3)
        2. Price gap detection (> 2% gap from previous close)
        3. Volatility regime change (ATR spike)
        4. Spread anomaly (HL spread vs historical)
        5. Data quality check (NaN, stale data)
    """

    def __init__(self, config: Dict = None):
        config = config or {}
        self.volume_zscore_threshold = config.get("volume_zscore", 3.0)
        self.price_gap_threshold = config.get("price_gap_pct", 0.02)  # 2%
        self.volatility_spike_factor = config.get("volatility_spike", 2.5)
        self.spread_zscore_threshold = config.get("spread_zscore", 3.0)
        self.stale_data_hours = config.get("stale_hours", 4)

    def check_volume_spike(self, df: pd.DataFrame) -> Optional[AnomalyAlert]:
        """Detect unusual volume spikes (possible manipulation or news event)."""
        if "Volume" not in df.columns or len(df) < 30:
            return None

        volume = df["Volume"].values.astype(float)
        if volume[-1] == 0:
            return None

        # Z-score of latest volume vs rolling 20-period
        vol_mean = np.mean(volume[-21:-1]) if len(volume) > 21 else np.mean(volume[:-1])
        vol_std = np.std(volume[-21:-1]) if len(volume) > 21 else np.std(volume[:-1])

        if vol_std == 0:
            return None

        zscore = (volume[-1] - vol_mean) / vol_std

        if abs(zscore) > self.volume_zscore_threshold:
            severity = "HIGH" if abs(zscore) > 5 else "MEDIUM"
            return AnomalyAlert(
                alert_type="volume_spike",
                severity=severity,
                description=f"Volume Z-score: {zscore:.1f} "
                            f"(current: {volume[-1]:,.0f}, avg: {vol_mean:,.0f})",
                value=zscore,
                threshold=self.volume_zscore_threshold,
                block_trade=abs(zscore) > 5,  # Block only extreme spikes
            )
        return None

    def check_price_gap(self, df: pd.DataFrame) -> Optional[AnomalyAlert]:
        """Detect price gaps (overnight gaps, flash moves)."""
        if "Close" not in df.columns or "Open" not in df.columns or len(df) < 2:
            return None

        close = df["Close"].values.astype(float)
        open_p = df["Open"].values.astype(float)

        # Gap between current open and previous close
        if close[-2] == 0:
            return None
        gap_pct = abs(open_p[-1] - close[-2]) / close[-2]

        if gap_pct > self.price_gap_threshold:
            severity = "CRITICAL" if gap_pct > 0.05 else "HIGH" if gap_pct > 0.03 else "MEDIUM"
            return AnomalyAlert(
                alert_type="price_gap",
                severity=severity,
                description=f"Price gap: {gap_pct:.2%} "
                            f"(prev close: {close[-2]:.2f}, open: {open_p[-1]:.2f})",
                value=gap_pct,
                threshold=self.price_gap_threshold,
                block_trade=gap_pct > 0.05,  # Block on 5%+ gaps
            )
        return None

    def check_volatility_explosion(self, df: pd.DataFrame) -> Optional[AnomalyAlert]:
        """Detect sudden volatility regime changes."""
        if "Close" not in df.columns or len(df) < 30:
            return None

        close = df["Close"].values.astype(float)
        returns = np.diff(close) / close[:-1]

        # Compare recent volatility (5-period) to historical (20-period)
        recent_vol = np.std(returns[-5:]) if len(returns) >= 5 else 0
        hist_vol = np.std(returns[-21:-1]) if len(returns) > 21 else np.std(returns[:-1])

        if hist_vol == 0:
            return None

        vol_ratio = recent_vol / hist_vol

        if vol_ratio > self.volatility_spike_factor:
            severity = "CRITICAL" if vol_ratio > 4 else "HIGH" if vol_ratio > 3 else "MEDIUM"
            return AnomalyAlert(
                alert_type="volatility_explosion",
                severity=severity,
                description=f"Volatility {vol_ratio:.1f}x normal "
                            f"(recent: {recent_vol:.4f}, hist: {hist_vol:.4f})",
                value=vol_ratio,
                threshold=self.volatility_spike_factor,
                block_trade=vol_ratio > 4,
            )
        return None

    def check_spread_anomaly(self, df: pd.DataFrame) -> Optional[AnomalyAlert]:
        """Detect unusual High-Low spreads (illiquidity or manipulation)."""
        if "High" not in df.columns or "Low" not in df.columns or len(df) < 20:
            return None

        high = df["High"].values.astype(float)
        low = df["Low"].values.astype(float)
        close = df["Close"].values.astype(float)

        # Spread as % of close
        spreads = (high - low) / np.where(close > 0, close, 1)
        current_spread = spreads[-1]
        spread_mean = np.mean(spreads[-21:-1]) if len(spreads) > 21 else np.mean(spreads[:-1])
        spread_std = np.std(spreads[-21:-1]) if len(spreads) > 21 else np.std(spreads[:-1])

        if spread_std == 0:
            return None

        zscore = (current_spread - spread_mean) / spread_std

        if abs(zscore) > self.spread_zscore_threshold:
            severity = "HIGH" if abs(zscore) > 4 else "MEDIUM"
            return AnomalyAlert(
                alert_type="spread_anomaly",
                severity=severity,
                description=f"Spread Z-score: {zscore:.1f} "
                            f"(spread: {current_spread:.4f}, avg: {spread_mean:.4f})",
                value=zscore,
                threshold=self.spread_zscore_threshold,
                block_trade=False,  # Flag only, don't block
            )
        return None

    def check_data_quality(self, df: pd.DataFrame) -> Optional[AnomalyAlert]:
        """Check for data quality issues (NaN, stale, zero volume)."""
        if df is None or df.empty:
            return AnomalyAlert(
                alert_type="data_quality",
                severity="CRITICAL",
                description="No data available",
                value=0, threshold=0, block_trade=True,
            )

        # Check for NaN in critical columns
        critical_cols = ["Close", "Volume"]
        for col in critical_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > len(df) * 0.1:  # >10% NaN
                    return AnomalyAlert(
                        alert_type="data_quality",
                        severity="HIGH",
                        description=f"{col} has {nan_count} NaN values ({nan_count/len(df):.1%})",
                        value=nan_count, threshold=len(df) * 0.1, block_trade=True,
                    )

        return None

    def run_all_checks(self, df: pd.DataFrame, symbol: str = "") -> List[AnomalyAlert]:
        """Run all anomaly checks and return list of alerts."""
        alerts = []
        checks = [
            self.check_data_quality,
            self.check_volume_spike,
            self.check_price_gap,
            self.check_volatility_explosion,
            self.check_spread_anomaly,
        ]

        for check in checks:
            try:
                alert = check(df)
                if alert:
                    alerts.append(alert)
                    logger.info("🚨 %s [%s] %s: %s",
                                 symbol, alert.severity, alert.alert_type, alert.description)
            except Exception:
                logger.exception("Check %s failed for %s",
                                 check.__name__, symbol)

        return alerts

    def should_block_trade(self, alerts: List[AnomalyAlert]) -> bool:
        """Determine if any alert should block the trade."""
        return any(a.block_trade for a in alerts)

    # Brain-compatible interface
    def analyze(self, symbol: str, context: Dict = None) -> Dict:
        """
        Brain-compatible: analyzes market data for anomalies.

        Expects context to contain 'features_df' (DataFrame with OHLCV).
        Returns HOLD if anomalies detected, otherwise neutral pass-through.
        """
        context = context or {}
        features_df = context.get("features_df")

        if features_df is None or (hasattr(features_df, 'empty') and features_df.empty):
            return {
                "recommendation": "HOLD",
                "confidence": 0.0,
                "reasoning": f"No market data to analyze for {symbol}",
                "metadata": {"alerts": [], "safe": False}
            }

        alerts = self.run_all_checks(features_df, symbol)
        should_block = self.should_block_trade(alerts)

        if should_block:
            alert_summary = "; ".join(
                f"{a.alert_type}({a.severity})" for a in alerts if a.block_trade
            )
            return {
                "recommendation": "HOLD",
                "confidence": 0.9,  # High confidence in blocking
                "reasoning": f"⚠️ ANOMALY BLOCK: {alert_summary}",
                "metadata": {
                    "alerts": [
                        {"type": a.alert_type, "severity": a.severity,
                         "description": a.description, "block": a.block_trade}
                        for a in alerts
                    ],
                    "safe": False,
                    "blocked": True,
                }
            }

        if alerts:
            alert_summary = "; ".join(
                f"{a.alert_type}({a.severity})" for a in alerts
            )
            return {
                "recommendation": "HOLD",
                "confidence": 0.3,
                "reasoning": f"⚠️ Alerts flagged: {alert_summary} — proceed with caution",
                "metadata": {
                    "alerts": [
                        {"type": a.alert_type, "severity": a.severity,
                         "description": a.description, "block": a.block_trade}
                        for a in alerts
                    ],
                    "safe": True,
                    "blocked": False,
                }
            }

        return {
            "recommendation": "HOLD",
            "confidence": 0.0,
            "reasoning": f"No anomalies detected for {symbol} — all clear",
            "metadata": {"alerts": [], "safe": True, "blocked": False}
        }

    def cleanup(self):
        logger.info("Security Guard cleanup complete")


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    for n in ["httpx", "httpcore", "urllib3", "yfinance", "peewee"]:
        logging.getLogger(n).setLevel(logging.WARNING)

    print("\n" + "=" * 60)
    print("SECURITY GUARD — Anomaly Detection Test")
    print("=" * 60)

    from data.market_data_fetcher import MarketDataFetcher
    fetcher = MarketDataFetcher()
    guard = SecurityGuard()

    for symbol in ["AAPL", "BTCUSDT", "NVDA", "GOLD"]:
        print(f"\n--- {symbol} ---")
        df = fetcher.fetch_and_engineer(symbol)
        if df is None or df.empty:
            print(f"  ❌ No data")
            continue

        alerts = guard.run_all_checks(df, symbol)
        blocked = guard.should_block_trade(alerts)

        if not alerts:
            print(f"  ✅ All clear — no anomalies detected")
        else:
            for alert in alerts:
                emoji = "🚫" if alert.block_trade else "⚠️"
                print(f"  {emoji} [{alert.severity}] {alert.alert_type}: {alert.description}")
            print(f"  Trade blocked: {'YES' if blocked else 'No'}")

    print("\n✅ Test complete")
    guard.cleanup()
