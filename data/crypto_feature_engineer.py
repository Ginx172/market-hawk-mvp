"""
SCRIPT NAME: crypto_feature_engineer.py
====================================
Execution Location: market-hawk-mvp/data/
Purpose: Crypto-specific feature engineering — 20 additional features on top of
         the base 60 features from MarketDataFetcher.engineer_features().
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-13

Provides CryptoFeatureEngineer that adds:
    - Volatility features (crypto is 3-5x more volatile than stocks)
    - Volume features (24/7 market, volume patterns differ from equities)
    - Momentum features (trend persistence differs in crypto)
    - Price structure features (Bollinger position, mean-reversion signal)
    - Session/cyclical features (Asia, Europe, US trading sessions)

Usage:
    from data.crypto_feature_engineer import CryptoFeatureEngineer, CRYPTO_EXTRA_FEATURES

    engineer = CryptoFeatureEngineer(periods_per_day=6)  # 6 bars/day for 4h candles
    df_with_crypto_features = engineer.add_features(df_with_base_60_features)
"""

import logging
import math
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("market_hawk.data.crypto_features")


# ============================================================
# CRYPTO-SPECIFIC FEATURE NAMES
# ============================================================

CRYPTO_EXTRA_FEATURES: List[str] = [
    # Volatility (4)
    "crypto_volatility_ratio",
    "crypto_range_pct",
    "crypto_atr_pct",
    "crypto_volatility_zscore",
    # Volume (4)
    "crypto_volume_ratio_8h",
    "crypto_volume_ratio_24h",
    "crypto_volume_trend",
    "crypto_obv_momentum",
    # Momentum (4)
    "crypto_momentum_3d",
    "crypto_momentum_7d",
    "crypto_rsi_divergence",
    "crypto_macd_histogram_momentum",
    # Price structure (4)
    "crypto_distance_from_high_20",
    "crypto_distance_from_low_20",
    "crypto_bb_position",
    "crypto_mean_reversion_signal",
    # Session / cyclical (4)
    "crypto_hour_sin",
    "crypto_hour_cos",
    "crypto_is_weekend",
    "crypto_day_of_week_sin",
]


# ============================================================
# CRYPTO FEATURE ENGINEER
# ============================================================

class CryptoFeatureEngineer:
    """Add 20 crypto-specific features on top of the base 60-feature set.

    This class expects a DataFrame that already has the 60 base features
    produced by ``MarketDataFetcher.engineer_features()``.  It reads:
    - ``Close``, ``High``, ``Low``, ``Volume`` — raw OHLCV
    - ``returns`` — per-bar return (pct_change)
    - ``ATR`` — Average True Range
    - ``RSI`` — Relative Strength Index (14-period)
    - ``MACD_histogram`` — MACD - signal
    - ``OBV`` — On-Balance Volume
    - ``BB_upper``, ``BB_lower`` — Bollinger Band edges
    - ``SMA_20`` — 20-period simple moving average
    - ``volatility_20`` — 20-period rolling std of returns
    - ``hour``, ``day_of_week`` — time features (0 if unavailable for daily data)

    Args:
        periods_per_day: Number of candle bars per trading day.
            Use 6 for 4h candles, 24 for 1h candles, 1 for daily.
            Used to scale momentum look-back windows.

    Raises:
        ValueError: If required base columns are missing from the DataFrame.
    """

    _REQUIRED_COLS = [
        "Close", "High", "Low", "Volume",
        "returns", "ATR", "RSI", "MACD_histogram", "OBV",
        "BB_upper", "BB_lower", "SMA_20", "volatility_20",
    ]

    def __init__(self, periods_per_day: int = 1) -> None:
        if periods_per_day < 1:
            raise ValueError(f"periods_per_day must be >= 1, got {periods_per_day}")
        self.periods_per_day = periods_per_day

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and append all 20 crypto-specific features.

        Args:
            df: DataFrame with base 60 features already computed.
                Must have at least the columns listed in ``_REQUIRED_COLS``.
                The DataFrame index should be a DatetimeIndex for time-based
                features; if not, ``hour`` and ``day_of_week`` columns are
                used instead (or zero-filled).

        Returns:
            New DataFrame (copy) with 20 additional feature columns appended.
            All NaN/Inf values are replaced with 0 to keep training clean.

        Raises:
            ValueError: If any required column is absent.
        """
        missing = [c for c in self._REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CryptoFeatureEngineer: missing required columns: {missing}. "
                "Run MarketDataFetcher.engineer_features() first."
            )

        result = df.copy()

        close = result["Close"].astype(float)
        high = result["High"].astype(float)
        low = result["Low"].astype(float)
        volume = result["Volume"].astype(float)
        returns = result["returns"].astype(float)
        atr = result["ATR"].astype(float)
        rsi = result["RSI"].astype(float)
        macd_hist = result["MACD_histogram"].astype(float)
        obv = result["OBV"].astype(float)
        bb_upper = result["BB_upper"].astype(float)
        bb_lower = result["BB_lower"].astype(float)
        sma_20 = result["SMA_20"].astype(float)
        vol_20 = result["volatility_20"].astype(float)

        ppd = self.periods_per_day

        # --------------------------------------------------------
        # 1. VOLATILITY FEATURES
        # --------------------------------------------------------

        # rolling std of returns over 10 and 50 bars
        roll_std_10 = returns.rolling(10).std()
        roll_std_50 = returns.rolling(50).std()
        result["crypto_volatility_ratio"] = roll_std_10 / roll_std_50.replace(0, np.nan)

        result["crypto_range_pct"] = (high - low) / close.replace(0, np.nan) * 100

        result["crypto_atr_pct"] = atr / close.replace(0, np.nan) * 100

        vol_mean = vol_20.rolling(50).mean()
        vol_std = vol_20.rolling(50).std()
        result["crypto_volatility_zscore"] = (
            (vol_20 - vol_mean) / vol_std.replace(0, np.nan)
        )

        # --------------------------------------------------------
        # 2. VOLUME FEATURES
        # --------------------------------------------------------

        vol_roll_8 = volume.rolling(8).mean()
        result["crypto_volume_ratio_8h"] = volume / vol_roll_8.replace(0, np.nan)

        vol_roll_24 = volume.rolling(24).mean()
        result["crypto_volume_ratio_24h"] = volume / vol_roll_24.replace(0, np.nan)

        vol_roll_5 = volume.rolling(5).mean()
        vol_roll_20 = volume.rolling(20).mean()
        result["crypto_volume_trend"] = vol_roll_5 / vol_roll_20.replace(0, np.nan)

        obv_diff5 = obv.diff(5)
        obv_abs_roll20 = obv.abs().rolling(20).mean()
        result["crypto_obv_momentum"] = obv_diff5 / obv_abs_roll20.replace(0, np.nan)

        # --------------------------------------------------------
        # 3. MOMENTUM FEATURES
        # --------------------------------------------------------

        lookback_3d = max(1, 3 * ppd)
        lookback_7d = max(1, 7 * ppd)

        result["crypto_momentum_3d"] = returns.rolling(lookback_3d).sum()
        result["crypto_momentum_7d"] = returns.rolling(lookback_7d).sum()

        result["crypto_rsi_divergence"] = rsi.diff(5)

        result["crypto_macd_histogram_momentum"] = macd_hist.diff(3)

        # --------------------------------------------------------
        # 4. PRICE STRUCTURE FEATURES
        # --------------------------------------------------------

        high_20_max = high.rolling(20).max()
        low_20_min = low.rolling(20).min()

        result["crypto_distance_from_high_20"] = (
            (close - high_20_max) / close.replace(0, np.nan)
        )
        result["crypto_distance_from_low_20"] = (
            (close - low_20_min) / close.replace(0, np.nan)
        )

        bb_range = (bb_upper - bb_lower).replace(0, np.nan)
        result["crypto_bb_position"] = (close - bb_lower) / bb_range

        close_std_20 = close.rolling(20).std()
        result["crypto_mean_reversion_signal"] = (
            (close - sma_20) / close_std_20.replace(0, np.nan)
        )

        # --------------------------------------------------------
        # 5. SESSION / CYCLICAL FEATURES
        # --------------------------------------------------------

        # Prefer DatetimeIndex; fall back to 'hour' / 'day_of_week' columns
        if isinstance(result.index, pd.DatetimeIndex):
            hour_vals = result.index.hour.astype(float)
            dow_vals = result.index.dayofweek.astype(float)
        elif "hour" in result.columns and "day_of_week" in result.columns:
            hour_vals = result["hour"].astype(float)
            dow_vals = result["day_of_week"].astype(float)
        else:
            logger.warning(
                "No DatetimeIndex or hour/day_of_week columns — "
                "session features will be zero-filled."
            )
            hour_vals = pd.Series(np.zeros(len(result)), index=result.index)
            dow_vals = pd.Series(np.zeros(len(result)), index=result.index)

        result["crypto_hour_sin"] = np.sin(2 * math.pi * hour_vals / 24)
        result["crypto_hour_cos"] = np.cos(2 * math.pi * hour_vals / 24)
        result["crypto_is_weekend"] = (dow_vals >= 5).astype(int)
        result["crypto_day_of_week_sin"] = np.sin(2 * math.pi * dow_vals / 7)

        # --------------------------------------------------------
        # CLEAN: replace inf/nan with 0 (consistent with base pipeline)
        # --------------------------------------------------------
        for col in CRYPTO_EXTRA_FEATURES:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan).fillna(0)

        logger.info(
            "CryptoFeatureEngineer: added %d features, total cols=%d",
            len(CRYPTO_EXTRA_FEATURES), len(result.columns),
        )
        return result
