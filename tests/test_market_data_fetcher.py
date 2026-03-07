"""
Tests for data/market_data_fetcher.py

Covers:
    - engineer_features: correct column count, NaN handling
    - OBV vectorized computation: matches expected logic
    - Session detection: London, NY, Asia boundaries
    - Volume ratio calculation
    - Symbol mapping and category detection
    - No external API calls (all tests use synthetic data)
"""

import pytest
import numpy as np
import pandas as pd

from data.market_data_fetcher import (
    MarketDataFetcher,
    get_yfinance_ticker,
    get_symbol_category,
    TICKER_MAP,
)


# ============================================================
# SYMBOL MAPPING
# ============================================================

class TestSymbolMapping:
    """Test ticker conversion and category detection."""

    def test_btcusdt_maps_to_yfinance(self):
        assert get_yfinance_ticker("BTCUSDT") == "BTC-USD"

    def test_gold_maps_to_futures(self):
        assert get_yfinance_ticker("GOLD") == "GC=F"

    def test_unknown_symbol_passthrough(self):
        assert get_yfinance_ticker("AAPL") == "AAPL"

    def test_category_crypto(self):
        assert get_symbol_category("BTCUSDT") == "crypto"

    def test_category_stock(self):
        assert get_symbol_category("AAPL") == "tech_stocks"

    def test_category_forex_heuristic(self):
        assert get_symbol_category("GBPUSD") == "forex"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

class TestEngineerFeatures:
    """Test the 60-feature engineering pipeline."""

    @pytest.fixture
    def fetcher(self):
        return MarketDataFetcher()

    def test_output_has_60_plus_columns(self, fetcher, sample_ohlcv_df):
        result = fetcher.engineer_features(sample_ohlcv_df, symbol="TEST")
        # Should have at least 60 feature columns
        assert len(result.columns) >= 60

    def test_no_nan_after_engineering(self, fetcher, sample_ohlcv_df):
        result = fetcher.engineer_features(sample_ohlcv_df, symbol="TEST")
        # ffill + bfill + fillna(0) should eliminate NaN
        assert result.isna().sum().sum() == 0

    def test_no_inf_after_engineering(self, fetcher, sample_ohlcv_df):
        result = fetcher.engineer_features(sample_ohlcv_df, symbol="TEST")
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_row_count_preserved(self, fetcher, sample_ohlcv_df):
        result = fetcher.engineer_features(sample_ohlcv_df, symbol="TEST")
        assert len(result) == len(sample_ohlcv_df)

    def test_source_file_is_numeric(self, fetcher, sample_ohlcv_df):
        result = fetcher.engineer_features(sample_ohlcv_df, symbol="TEST")
        assert "source_file" in result.columns
        assert result["source_file"].dtype in [np.int64, np.float64, int, float]


# ============================================================
# OBV VECTORIZED
# ============================================================

class TestOBV:
    """Test On-Balance Volume vectorized computation."""

    def test_obv_all_up(self):
        """When price goes up every bar, OBV should be cumulative volume."""
        fetcher = MarketDataFetcher()
        dates = pd.date_range("2025-01-01", periods=5, freq="h")
        df = pd.DataFrame({
            "Open":   [10, 11, 12, 13, 14],
            "High":   [11, 12, 13, 14, 15],
            "Low":    [ 9, 10, 11, 12, 13],
            "Close":  [11, 12, 13, 14, 15],  # Monotonically increasing
            "Volume": [100, 200, 300, 400, 500],
        }, index=dates)

        result = fetcher.engineer_features(df, symbol="TEST")
        obv = result["OBV"].values

        # First bar: sign(diff=0)*vol = 0
        # Bars 1-4: all positive diffs, so OBV accumulates
        # OBV = cumsum([0, 200, 300, 400, 500]) = [0, 200, 500, 900, 1400]
        assert obv[0] == 0  # prepend makes first diff = 0
        assert obv[-1] > 0  # Final OBV positive for uptrend

    def test_obv_all_down(self):
        """When price goes down every bar, OBV should be negative cumulative."""
        fetcher = MarketDataFetcher()
        dates = pd.date_range("2025-01-01", periods=5, freq="h")
        df = pd.DataFrame({
            "Open":   [15, 14, 13, 12, 11],
            "High":   [16, 15, 14, 13, 12],
            "Low":    [14, 13, 12, 11, 10],
            "Close":  [14, 13, 12, 11, 10],  # Monotonically decreasing
            "Volume": [100, 200, 300, 400, 500],
        }, index=dates)

        result = fetcher.engineer_features(df, symbol="TEST")
        obv = result["OBV"].values
        assert obv[-1] < 0  # Final OBV negative for downtrend


# ============================================================
# SESSION DETECTION
# ============================================================

class TestSessionDetection:
    """Test London/NY/Asia session flags are vectorized correctly."""

    def _make_hourly_df(self, hours):
        """Create a minimal DataFrame with specific hours."""
        n = len(hours)
        dates = [pd.Timestamp(f"2025-01-01 {h:02d}:00:00") for h in hours]
        return pd.DataFrame({
            "Open": [100] * n, "High": [101] * n, "Low": [99] * n,
            "Close": [100] * n, "Volume": [1000] * n,
        }, index=pd.DatetimeIndex(dates))

    def test_london_session_boundaries(self):
        fetcher = MarketDataFetcher()
        df = self._make_hourly_df([7, 8, 12, 16, 17])
        result = fetcher.engineer_features(df, symbol="TEST")
        london = result["is_london_session"].values
        # hour 7 -> 0, hour 8 -> 1, hour 12 -> 1, hour 16 -> 1, hour 17 -> 0
        assert london[0] == 0
        assert london[1] == 1
        assert london[2] == 1
        assert london[3] == 1
        assert london[4] == 0

    def test_ny_session_boundaries(self):
        fetcher = MarketDataFetcher()
        df = self._make_hourly_df([12, 13, 17, 21, 22])
        result = fetcher.engineer_features(df, symbol="TEST")
        ny = result["is_ny_session"].values
        # hour 12 -> 0, hour 13 -> 1, hour 17 -> 1, hour 21 -> 1, hour 22 -> 0
        assert ny[0] == 0
        assert ny[1] == 1
        assert ny[3] == 1
        assert ny[4] == 0

    def test_asia_session_boundaries(self):
        fetcher = MarketDataFetcher()
        df = self._make_hourly_df([0, 4, 8, 9])
        result = fetcher.engineer_features(df, symbol="TEST")
        asia = result["is_asia_session"].values
        # hour 0 -> 1, hour 4 -> 1, hour 8 -> 1, hour 9 -> 0
        assert asia[0] == 1
        assert asia[1] == 1
        assert asia[2] == 1
        assert asia[3] == 0


# ============================================================
# VOLUME RATIO
# ============================================================

class TestVolumeRatio:
    """Test volume_ratio = volume / volume_SMA(20)."""

    def test_volume_ratio_exists(self, sample_ohlcv_df):
        fetcher = MarketDataFetcher()
        result = fetcher.engineer_features(sample_ohlcv_df, symbol="TEST")
        assert "volume_ratio" in result.columns
        # After fillna, no NaN
        assert result["volume_ratio"].isna().sum() == 0

    def test_volume_ratio_near_one_for_stable(self):
        """Constant volume should give ratio ~1.0 after warmup."""
        fetcher = MarketDataFetcher()
        n = 50
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "Open": [100] * n, "High": [101] * n, "Low": [99] * n,
            "Close": [100] * n, "Volume": [1000.0] * n,
        }, index=dates)
        result = fetcher.engineer_features(df, symbol="TEST")
        # After 20-bar warmup, ratio should be ~1.0
        late_ratios = result["volume_ratio"].iloc[25:].values
        np.testing.assert_allclose(late_ratios, 1.0, atol=0.01)
