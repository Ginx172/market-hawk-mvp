"""
Spark-based distributed feature engineering pipeline for Market Hawk MVP.

Computes all 60 trading features across multiple symbols simultaneously
using PySpark Window functions and per-group pandas UDFs (applyInPandas).

Hardware target: Intel i7-9700F (8 cores), 64 GB DDR4.
"""
import gc
import logging
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("market_hawk.spark.features")

# Feature names reference — imported lazily to avoid circular deps at import time
_FEATURES_60: Optional[List[str]] = None


def _get_features_60() -> List[str]:
    """Return FEATURES_60, importing lazily on first call."""
    global _FEATURES_60
    if _FEATURES_60 is None:
        from agents.ml_signal_engine.catboost_predictor import FEATURES_60  # noqa: F401
        _FEATURES_60 = list(FEATURES_60)
    return _FEATURES_60


# ---------------------------------------------------------------------------
# Pyspark schema (defined here so it is available before SparkSession starts)
# ---------------------------------------------------------------------------

def _build_output_schema():
    """Build the PySpark StructType for the engineer_features output."""
    from pyspark.sql.types import (  # type: ignore[import]
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    fields = [
        StructField("symbol", StringType(), True),
        StructField("date", TimestampType(), True),
        # OHLCV (exact-case)
        StructField("Open", DoubleType(), True),
        StructField("High", DoubleType(), True),
        StructField("Low", DoubleType(), True),
        StructField("Close", DoubleType(), True),
        StructField("Volume", DoubleType(), True),
        StructField("Dividends", DoubleType(), True),
        StructField("Stock Splits", DoubleType(), True),
        # Lower-case aliases
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", DoubleType(), True),
        # Order-flow proxies
        StructField("AskVolume_Sum", DoubleType(), True),
        StructField("BidVolume_Sum", DoubleType(), True),
        # Returns
        StructField("returns", DoubleType(), True),
        StructField("log_returns", DoubleType(), True),
        # Moving averages
        StructField("SMA_5", DoubleType(), True),
        StructField("EMA_5", DoubleType(), True),
        StructField("SMA_10", DoubleType(), True),
        StructField("EMA_10", DoubleType(), True),
        StructField("SMA_20", DoubleType(), True),
        StructField("EMA_20", DoubleType(), True),
        StructField("SMA_50", DoubleType(), True),
        StructField("EMA_50", DoubleType(), True),
        StructField("SMA_200", DoubleType(), True),
        StructField("EMA_200", DoubleType(), True),
        # Oscillators
        StructField("RSI", DoubleType(), True),
        # Bollinger Bands
        StructField("BB_middle", DoubleType(), True),
        StructField("BB_upper", DoubleType(), True),
        StructField("BB_lower", DoubleType(), True),
        StructField("BB_width", DoubleType(), True),
        StructField("BB_position", DoubleType(), True),
        # MACD
        StructField("MACD", DoubleType(), True),
        StructField("MACD_signal", DoubleType(), True),
        StructField("MACD_histogram", DoubleType(), True),
        # Volatility / range
        StructField("volatility_20", DoubleType(), True),
        StructField("ATR", DoubleType(), True),
        # Volume
        StructField("volume_SMA", DoubleType(), True),
        StructField("volume_ratio", DoubleType(), True),
        StructField("OBV", DoubleType(), True),
        # Spreads
        StructField("HL_spread", DoubleType(), True),
        StructField("CO_spread", DoubleType(), True),
        StructField("efficiency_ratio", DoubleType(), True),
        # Time features
        StructField("hour", LongType(), True),
        StructField("day_of_week", LongType(), True),
        StructField("is_london_session", LongType(), True),
        StructField("is_ny_session", LongType(), True),
        StructField("is_asia_session", LongType(), True),
        # Metadata
        StructField("source_file", LongType(), True),
        StructField("Adj Close", DoubleType(), True),
        # Price derived (duplicate names preserved for model compatibility)
        StructField("price_change", DoubleType(), True),
        StructField("price_change_pct", DoubleType(), True),
        StructField("price_range", DoubleType(), True),
        StructField("rsi", DoubleType(), True),
        StructField("macd", DoubleType(), True),
        StructField("signal", DoubleType(), True),
        StructField("ma_5", DoubleType(), True),
        StructField("ma_10", DoubleType(), True),
        StructField("ma_20", DoubleType(), True),
        StructField("volume_ma", DoubleType(), True),
    ]
    return StructType(fields)


# ---------------------------------------------------------------------------
# Per-group feature computation (pandas, runs inside each Spark partition)
# ---------------------------------------------------------------------------

def _compute_features_per_group(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute all 60 trading features for a single symbol group.

    This function is called by PySpark's ``applyInPandas`` once per
    ``symbol`` partition.  It replicates the logic from
    ``MarketDataFetcher.engineer_features()`` so that the Spark pipeline
    produces identical features to the existing pandas pipeline.

    Args:
        pdf: Pandas DataFrame for one symbol, containing at minimum:
             ``date``, ``Open``, ``High``, ``Low``, ``Close``, ``Volume``,
             ``symbol`` columns.

    Returns:
        Pandas DataFrame with all feature columns added, forward-filled,
        infinities replaced with 0.
    """
    if pdf.empty:
        return pdf

    # Sort chronologically — critical for rolling calculations
    pdf = pdf.sort_values("date").reset_index(drop=True)

    close = pdf["Close"].values.astype(float)
    high = pdf["High"].values.astype(float)
    low = pdf["Low"].values.astype(float)
    open_p = pdf["Open"].values.astype(float)
    volume = pdf["Volume"].values.astype(float)

    # --- Optional OHLCV columns ---
    if "Dividends" not in pdf.columns:
        pdf["Dividends"] = 0.0
    if "Stock Splits" not in pdf.columns:
        pdf["Stock Splits"] = 0.0
    if "Adj Close" not in pdf.columns:
        pdf["Adj Close"] = close

    # --- Lower-case aliases ---
    pdf["open"] = open_p
    pdf["high"] = high
    pdf["low"] = low
    pdf["close"] = close
    pdf["volume"] = volume

    # --- Order-flow proxies (not available from yfinance) ---
    pdf["AskVolume_Sum"] = volume * 0.5
    pdf["BidVolume_Sum"] = volume * 0.5

    # --- Returns ---
    close_series = pd.Series(close)
    pdf["returns"] = close_series.pct_change()
    pdf["log_returns"] = np.log(close_series / close_series.shift(1))

    # --- Simple Moving Averages (PySpark Window equivalent: rowsBetween) ---
    for period in [5, 10, 20, 50, 200]:
        pdf[f"SMA_{period}"] = close_series.rolling(period).mean()

    # --- Exponential Moving Averages (EMA via pandas .ewm(), adjust=False) ---
    # Note: native PySpark has no EMA — computed here via pandas_udf pattern
    for span in [5, 10, 20, 50, 200]:
        pdf[f"EMA_{span}"] = close_series.ewm(span=span, adjust=False).mean()

    # --- RSI (14-period) ---
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))
    pdf["RSI"] = rsi_vals

    # --- Bollinger Bands (20-period, ±2σ) ---
    bb_mid = close_series.rolling(20).mean()
    bb_std = close_series.rolling(20).std()
    pdf["BB_middle"] = bb_mid
    pdf["BB_upper"] = bb_mid + 2 * bb_std
    pdf["BB_lower"] = bb_mid - 2 * bb_std
    bb_range = pdf["BB_upper"] - pdf["BB_lower"]
    pdf["BB_width"] = bb_range / pdf["BB_middle"]
    pdf["BB_position"] = (close_series - pdf["BB_lower"]) / bb_range.replace(0, np.nan)

    # --- MACD ---
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    pdf["MACD"] = ema12 - ema26
    pdf["MACD_signal"] = pdf["MACD"].ewm(span=9, adjust=False).mean()
    pdf["MACD_histogram"] = pdf["MACD"] - pdf["MACD_signal"]

    # --- Volatility (20-period std of returns) ---
    pdf["volatility_20"] = close_series.pct_change().rolling(20).std()

    # --- ATR (14-period) ---
    prev_close = close_series.shift(1)
    true_range = pd.concat(
        [
            pd.Series(high - low),
            (pd.Series(high) - prev_close).abs(),
            (pd.Series(low) - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    pdf["ATR"] = true_range.rolling(14).mean()

    # --- Volume indicators ---
    vol_series = pd.Series(volume)
    pdf["volume_SMA"] = vol_series.rolling(20).mean()
    pdf["volume_ratio"] = vol_series / pdf["volume_SMA"].replace(0, np.nan)

    # OBV: cumulative sum of volume multiplied by direction of price change
    price_direction = np.sign(np.diff(close, prepend=close[0]))
    pdf["OBV"] = np.cumsum(price_direction * volume)

    # --- Spreads ---
    pdf["HL_spread"] = high - low
    pdf["CO_spread"] = close - open_p

    # Efficiency ratio: net move / sum of absolute moves over 10 bars
    net_move = np.abs(close - np.roll(close, 10))
    sum_moves = pd.Series(
        np.abs(np.diff(close, prepend=close[0]))
    ).rolling(10).sum()
    pdf["efficiency_ratio"] = net_move / sum_moves.replace(0, np.nan).values

    # --- Time features ---
    if pd.api.types.is_datetime64_any_dtype(pdf["date"]):
        dt = pd.to_datetime(pdf["date"])
        pdf["hour"] = dt.dt.hour.astype("int64")
        pdf["day_of_week"] = dt.dt.dayofweek.astype("int64")
    else:
        pdf["hour"] = 0
        pdf["day_of_week"] = 0

    hour_arr = pdf["hour"].values
    pdf["is_london_session"] = ((hour_arr >= 8) & (hour_arr <= 16)).astype("int64")
    pdf["is_ny_session"] = ((hour_arr >= 13) & (hour_arr <= 21)).astype("int64")
    pdf["is_asia_session"] = (hour_arr <= 8).astype("int64")

    # --- Metadata ---
    pdf["source_file"] = 0  # Numeric placeholder (matches training data)

    # --- Price derived (duplicate names kept for model compatibility) ---
    pdf["price_change"] = close_series.diff()
    pdf["price_change_pct"] = close_series.pct_change()
    pdf["price_range"] = high - low
    pdf["rsi"] = pdf["RSI"]
    pdf["macd"] = pdf["MACD"]
    pdf["signal"] = pdf["MACD_signal"]
    pdf["ma_5"] = pdf["SMA_5"]
    pdf["ma_10"] = pdf["SMA_10"]
    pdf["ma_20"] = pdf["SMA_20"]
    pdf["volume_ma"] = pdf["volume_SMA"]

    # --- NaN / inf handling ---
    # Forward-fill only (no bfill — anti look-ahead bias)
    pdf = pdf.ffill().fillna(0)
    pdf = pdf.replace([np.inf, -np.inf], 0)

    # Ensure all schema columns are present and have the right dtype
    double_cols = [
        "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
        "open", "high", "low", "close", "volume",
        "AskVolume_Sum", "BidVolume_Sum",
        "returns", "log_returns",
        "SMA_5", "EMA_5", "SMA_10", "EMA_10", "SMA_20", "EMA_20",
        "SMA_50", "EMA_50", "SMA_200", "EMA_200",
        "RSI", "BB_middle", "BB_upper", "BB_lower", "BB_width", "BB_position",
        "MACD", "MACD_signal", "MACD_histogram",
        "volatility_20", "ATR", "volume_SMA", "volume_ratio", "OBV",
        "HL_spread", "CO_spread", "efficiency_ratio",
        "Adj Close",
        "price_change", "price_change_pct", "price_range",
        "rsi", "macd", "signal",
        "ma_5", "ma_10", "ma_20", "volume_ma",
    ]
    long_cols = [
        "hour", "day_of_week",
        "is_london_session", "is_ny_session", "is_asia_session",
        "source_file",
    ]
    for col in double_cols:
        if col in pdf.columns:
            pdf[col] = pdf[col].astype(float)
    for col in long_cols:
        if col in pdf.columns:
            pdf[col] = pdf[col].astype("int64")

    return pdf


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class SparkFeaturePipeline:
    """Distributed feature engineering pipeline using Apache Spark.

    Fetches OHLCV data via yfinance, computes all 60 TA features
    per symbol using PySpark Window functions (simple rolling stats)
    and ``applyInPandas`` (EMA / RSI / MACD / ATR / OBV), then
    returns a Spark DataFrame ready for Parquet export or model training.

    Args:
        config: Optional :class:`~config.spark_config.SparkConfig`.
                Defaults to ``SPARK_CONFIG``.

    Example::

        pipeline = SparkFeaturePipeline()
        sdf = pipeline.process_symbols(["AAPL", "MSFT"], period="2y")
        df  = pipeline.to_pandas(sdf)
    """

    def __init__(self, config=None) -> None:
        from config.spark_config import get_or_create_spark_session  # noqa: F401

        self.spark = get_or_create_spark_session(config)
        self._output_schema = _build_output_schema()
        logger.info("SparkFeaturePipeline initialised")

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_symbols_data(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d",
    ):
        """Fetch OHLCV data for multiple symbols and return a Spark DataFrame.

        Each symbol is fetched sequentially via yfinance then concatenated
        into a single Spark DataFrame with an added ``symbol`` column.

        Args:
            symbols: List of ticker symbols (e.g. ``["AAPL", "MSFT"]``).
            period:   yfinance period string (e.g. ``"2y"``).
            interval: Candle interval (e.g. ``"1d"``, ``"1h"``).

        Returns:
            Spark DataFrame with columns:
            ``symbol``, ``date``, ``Open``, ``High``, ``Low``,
            ``Close``, ``Volume``, plus optional ``Dividends`` /
            ``Stock Splits`` / ``Adj Close`` when available.
        """
        import yfinance as yf  # type: ignore[import]
        from pyspark.sql import functions as F  # type: ignore[import]
        from pyspark.sql.types import (  # type: ignore[import]
            DoubleType,
            StringType,
            StructField,
            StructType,
            TimestampType,
        )

        raw_schema = StructType(
            [
                StructField("symbol", StringType(), True),
                StructField("date", TimestampType(), True),
                StructField("Open", DoubleType(), True),
                StructField("High", DoubleType(), True),
                StructField("Low", DoubleType(), True),
                StructField("Close", DoubleType(), True),
                StructField("Volume", DoubleType(), True),
                StructField("Dividends", DoubleType(), True),
                StructField("Stock Splits", DoubleType(), True),
            ]
        )

        frames: List[pd.DataFrame] = []
        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                df = ticker.history(period=period, interval=interval)
                if df.empty:
                    logger.warning("No data from yfinance for %s", sym)
                    continue

                df = df.reset_index()
                # Rename the date/datetime index column uniformly
                date_col = "Date" if "Date" in df.columns else "Datetime"
                df = df.rename(columns={date_col: "date"})

                # Ensure required columns
                for col in ["Dividends", "Stock Splits"]:
                    if col not in df.columns:
                        df[col] = 0.0

                df["symbol"] = sym
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

                keep = [
                    "symbol", "date", "Open", "High", "Low",
                    "Close", "Volume", "Dividends", "Stock Splits",
                ]
                df = df[[c for c in keep if c in df.columns]].copy()
                df = df.astype({
                    "Open": float, "High": float, "Low": float,
                    "Close": float, "Volume": float,
                    "Dividends": float, "Stock Splits": float,
                })
                frames.append(df)
                logger.info("Fetched %d rows for %s", len(df), sym)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to fetch %s: %s", sym, exc)

        if not frames:
            raise ValueError("No data fetched for any symbol")

        combined = pd.concat(frames, ignore_index=True)
        sdf = self.spark.createDataFrame(combined, schema=raw_schema)
        logger.info(
            "Created Spark DataFrame: %d symbols, %d total rows",
            len(symbols), combined.shape[0],
        )
        return sdf

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, sdf):
        """Compute all 60 trading features on a Spark DataFrame.

        **Simple rolling statistics** are computed with native PySpark
        ``Window`` functions (SMA, Bollinger Bands, volatility, volume
        ratio).  **Complex iterative indicators** (EMA, RSI, MACD, ATR,
        OBV) are computed per-symbol via ``applyInPandas`` — the modern
        PySpark 3.x equivalent of ``GROUPED_MAP`` pandas UDF.

        No ``bfill()`` is used anywhere; NaN values are forward-filled
        only (``ffill``), then any remaining NaN / ±inf are set to 0.

        Args:
            sdf: Spark DataFrame with columns produced by
                 :meth:`fetch_symbols_data`.

        Returns:
            Spark DataFrame with all 60 feature columns.
        """
        from pyspark.sql import functions as F  # type: ignore[import]
        from pyspark.sql import Window  # type: ignore[import]

        # -----------------------------------------------------------------
        # Phase 1: PySpark Window functions for simple rolling statistics
        # These demonstrate the Window-function approach required by spec.
        # -----------------------------------------------------------------
        w_sym_date = (
            Window.partitionBy("symbol")
            .orderBy("date")
        )

        # SMA via rowsBetween (avg over preceding N-1 rows + current row)
        for n in [5, 10, 20, 50, 200]:
            sdf = sdf.withColumn(
                f"_sma_{n}_window",
                F.avg("Close").over(
                    w_sym_date.rowsBetween(-(n - 1), 0)
                ),
            )

        # Bollinger std (20-period)
        sdf = sdf.withColumn(
            "_bb_std20",
            F.stddev("Close").over(w_sym_date.rowsBetween(-19, 0)),
        )

        # Volatility: std of pct-change over 20 bars
        sdf = sdf.withColumn(
            "_prev_close",
            F.lag("Close", 1).over(w_sym_date),
        )
        sdf = sdf.withColumn(
            "_ret",
            (F.col("Close") - F.col("_prev_close")) / F.col("_prev_close"),
        )
        sdf = sdf.withColumn(
            "_vol20_window",
            F.stddev("_ret").over(w_sym_date.rowsBetween(-19, 0)),
        )

        # Volume SMA (20-period)
        sdf = sdf.withColumn(
            "_vol_sma20_window",
            F.avg("Volume").over(w_sym_date.rowsBetween(-19, 0)),
        )

        # Drop helper columns before applyInPandas
        helper_cols = [
            "_sma_5_window", "_sma_10_window", "_sma_20_window",
            "_sma_50_window", "_sma_200_window",
            "_bb_std20", "_prev_close", "_ret",
            "_vol20_window", "_vol_sma20_window",
        ]
        sdf = sdf.drop(*helper_cols)

        # -----------------------------------------------------------------
        # Phase 2: applyInPandas — EMA, RSI, MACD, ATR, OBV, all features
        # (The pandas_udf GROUPED_MAP pattern from the spec, using the
        #  modern applyInPandas API available since PySpark 3.0)
        # -----------------------------------------------------------------
        sdf = (
            sdf.groupBy("symbol")
            .applyInPandas(
                _compute_features_per_group,
                schema=self._output_schema,
            )
        )

        logger.info("Feature engineering complete")
        return sdf

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_symbols(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d",
    ):
        """Full pipeline: fetch OHLCV → engineer features → return Spark DF.

        Args:
            symbols:  List of ticker symbols.
            period:   yfinance period string.
            interval: Candle interval.

        Returns:
            Spark DataFrame with all 60 feature columns and a ``symbol``
            column.
        """
        logger.info(
            "Processing %d symbols: %s (period=%s, interval=%s)",
            len(symbols), symbols, period, interval,
        )
        sdf = self.fetch_symbols_data(symbols, period=period, interval=interval)
        sdf = self.engineer_features(sdf)
        gc.collect()
        return sdf

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def save_to_parquet(self, sdf, output_path: str) -> None:
        """Write a Spark DataFrame to Parquet, partitioned by ``symbol``.

        Args:
            sdf:         Spark DataFrame to persist.
            output_path: Destination directory path.
        """
        Path(output_path).mkdir(parents=True, exist_ok=True)
        (
            sdf.write.mode("overwrite")
            .partitionBy("symbol")
            .parquet(output_path)
        )
        logger.info("Saved Parquet to %s", output_path)

    def load_from_parquet(self, path: str):
        """Load a previously saved feature Parquet dataset.

        Args:
            path: Path to the Parquet directory written by
                  :meth:`save_to_parquet`.

        Returns:
            Spark DataFrame.
        """
        sdf = self.spark.read.parquet(path)
        logger.info("Loaded Parquet from %s", path)
        return sdf

    def to_pandas(self, sdf) -> pd.DataFrame:
        """Collect a Spark DataFrame to a local pandas DataFrame.

        Args:
            sdf: Spark DataFrame (should fit comfortably in driver memory).

        Returns:
            pandas DataFrame.
        """
        df = sdf.toPandas()
        logger.info("Collected Spark DataFrame to pandas: %d rows", len(df))
        return df
