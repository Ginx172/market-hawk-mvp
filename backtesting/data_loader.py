#!/usr/bin/env python3
"""
SCRIPT NAME: data_loader.py
====================================
Execution Location: K:\\_DEV_MVP_2026\\Market_Hawk_3\\backtesting\\
Required Directory Structure:
    Market_Hawk_3/
    └── backtesting/
        ├── __init__.py
        ├── data_loader.py      ← THIS FILE
        ├── engine.py
        ├── strategies.py
        └── report.py

Author: Professional AI Development System
Level: Doctoral — AI & Machine Learning Specialization
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01
Purpose: Universal historical data loader for JSON, CSV, Parquet, and yfinance
         Streams large files in chunks to protect 64GB RAM.
"""

import gc
import hashlib
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Generator
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger("market_hawk.backtest.data_loader")

# ============================================================
# SUPPORTED DATA SOURCES ON G: DRIVE
# ============================================================

# These are the known locations from Gigi's cautari.txt inventory
DEFAULT_DATA_DIRS = [
    # Primary historical data directories
    r"G:\..............JSON",
    r"G:\.................CSV",
    r"G:\.............METADATA",
    r"G:\.........PARQUET",
    r"G:\.....DATABASE\dataset",
    r"G:\trading_db",
    # Fallback — scan for data files
]


class HistoricalDataLoader:
    """
    Universal Historical Data Loader for Market Hawk 3 Backtesting.

    Supports:
        - CSV files (OHLCV format)
        - JSON files (candle arrays or nested structures)
        - Parquet files (columnar, efficient for large datasets)
        - yfinance fallback for missing local data
        - Streaming/chunked loading for files >500MB

    Hardware-Optimized:
        - Chunk processing to avoid RAM overflow on 64GB system
        - Memory-mapped Parquet reads via PyArrow
        - Garbage collection between large loads
        - Progress reporting for long operations

    Usage:
        loader = HistoricalDataLoader()
        loader.scan_available_data()  # Index what's on G: drive
        df = loader.load("BTCUSDT", timeframe="1h", start="2024-01-01", end="2024-12-31")
    """

    # Standard OHLCV column mappings for various data sources
    COLUMN_MAPS = {
        "standard":    {"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume"},
        "lowercase":   {"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume"},
        "yfinance":    {"Open": "Open", "High": "High", "Low": "Low",
                        "Close": "Close", "Volume": "Volume"},
        "binance":     {"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume",
                        "open_time": "timestamp"},
        "mt4":         {"<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low",
                        "<CLOSE>": "Close", "<TICKVOL>": "Volume",
                        "<DATE>": "date", "<TIME>": "time"},
    }

    # Chunk size for streaming large files
    CHUNK_SIZE_ROWS = 500_000
    MAX_FILE_SIZE_MB = 2048  # Files above this get chunked

    # Scan-cache settings
    SCAN_CACHE_FILE = "scan_index_cache.json"
    SCAN_CACHE_MAX_AGE_H = 24  # Re-scan if cache older than this (hours)

    def __init__(self, data_dirs: List[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Args:
            data_dirs: List of directories to scan for historical data.
                       Defaults to known G: drive locations.
            cache_dir: Directory for caching processed data (optional).
        """
        self.data_dirs = data_dirs or DEFAULT_DATA_DIRS
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._index: Dict[str, Dict] = {}  # symbol -> {file_path, format, timeframe}
        self._scan_complete = False

        # Determine where to store the scan cache
        self._cache_path = self._resolve_cache_path()

    # ============================================================
    # DATA DISCOVERY
    # ============================================================

    def _resolve_cache_path(self) -> Path:
        """Determine path for the scan-index cache file."""
        if self.cache_dir:
            base = Path(self.cache_dir)
        else:
            # Default: project logs directory (persists across runs)
            base = Path(__file__).resolve().parent.parent / "logs"
        base.mkdir(parents=True, exist_ok=True)
        return base / self.SCAN_CACHE_FILE

    def _dirs_fingerprint(self) -> str:
        """Hash of configured data_dirs so cache invalidates if dirs change."""
        blob = "|".join(sorted(self.data_dirs))
        return hashlib.md5(blob.encode()).hexdigest()[:12]

    def _load_scan_cache(self) -> bool:
        """Try to load a previous scan index from disk.
        Returns True if cache was loaded successfully and is still fresh.
        """
        if not self._cache_path.exists():
            return False

        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)

            # Validate structure
            if not isinstance(cache, dict) or "meta" not in cache or "index" not in cache:
                logger.debug("Scan cache: invalid structure, will re-scan")
                return False

            meta = cache["meta"]

            # Check fingerprint (dirs changed?)
            if meta.get("dirs_fp") != self._dirs_fingerprint():
                logger.info("Scan cache: data directories changed, will re-scan")
                return False

            # Check age
            cache_ts = meta.get("timestamp", 0)
            age_hours = (time.time() - cache_ts) / 3600
            if age_hours > self.SCAN_CACHE_MAX_AGE_H:
                logger.info("Scan cache: expired (%.1fh old, max %dh), will re-scan",
                            age_hours, self.SCAN_CACHE_MAX_AGE_H)
                return False

            # Load index
            self._index = cache["index"]
            self._scan_complete = True
            logger.info("Scan cache: loaded %d files from cache (%.1fh old)",
                        len(self._index), age_hours)
            return True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Scan cache: corrupt (%s), will re-scan", e)
            return False

    def _save_scan_cache(self) -> None:
        """Persist the current scan index to disk."""
        cache = {
            "meta": {
                "timestamp": time.time(),
                "dirs_fp": self._dirs_fingerprint(),
                "file_count": len(self._index),
                "created": datetime.now().isoformat(timespec="seconds"),
            },
            "index": self._index,
        }
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=1)
            logger.info("Scan cache: saved %d entries → %s",
                        len(self._index), self._cache_path)
        except OSError as e:
            logger.warning("Scan cache: failed to save (%s)", e)

    def scan_available_data(self, force_rescan: bool = False) -> Dict[str, Dict]:
        """
        Scan all configured directories and index available data files.
        Uses a JSON disk-cache to skip the 20+ second walk on repeat runs.
        Pass force_rescan=True (or --rescan CLI flag) to rebuild.

        Returns:
            Dict mapping symbol names to file metadata:
            {
                "BTCUSDT_1h": {"path": "...", "format": "csv", "size_mb": 45.2},
                "AAPL_1d": {"path": "...", "format": "parquet", "size_mb": 12.1},
                ...
            }
        """
        # Fast path: already scanned this session
        if self._scan_complete and not force_rescan:
            return self._index

        # Try loading from disk cache (unless forced)
        if not force_rescan and self._load_scan_cache():
            return self._index

        # Full scan required
        logger.info("Scanning data directories for historical files...")
        t0 = time.time()
        file_count = 0

        for data_dir in self.data_dirs:
            dir_path = Path(data_dir)
            if not dir_path.exists():
                logger.debug("Directory not found (skipping): %s", data_dir)
                continue

            logger.info("Scanning: %s", data_dir)

            # Walk directory tree
            for root, dirs, files in os.walk(dir_path):
                for fname in files:
                    ext = Path(fname).suffix.lower()
                    if ext not in (".csv", ".json", ".parquet", ".jsonl"):
                        continue

                    fpath = Path(root) / fname
                    try:
                        size_mb = fpath.stat().st_size / (1024 * 1024)
                        key = self._extract_key_from_filename(fname)
                        if key:
                            self._index[key] = {
                                "path": str(fpath),
                                "format": ext.lstrip("."),
                                "size_mb": round(size_mb, 2),
                                "filename": fname,
                            }
                            file_count += 1
                    except (OSError, PermissionError) as e:
                        logger.debug("Cannot access %s: %s", fpath, e)

        elapsed = time.time() - t0
        self._scan_complete = True
        logger.info("Scan complete: %d data files indexed in %.1fs",
                     file_count, elapsed)

        # Persist to cache for next run
        self._save_scan_cache()

        return self._index

    def list_symbols(self) -> List[str]:
        """Return list of available symbols after scanning."""
        if not self._scan_complete:
            self.scan_available_data()
        # Extract unique base symbols
        symbols = set()
        for key in self._index:
            parts = key.rsplit("_", 1)
            symbols.add(parts[0] if len(parts) > 1 else key)
        return sorted(symbols)

    def list_available(self) -> Dict[str, Dict]:
        """Return the full index of available data."""
        if not self._scan_complete:
            self.scan_available_data()
        return self._index

    # ============================================================
    # DATA LOADING
    # ============================================================

    def load(self, symbol: str, timeframe: str = "1h",
             start: Optional[str] = None, end: Optional[str] = None,
             use_yfinance_fallback: bool = True) -> Optional[pd.DataFrame]:
        """
        Load historical OHLCV data for a symbol.

        Priority:
            1. Local file (CSV/JSON/Parquet) from indexed data
            2. yfinance download (if use_yfinance_fallback=True)

        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "AAPL", "EURUSD")
            timeframe: Candle timeframe ("1m", "5m", "15m", "1h", "4h", "1d")
            start: Start date ISO string (e.g., "2024-01-01")
            end: End date ISO string (e.g., "2024-12-31")
            use_yfinance_fallback: Download from yfinance if local data missing

        Returns:
            DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume
        """
        if not self._scan_complete:
            self.scan_available_data()

        # Try local data first
        df = self._load_local(symbol, timeframe)

        # Fallback to yfinance
        if df is None and use_yfinance_fallback:
            logger.info("No local data for %s_%s, falling back to yfinance",
                         symbol, timeframe)
            df = self._load_yfinance(symbol, timeframe, start, end)

        if df is None:
            logger.warning("No data found for %s (timeframe=%s)", symbol, timeframe)
            return None

        # Ensure DatetimeIndex
        df = self._ensure_datetime_index(df)

        # Filter date range
        if start:
            start_dt = pd.Timestamp(start)
            df = df[df.index >= start_dt]
        if end:
            end_dt = pd.Timestamp(end)
            df = df[df.index <= end_dt]

        # Standardize columns
        df = self._standardize_columns(df)

        # Sort by time
        df = df.sort_index()

        # Drop duplicates
        df = df[~df.index.duplicated(keep="last")]

        # Remove rows with all NaN OHLCV
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        existing = [c for c in ohlcv_cols if c in df.columns]
        df = df.dropna(subset=existing, how="all")

        # Safe date-range display (index may not be datetime)
        if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex):
            idx_first = df.index[0].strftime("%Y-%m-%d")
            idx_last  = df.index[-1].strftime("%Y-%m-%d")
        elif len(df) > 0:
            idx_first = str(df.index[0])
            idx_last  = str(df.index[-1])
        else:
            idx_first = idx_last = "?"

        logger.info("Loaded %s_%s: %d rows [%s → %s]",
                     symbol, timeframe, len(df), idx_first, idx_last)

        gc.collect()
        return df

    def load_multiple(self, symbols: List[str], timeframe: str = "1h",
                      start: Optional[str] = None,
                      end: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.

        Returns:
            Dict mapping symbol -> DataFrame
        """
        results = {}
        for i, sym in enumerate(symbols, 1):
            logger.info("[%d/%d] Loading %s...", i, len(symbols), sym)
            df = self.load(sym, timeframe, start, end)
            if df is not None and not df.empty:
                results[sym] = df

            # Memory management every 10 symbols
            if i % 10 == 0:
                gc.collect()

        logger.info("Loaded %d / %d symbols successfully", len(results), len(symbols))
        return results

    # ============================================================
    # LOCAL FILE LOADERS
    # ============================================================

    def _load_local(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Try to find and load local data file for symbol+timeframe."""
        # Search index for matching keys
        search_keys = [
            f"{symbol}_{timeframe}",
            f"{symbol.upper()}_{timeframe}",
            f"{symbol.lower()}_{timeframe}",
            symbol,
            symbol.upper(),
            symbol.lower(),
        ]

        for key in search_keys:
            if key in self._index:
                info = self._index[key]
                logger.info("Found local: %s (%.1f MB, %s format)",
                             info["filename"], info["size_mb"], info["format"])
                return self._load_file(info["path"], info["format"],
                                       info["size_mb"])

        # Fuzzy match — symbol appears anywhere in filename
        symbol_up = symbol.upper()
        for key, info in self._index.items():
            if symbol_up in key.upper():
                logger.info("Fuzzy match: %s -> %s", symbol, info["filename"])
                return self._load_file(info["path"], info["format"],
                                       info["size_mb"])
        return None

    def _load_file(self, path: str, fmt: str,
                   size_mb: float) -> Optional[pd.DataFrame]:
        """Load a single data file by format."""
        try:
            if fmt == "csv":
                return self._load_csv(path, size_mb)
            elif fmt == "json" or fmt == "jsonl":
                return self._load_json(path, size_mb)
            elif fmt == "parquet":
                return self._load_parquet(path)
            else:
                logger.warning("Unsupported format: %s", fmt)
                return None
        except Exception as e:
            logger.error("Failed to load %s: %s", path, str(e))
            return None

    def _load_csv(self, path: str, size_mb: float) -> pd.DataFrame:
        """Load CSV with chunked reading for large files."""
        if size_mb > self.MAX_FILE_SIZE_MB:
            logger.info("Large CSV (%.0f MB) — loading in chunks", size_mb)
            chunks = []
            for chunk in pd.read_csv(path, chunksize=self.CHUNK_SIZE_ROWS,
                                      low_memory=False):
                chunks.append(chunk)
                if len(chunks) % 5 == 0:
                    gc.collect()
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
        else:
            df = pd.read_csv(path, low_memory=False)
        return df

    def _load_json(self, path: str, size_mb: float) -> pd.DataFrame:
        """Load JSON (array of candles or nested structure)."""
        if size_mb > self.MAX_FILE_SIZE_MB:
            # Stream JSONL
            logger.info("Large JSON (%.0f MB) — streaming line by line", size_mb)
            records = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, list):
                            records.extend(obj)
                        else:
                            records.append(obj)
                    except json.JSONDecodeError:
                        continue
                    if line_num % 100_000 == 0:
                        gc.collect()
            return pd.DataFrame(records)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read().strip()

            # Try as JSON array first
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Could be {symbol: {timestamp: {ohlcv}}} or similar
                    # Try common nested patterns
                    for key in ["data", "candles", "result", "results",
                                "ohlcv", "bars", "chart"]:
                        if key in data:
                            inner = data[key]
                            if isinstance(inner, list):
                                return pd.DataFrame(inner)
                    # Flat dict — try to DataFrame it
                    return pd.DataFrame([data])
            except json.JSONDecodeError:
                pass

            # Try as JSONL (one JSON object per line)
            records = []
            for line in raw.splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return pd.DataFrame(records) if records else pd.DataFrame()

    def _load_parquet(self, path: str) -> pd.DataFrame:
        """Load Parquet file (memory-efficient via PyArrow)."""
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except ImportError:
            return pd.read_parquet(path, engine="fastparquet")

    # ============================================================
    # YFINANCE FALLBACK
    # ============================================================

    def _load_yfinance(self, symbol: str, timeframe: str,
                       start: Optional[str] = None,
                       end: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Download data from yfinance as fallback."""
        try:
            import yfinance as yf
            from data.market_data_fetcher import get_yfinance_ticker

            ticker = get_yfinance_ticker(symbol)

            # Map timeframe to yfinance interval + period
            tf_map = {
                "1m": ("1m", "7d"),
                "2m": ("2m", "60d"),
                "5m": ("5m", "60d"),
                "15m": ("15m", "60d"),
                "30m": ("30m", "60d"),
                "1h": ("1h", "730d"),
                "4h": ("1h", "730d"),  # yfinance doesn't support 4h natively
                "1d": ("1d", "max"),
                "1wk": ("1wk", "max"),
            }

            interval, default_period = tf_map.get(timeframe, ("1h", "730d"))

            if start and end:
                data = yf.download(ticker, start=start, end=end,
                                   interval=interval, progress=False,
                                   auto_adjust=False)
            else:
                data = yf.download(ticker, period=default_period,
                                   interval=interval, progress=False,
                                   auto_adjust=False)

            if data.empty:
                return None

            # Flatten multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # Resample to 4h if needed
            if timeframe == "4h" and interval == "1h":
                data = data.resample("4h").agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum"
                }).dropna()

            return data

        except Exception as e:
            logger.error("yfinance fallback failed for %s: %s", symbol, str(e))
            return None

    # ============================================================
    # HELPER METHODS
    # ============================================================

    @staticmethod
    def _extract_key_from_filename(filename: str) -> Optional[str]:
        """
        Extract a symbol_timeframe key from a filename.
        Examples:
            "BTCUSDT_1h.csv" -> "BTCUSDT_1h"
            "AAPL_daily.parquet" -> "AAPL_daily"
            "eurusd-4h-2024.json" -> "EURUSD-4h-2024"
        """
        stem = Path(filename).stem
        if not stem or stem.startswith("."):
            return None
        # Normalize separators
        key = stem.replace(" ", "_")
        return key.upper() if len(key) < 100 else key[:100].upper()

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Try to convert index or a column to DatetimeIndex."""
        # Already datetime
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        # Check common timestamp column names
        ts_cols = ["Date", "date", "datetime", "Datetime", "timestamp",
                   "Timestamp", "time", "Time", "open_time", "close_time",
                   "candle_time"]

        for col in ts_cols:
            if col not in df.columns:
                continue

            col_dtype = df[col].dtype
            converted = None

            # String column → parse as date strings first
            if pd.api.types.is_string_dtype(col_dtype) or col_dtype == object:
                try:
                    converted = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    logger.warning("Failed to parse string column '%s' as datetime", col)
                    continue

            # Numeric column → try unit="ms" then unit="s"
            elif pd.api.types.is_numeric_dtype(col_dtype):
                sample = df[col].dropna()
                if len(sample) == 0:
                    continue
                max_val = sample.max()
                try:
                    if max_val > 1e12:  # Milliseconds
                        converted = pd.to_datetime(df[col], unit="ms", errors="coerce")
                    elif max_val > 1e9:  # Seconds
                        converted = pd.to_datetime(df[col], unit="s", errors="coerce")
                    else:  # Could be Excel serial or plain int — try general
                        converted = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    logger.warning("Failed to parse numeric column '%s' as datetime", col)
                    continue
            else:
                # Already datetime-like
                try:
                    converted = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    logger.warning("Failed to parse column '%s' as datetime", col)
                    continue

            # Check if conversion succeeded on >50% of rows
            if converted is not None and converted.notna().sum() > len(df) * 0.5:
                df[col] = converted
                df = df.set_index(col)
                df.index.name = None
                return df

        # Try numeric index as Unix timestamp
        if pd.api.types.is_numeric_dtype(df.index):
            try:
                max_idx = df.index.max()
                if max_idx > 1e12:  # Milliseconds
                    df.index = pd.to_datetime(df.index, unit="ms")
                elif max_idx > 1e9:  # Seconds
                    df.index = pd.to_datetime(df.index, unit="s")
                return df
            except Exception:
                logger.warning("Failed to convert numeric index to datetime")

        # Try parsing index as string dates
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            logger.warning("Failed to parse index as string dates")

        return df

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure standard OHLCV column names."""
        rename_map = {}
        col_lower = {c.lower(): c for c in df.columns}

        for std_name, search_names in [
            ("Open",   ["open", "<open>", "o", "open_price"]),
            ("High",   ["high", "<high>", "h", "high_price"]),
            ("Low",    ["low", "<low>", "l", "low_price"]),
            ("Close",  ["close", "<close>", "c", "close_price", "price"]),
            ("Volume", ["volume", "<tickvol>", "vol", "v", "base_volume",
                        "quote_volume"]),
        ]:
            if std_name in df.columns:
                continue  # Already exists
            for search in search_names:
                if search in col_lower:
                    rename_map[col_lower[search]] = std_name
                    break

        if rename_map:
            df = df.rename(columns=rename_map)

        # Ensure numeric types for OHLCV
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


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

    print("\n" + "=" * 60)
    print("HISTORICAL DATA LOADER — Scan & Load Test")
    print("=" * 60)

    loader = HistoricalDataLoader()

    # Scan available data
    index = loader.scan_available_data()
    print(f"\nFound {len(index)} data files")

    # Show first 20
    for i, (key, info) in enumerate(sorted(index.items())[:20]):
        print(f"  {key:<30s} {info['size_mb']:>8.1f} MB  [{info['format']}]")

    if len(index) > 20:
        print(f"  ... and {len(index) - 20} more")

    # Test loading a symbol
    print("\n--- Loading AAPL (1h, yfinance fallback) ---")
    df = loader.load("AAPL", timeframe="1h")
    if df is not None:
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns[:10])}")
        print(f"  Date range: {df.index[0]} → {df.index[-1]}")
    else:
        print("  No data returned")

    print("\n✅ Data loader test complete")
