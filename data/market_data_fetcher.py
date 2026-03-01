"""
SCRIPT NAME: market_data_fetcher.py
====================================
Execution Location: market-hawk-mvp/data/
Purpose: Live market data fetcher + feature engineering (60 features)
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01

Fetches OHLCV data via yfinance/ccxt, calculates all 60 TA features
required by the ML Signal Engine models.

FEATURE SET (60 features - matches Modele_AI_80_Plus training):
    OHLCV (12): Open, High, Low, Close, Volume, Dividends, Stock Splits,
                open, high, low, close, volume
    Order Flow (2): AskVolume_Sum, BidVolume_Sum
    Returns (2): returns, log_returns
    Moving Averages (10): SMA/EMA for 5, 10, 20, 50, 200
    RSI (1): RSI
    Bollinger (5): BB_middle, BB_upper, BB_lower, BB_width, BB_position
    MACD (3): MACD, MACD_signal, MACD_histogram
    Volatility (4): volatility_20, ATR, volume_SMA, volume_ratio
    Volume (1): OBV
    Spreads (3): HL_spread, CO_spread, efficiency_ratio
    Time (5): hour, day_of_week, is_london_session, is_ny_session, is_asia_session
    Meta (1): source_file
    Price derived (7): Adj Close, price_change, price_change_pct, price_range,
                       rsi, macd, signal
    Simple MA (4): ma_5, ma_10, ma_20, volume_ma
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime

logger = logging.getLogger("market_hawk.data")


# ============================================================
# SYMBOL CATEGORIES
# ============================================================

SYMBOL_CATEGORIES = {
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    "tech_stocks": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "AMD", "AMZN"],
    "indices": ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^GDAXI"],
    "forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
    "commodities": ["GC=F", "SI=F", "CL=F"],  # Gold, Silver, Crude Oil
    "etf": ["SPY", "QQQ", "IWM", "GLD", "TLT"],
}

# Map user-friendly names to yfinance tickers
TICKER_MAP = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "OIL": "CL=F",
    "SP500": "^GSPC",
}


def get_yfinance_ticker(symbol: str) -> str:
    """Convert user symbol to yfinance ticker."""
    return TICKER_MAP.get(symbol.upper(), symbol)


def get_symbol_category(symbol: str) -> str:
    """Determine which category a symbol belongs to."""
    ticker = get_yfinance_ticker(symbol)
    for category, symbols in SYMBOL_CATEGORIES.items():
        if ticker in symbols:
            return category
    # Heuristic
    if "USD" in symbol.upper() or "=X" in symbol:
        return "forex"
    if symbol.startswith("^"):
        return "indices"
    return "tech_stocks"  # Default


# ============================================================
# DATA FETCHER
# ============================================================

class MarketDataFetcher:
    """
    Fetches live market data and calculates all 60 features
    required by the ML Signal Engine.

    Usage:
        fetcher = MarketDataFetcher()
        features_df = fetcher.fetch_and_engineer("AAPL", period="60d", interval="1h")
        latest_features = fetcher.get_latest_features("AAPL")
    """

    def __init__(self):
        self._cache = {}

    def fetch_ohlcv(self, symbol: str, period: str = "60d",
                    interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data via yfinance.

        Args:
            symbol: Trading symbol
            period: Data period (e.g., "60d", "6mo", "1y")
            interval: Candle interval (e.g., "1h", "4h", "1d")

        Returns:
            DataFrame with OHLCV columns or None
        """
        try:
            import yfinance as yf

            ticker = get_yfinance_ticker(symbol)
            logger.info("Fetching %s (%s) — period=%s, interval=%s",
                         symbol, ticker, period, interval)

            data = yf.download(ticker, period=period, interval=interval,
                               progress=False, auto_adjust=False)

            if data.empty:
                logger.warning("No data returned for %s", symbol)
                return None

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            logger.info("Fetched %d candles for %s", len(data), symbol)
            return data

        except Exception as e:
            logger.error("Failed to fetch %s: %s", symbol, str(e))
            return None

    def engineer_features(self, df: pd.DataFrame, symbol: str = "unknown") -> pd.DataFrame:
        """
        Calculate all 60 features from OHLCV data.
        Matches the feature set used in Modele_AI_80_Plus training.

        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns
            symbol: Symbol name for source_file field

        Returns:
            DataFrame with 60 feature columns
        """
        result = df.copy()

        # Save datetime index for time features, then reset to numeric
        dt_index = None
        if isinstance(result.index, pd.DatetimeIndex):
            dt_index = result.index
        result = result.reset_index(drop=True)

        # Ensure standard column names exist
        for col_lower, col_upper in [("open", "Open"), ("high", "High"),
                                       ("low", "Low"), ("close", "Close"),
                                       ("volume", "Volume")]:
            if col_upper in result.columns and col_lower not in result.columns:
                result[col_lower] = result[col_upper]
            elif col_lower in result.columns and col_upper not in result.columns:
                result[col_upper] = result[col_lower]

        close = result["Close"].values.astype(float)
        high = result["High"].values.astype(float)
        low = result["Low"].values.astype(float)
        open_p = result["Open"].values.astype(float)
        volume = result["Volume"].values.astype(float)

        # --- Dividends & Stock Splits (fill 0 if missing) ---
        if "Dividends" not in result.columns:
            result["Dividends"] = 0.0
        if "Stock Splits" not in result.columns:
            result["Stock Splits"] = 0.0
        if "Adj Close" not in result.columns:
            result["Adj Close"] = close

        # --- Order Flow proxies (not available from yfinance) ---
        result["AskVolume_Sum"] = volume * 0.5  # Placeholder
        result["BidVolume_Sum"] = volume * 0.5  # Placeholder

        # --- Returns ---
        result["returns"] = pd.Series(close).pct_change()
        result["log_returns"] = np.log(pd.Series(close) / pd.Series(close).shift(1))

        # --- Moving Averages ---
        for period in [5, 10, 20, 50, 200]:
            result[f"SMA_{period}"] = pd.Series(close).rolling(period).mean()
            result[f"EMA_{period}"] = pd.Series(close).ewm(span=period, adjust=False).mean()

        # --- RSI ---
        result["RSI"] = self._calc_rsi(close, 14)

        # --- Bollinger Bands ---
        bb_mid = pd.Series(close).rolling(20).mean()
        bb_std = pd.Series(close).rolling(20).std()
        result["BB_middle"] = bb_mid.values
        result["BB_upper"] = (bb_mid + 2 * bb_std).values
        result["BB_lower"] = (bb_mid - 2 * bb_std).values
        result["BB_width"] = (result["BB_upper"] - result["BB_lower"]) / result["BB_middle"]
        result["BB_position"] = (close - result["BB_lower"].values) / \
                                 (result["BB_upper"].values - result["BB_lower"].values)

        # --- MACD ---
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        result["MACD"] = ema12 - ema26
        result["MACD_signal"] = result["MACD"].ewm(span=9, adjust=False).mean()
        result["MACD_histogram"] = result["MACD"] - result["MACD_signal"]

        # --- Volatility ---
        result["volatility_20"] = pd.Series(close).pct_change().rolling(20).std()

        # ATR
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        result["ATR"] = pd.Series(tr).rolling(14).mean()

        # Volume indicators
        result["volume_SMA"] = pd.Series(volume).rolling(20).mean()
        result["volume_ratio"] = volume / result["volume_SMA"].values

        # OBV
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        result["OBV"] = obv

        # --- Spreads ---
        result["HL_spread"] = high - low
        result["CO_spread"] = close - open_p
        # Efficiency ratio
        net_move = np.abs(close - np.roll(close, 10))
        sum_moves = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(10).sum()
        result["efficiency_ratio"] = net_move / sum_moves.values

        # --- Time features ---
        if dt_index is not None:
            result["hour"] = dt_index.hour.values
            result["day_of_week"] = dt_index.dayofweek.values
        else:
            result["hour"] = 0
            result["day_of_week"] = 0

        result["is_london_session"] = result["hour"].apply(
            lambda h: 1 if 8 <= h <= 16 else 0)
        result["is_ny_session"] = result["hour"].apply(
            lambda h: 1 if 13 <= h <= 21 else 0)
        result["is_asia_session"] = result["hour"].apply(
            lambda h: 1 if 0 <= h <= 8 else 0)

        # --- Meta ---
        result["source_file"] = 0  # Numeric placeholder (was filename in training)

        # --- Price derived (duplicate names from training) ---
        result["price_change"] = pd.Series(close).diff()
        result["price_change_pct"] = pd.Series(close).pct_change()
        result["price_range"] = high - low
        result["rsi"] = result["RSI"]  # Duplicate name
        result["macd"] = result["MACD"]  # Duplicate name
        result["signal"] = result["MACD_signal"]  # Duplicate name

        # --- Simple MAs ---
        result["ma_5"] = pd.Series(close).rolling(5).mean()
        result["ma_10"] = pd.Series(close).rolling(10).mean()
        result["ma_20"] = pd.Series(close).rolling(20).mean()
        result["volume_ma"] = pd.Series(volume).rolling(20).mean()

        # Fill NaN from rolling calculations (first 200 rows have NaN from SMA_200)
        # Use forward fill then backward fill, then 0 for any remaining
        result = result.ffill().bfill().fillna(0)
        
        # Replace inf values
        result = result.replace([np.inf, -np.inf], 0)

        logger.info("Engineered %d features for %d rows", 60, len(result))
        return result

    def fetch_and_engineer(self, symbol: str, period: str = "60d",
                           interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Full pipeline: fetch data + calculate features.

        Returns:
            DataFrame with 60 features or None
        """
        df = self.fetch_ohlcv(symbol, period, interval)
        if df is None:
            return None

        features_df = self.engineer_features(df, symbol)
        self._cache[symbol] = features_df
        return features_df

    def get_latest_features(self, symbol: str, period: str = "60d",
                            interval: str = "1h") -> Optional[np.ndarray]:
        """
        Get the latest feature vector for a symbol (single row, 60 features).
        This is what gets passed to the ML Signal Engine.
        """
        from agents.ml_signal_engine.catboost_predictor import FEATURES_60

        if symbol not in self._cache:
            self.fetch_and_engineer(symbol, period, interval)

        if symbol not in self._cache or self._cache[symbol].empty:
            logger.warning("No cached data for %s", symbol)
            return None

        df = self._cache[symbol]

        # Select the 60 features in correct order
        available = [f for f in FEATURES_60 if f in df.columns]
        missing = [f for f in FEATURES_60 if f not in df.columns]

        if missing:
            logger.warning("Missing features for %s: %s", symbol, missing)

        # Get last row
        latest = df[available].iloc[-1].values.astype(float)

        # Fill missing features with 0
        if len(latest) < 60:
            latest = np.concatenate([latest, np.zeros(60 - len(latest))])

        return latest

    @staticmethod
    def _calc_rsi(close: np.ndarray, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    print("\n" + "=" * 60)
    print("MARKET DATA FETCHER — Feature Engineering Test")
    print("=" * 60)

    fetcher = MarketDataFetcher()

    test_symbols = ["AAPL", "BTCUSDT", "EURUSD"]

    for symbol in test_symbols:
        print(f"\n--- {symbol} ---")
        features_df = fetcher.fetch_and_engineer(symbol)

        if features_df is not None:
            print(f"  Rows: {len(features_df)}")
            print(f"  Columns: {len(features_df.columns)}")

            latest = fetcher.get_latest_features(symbol)
            if latest is not None:
                print(f"  Latest feature vector: {latest.shape} — "
                      f"[{latest[0]:.2f}, {latest[1]:.2f}, ... {latest[-1]:.2f}]")
            else:
                print("  ⚠️ Could not get latest features")
        else:
            print("  ⚠️ No data returned")

    print("\n✅ Test complete")
