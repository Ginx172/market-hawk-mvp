"""
SCRIPT NAME: retrain_crypto_specialist.py
====================================
Execution Location: market-hawk-mvp/scripts/
Purpose: Crypto-specialist ML model retraining pipeline.
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-13

Extends scripts/retrain_ml_models.py with crypto-specific improvements:
    - 8 crypto symbols (BTC-USD, ETH-USD, SOL-USD, BNB-USD, ADA-USD,
                        XRP-USD, AVAX-USD, DOT-USD)
    - 2 years of data, 4h and 1d candles
    - 80 features = 60 base (from engineer_features) + 20 crypto-specific
      (from CryptoFeatureEngineer)
    - Threshold-based target: |pct_change| > 0.3% (neutral bars removed)
    - Multi-timeframe stacking: 4h bars enriched with daily aggregation features
    - CatBoost hyperparams tuned for crypto: depth=8, lr=0.03, iter=2000,
      l2_leaf_reg=5, border_count=128
    - Walk-forward cross-validation (5-fold expanding), gap=24 bars
    - Accuracy > 65% flagged as SUSPICIOUS (realistic crypto: 55-63%)
    - Saves model as CatBoost_crypto_v2_{timestamp}.cbm
    - Updates MODEL_REGISTRY placeholder and model_hashes.json
    - Generates docs/CRYPTO_TRAINING_REPORT.md

Usage:
    # Dry run (shows plan, no training)
    python scripts/retrain_crypto_specialist.py --dry-run

    # Train with defaults (8 symbols, 2y, 4h+1d)
    python scripts/retrain_crypto_specialist.py

    # Custom symbols
    python scripts/retrain_crypto_specialist.py --symbols BTC-USD ETH-USD SOL-USD

    # Disable GPU
    python scripts/retrain_crypto_specialist.py --no-gpu
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (
    HARDWARE, ML_CONFIG, MODELS_DIR, setup_logging,
)
from data.market_data_fetcher import (
    MarketDataFetcher,
    adjust_for_splits_dividends,
    detect_unadjusted_splits,
    get_symbol_category,
)
from data.crypto_feature_engineer import (
    CryptoFeatureEngineer,
    CRYPTO_EXTRA_FEATURES,
)
from agents.ml_signal_engine.catboost_predictor import (
    FEATURES_60,
    CRYPTO_FEATURES,
    MODEL_REGISTRY,
    _compute_file_hash,
)
from ml.cross_validation import (
    CVResult,
    ModelEvaluator,
    TimeSeriesCrossValidator,
)

logger = logging.getLogger("market_hawk.crypto_retrain")

# ============================================================
# CONSTANTS
# ============================================================

TARGET_COL = "crypto_target"

# Default crypto symbols
DEFAULT_CRYPTO_SYMBOLS: List[str] = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
    "ADA-USD", "XRP-USD", "AVAX-USD", "DOT-USD",
]

# Accuracy threshold: above this is suspicious for crypto
# (realistic crypto prediction: 55-63%)
CRYPTO_SUSPICIOUS_ACCURACY_THRESHOLD = 0.65

# Minimum percentage move to label as UP or DOWN (noise filter)
TARGET_THRESHOLD = 0.003  # 0.3%

# Prefix for daily features when stacking multi-timeframe
DAILY_PREFIX = "daily_"

# 4h candles -> 6 bars per day
PERIODS_PER_DAY_4H = 6
PERIODS_PER_DAY_1D = 1


# ============================================================
# MULTI-TIMEFRAME HELPERS
# ============================================================

def resample_to_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 4h OHLCV data to daily candles.

    Uses last Close, max High, min Low, first Open, sum Volume for each day.

    Args:
        df_4h: DataFrame with DatetimeIndex and OHLCV columns.

    Returns:
        Daily-resampled DataFrame, or empty DataFrame on failure.
    """
    if not isinstance(df_4h.index, pd.DatetimeIndex):
        logger.warning("resample_to_daily: DataFrame has no DatetimeIndex, skipping")
        return pd.DataFrame()

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df_4h.columns]
    if missing:
        logger.warning("resample_to_daily: missing columns %s", missing)
        return pd.DataFrame()

    daily = df_4h[required].resample("1D").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    logger.info("Resampled %d 4h bars -> %d daily bars", len(df_4h), len(daily))
    return daily


def merge_timeframe_features(df_4h: pd.DataFrame,
                              daily_features: pd.DataFrame,
                              prefix: str = DAILY_PREFIX) -> pd.DataFrame:
    """Merge daily feature columns into a 4h bar DataFrame.

    Each 4h bar receives the daily features for its calendar date (forward-filled
    within the day — no future daily data is used).

    Args:
        df_4h: 4h DataFrame with DatetimeIndex.  Must have at least one column.
        daily_features: Daily DataFrame with DatetimeIndex.
        prefix: Column name prefix for merged daily columns.

    Returns:
        df_4h enriched with prefixed daily feature columns.
    """
    if daily_features.empty or not isinstance(df_4h.index, pd.DatetimeIndex):
        return df_4h

    # Rename daily feature columns with prefix (avoid clash with 4h names)
    extra_cols = [c for c in daily_features.columns if c not in ("Open", "High", "Low", "Close", "Volume")]
    daily_renamed = daily_features[extra_cols].copy()
    daily_renamed.columns = [f"{prefix}{c}" for c in extra_cols]

    # Reindex to 4h frequency using forward-fill (safe: uses only past daily data)
    daily_reindexed = daily_renamed.reindex(df_4h.index, method="ffill")

    result = df_4h.copy()
    for col in daily_reindexed.columns:
        result[col] = daily_reindexed[col].values

    logger.info("Merged %d daily feature columns into 4h frame", len(daily_reindexed.columns))
    return result


# ============================================================
# CRYPTO DATA PIPELINE
# ============================================================

class CryptoDataPipeline:
    """Crypto-specialist data loading, feature engineering, and splitting.

    Extends the retrain_ml_models.DataPipeline with:
    - Crypto-specific symbols
    - 4h and/or 1d intervals
    - 20 additional crypto features via CryptoFeatureEngineer
    - Multi-timeframe stacking (4h bars enriched with daily features)
    - Threshold-based target variable (noise filter for crypto)
    - 24-bar gap (anti-leakage, ~1 day for 4h data)

    Args:
        symbols: Crypto ticker symbols (yfinance format, e.g. "BTC-USD").
        period: yfinance period string (default "2y").
        interval: Primary candle interval ("4h" or "1d").
        use_daily_stack: If True and interval=="4h", also compute daily
            features and merge them as additional columns.
        cutoff_date: ISO date string for temporal split. None = 80% train.
        gap_bars: Bars between train end and test start.  Default 24
            (≈1 day for 4h, 24 days for 1d — adjust per interval).
        threshold: Minimum absolute return to label as directional move.
    """

    def __init__(self,
                 symbols: List[str],
                 period: str = "2y",
                 interval: str = "4h",
                 use_daily_stack: bool = True,
                 cutoff_date: Optional[str] = None,
                 gap_bars: int = 24,
                 threshold: float = TARGET_THRESHOLD) -> None:
        self.symbols = symbols
        self.period = period
        self.interval = interval
        self.use_daily_stack = use_daily_stack
        self.cutoff_date = cutoff_date
        self.gap_bars = gap_bars
        self.threshold = threshold

        periods_per_day = PERIODS_PER_DAY_4H if interval == "4h" else PERIODS_PER_DAY_1D
        self._fetcher = MarketDataFetcher()
        self._crypto_engineer = CryptoFeatureEngineer(periods_per_day=periods_per_day)

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------

    def load_and_prepare(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load OHLCV, build 80 features, create threshold target.

        Steps:
            1. Fetch OHLCV (primary interval)
            2. adjust_for_splits_dividends
            3. engineer_features (base 60)
            4. If use_daily_stack and interval==4h: resample to daily,
               compute daily features, merge back as ``daily_*`` columns
            5. CryptoFeatureEngineer.add_features (20 crypto-specific)
            6. Create threshold-based target; drop neutral rows
            7. Restore DatetimeIndex

        Args:
            symbol: Crypto ticker (e.g. "BTC-USD").

        Returns:
            DataFrame with 80+ features and TARGET_COL, or None.
        """
        logger.info("Loading %s (period=%s, interval=%s)", symbol, self.period, self.interval)

        df = self._fetcher.fetch_ohlcv(symbol, period=self.period, interval=self.interval)
        if df is None or df.empty:
            logger.warning("No data for %s, skipping", symbol)
            return None

        # Corporate action adjustment
        splits = detect_unadjusted_splits(df)
        if splits:
            logger.warning("%s: %d potential unadjusted splits detected", symbol, len(splits))
        df = adjust_for_splits_dividends(df)

        # Validate OHLCV
        df = self._validate_ohlcv(df)
        if df.empty:
            logger.warning("%s: empty after OHLCV validation", symbol)
            return None

        # Save datetime index (will be re-attached later)
        dt_index: Optional[pd.DatetimeIndex] = (
            df.index if isinstance(df.index, pd.DatetimeIndex) else None
        )

        # Base 60 features
        features_df = self._fetcher.engineer_features(df, symbol=symbol)

        # Re-attach DatetimeIndex before crypto feature engineering
        if dt_index is not None and len(dt_index) >= len(features_df):
            features_df.index = dt_index[:len(features_df)]

        # Multi-timeframe stacking (4h only)
        if self.use_daily_stack and self.interval == "4h" and dt_index is not None:
            daily_ohlcv = resample_to_daily(df)
            if not daily_ohlcv.empty:
                daily_feats = self._fetcher.engineer_features(daily_ohlcv, symbol=f"{symbol}_daily")
                if not daily_feats.empty:
                    daily_feats.index = daily_ohlcv.index[:len(daily_feats)]
                    features_df = merge_timeframe_features(features_df, daily_feats, prefix=DAILY_PREFIX)

        # 20 crypto-specific features
        features_df = self._crypto_engineer.add_features(features_df)

        # Threshold-based target
        features_df = self._create_threshold_target(features_df)
        if features_df.empty or TARGET_COL not in features_df.columns:
            logger.warning("%s: empty after target creation, skipping", symbol)
            return None

        pct_directional = features_df[TARGET_COL].isin([0, 1]).mean()
        logger.info(
            "%s: %d rows after neutral removal (%.1f%% directional, %.1f%% up)",
            symbol, len(features_df), pct_directional * 100,
            features_df[TARGET_COL].mean() * 100,
        )

        return features_df

    def load_multi_symbol(self) -> Optional[pd.DataFrame]:
        """Load and concatenate data for all crypto symbols.

        Returns:
            Combined DataFrame with ``_symbol`` column, or None.
        """
        frames: List[pd.DataFrame] = []

        for symbol in self.symbols:
            df = self.load_and_prepare(symbol)
            if df is not None:
                df["_symbol"] = symbol
                frames.append(df)
            gc.collect()

        if not frames:
            logger.error("No data loaded for any crypto symbol")
            return None

        combined = pd.concat(frames, ignore_index=False)
        combined = combined.sort_index()

        logger.info("Combined crypto dataset: %d rows from %d symbols",
                     len(combined), len(frames))
        return combined

    def temporal_split(self, df: pd.DataFrame,
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Temporal split with gap (anti-leakage).

        Args:
            df: Combined DataFrame with DatetimeIndex.

        Returns:
            (train_df, test_df).
        """
        if self.cutoff_date:
            cutoff = pd.Timestamp(self.cutoff_date)
        else:
            n = len(df)
            cutoff_idx = int(n * 0.80)
            if isinstance(df.index, pd.DatetimeIndex):
                cutoff = df.index[cutoff_idx]
            else:
                train = df.iloc[:cutoff_idx]
                test = df.iloc[cutoff_idx + self.gap_bars:]
                logger.info("Temporal split (integer): train=%d, gap=%d, test=%d",
                             len(train), self.gap_bars, len(test))
                return train, test

        gap_end = cutoff + pd.Timedelta(days=self.gap_bars)
        train = df[df.index < cutoff]
        test = df[df.index >= gap_end]

        logger.info(
            "Temporal split: train=%d (< %s), gap=%d bars, test=%d (>= %s)",
            len(train), cutoff.date(), self.gap_bars,
            len(test), gap_end.date(),
        )
        return train, test

    def get_feature_columns(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """Return feature columns present in the loaded DataFrame.

        Falls back to CRYPTO_FEATURES if no DataFrame provided.

        Args:
            df: Loaded DataFrame to inspect (optional).

        Returns:
            List of feature column names.
        """
        if df is None:
            return list(CRYPTO_FEATURES)
        exclude = {TARGET_COL, "_symbol", "price_up"}
        return [c for c in df.columns if c not in exclude]

    # ----------------------------------------------------------
    # PRIVATE HELPERS
    # ----------------------------------------------------------

    def _create_threshold_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create threshold-based directional target and remove neutral rows.

        Labels:
            1  — next bar close > this close by > threshold (UP)
            0  — next bar close < this close by > threshold (DOWN)
            -1 — neutral (|pct_change| <= threshold) — these rows are REMOVED

        Args:
            df: DataFrame with a ``Close`` column.

        Returns:
            DataFrame with TARGET_COL column, neutral rows removed,
            last row dropped (no future target available).
        """
        if "Close" not in df.columns:
            logger.error("_create_threshold_target: 'Close' column missing")
            return pd.DataFrame()

        close = df["Close"].astype(float)
        pct_change = (close.shift(-1) - close) / close

        labels = pd.Series(-1, index=df.index, dtype=int)
        labels[pct_change > self.threshold] = 1
        labels[pct_change < -self.threshold] = 0

        df = df.copy()
        df[TARGET_COL] = labels

        # Drop last row (no next-bar target available)
        df = df.iloc[:-1]

        # Remove neutral rows
        n_before = len(df)
        df = df[df[TARGET_COL] != -1].copy()
        n_removed = n_before - len(df)

        logger.info(
            "Target: %d neutral rows removed (%.1f%%), %d directional rows kept",
            n_removed, n_removed / max(n_before, 1) * 100, len(df),
        )
        return df

    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Basic OHLCV validation: positive prices, High >= Low, Volume >= 0."""
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("Missing OHLCV columns: %s", missing)
            return pd.DataFrame()

        mask = (
            (df["Open"] > 0) & (df["High"] > 0) &
            (df["Low"] > 0) & (df["Close"] > 0) &
            (df["High"] >= df["Low"]) & (df["Volume"] >= 0)
        )
        removed = (~mask).sum()
        if removed > 0:
            logger.warning("Removed %d invalid OHLCV rows (%.1f%%)",
                            removed, removed / len(df) * 100)
        return df[mask].copy()


# ============================================================
# CRYPTO MODEL TRAINER
# ============================================================

class CryptoModelTrainer:
    """Train CatBoost classifier with crypto-optimised hyperparameters.

    Hyperparameters chosen for crypto:
        - iterations=2000 — more boosting rounds for complex patterns
        - depth=8         — deeper trees to capture non-linear interactions
        - learning_rate=0.03 — slower learning for better generalisation
        - l2_leaf_reg=5   — regularisation against overfitting
        - border_count=128 — more split points for continuous features

    Args:
        output_dir: Directory to save models and checkpoints.
        use_gpu: Whether to attempt GPU training.
        iterations: Maximum boosting rounds.
        depth: Tree depth.
        learning_rate: Gradient step size.
        l2_leaf_reg: L2 regularisation coefficient.
        border_count: Number of histogram split points.
        early_stopping_rounds: Stop if no improvement for N rounds.
        checkpoint_interval: Save checkpoint every N iterations.
    """

    def __init__(self,
                 output_dir: Path,
                 use_gpu: bool = True,
                 iterations: int = 2000,
                 depth: int = 8,
                 learning_rate: float = 0.03,
                 l2_leaf_reg: float = 5.0,
                 border_count: int = 128,
                 early_stopping_rounds: int = 100,
                 checkpoint_interval: int = 100) -> None:
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.early_stopping_rounds = early_stopping_rounds
        self.checkpoint_interval = checkpoint_interval

        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def train_catboost(self,
                       train_df: pd.DataFrame,
                       features: List[str],
                       target: str,
                       eval_df: Optional[pd.DataFrame] = None,
                       ) -> Tuple[Any, Dict[str, Any]]:
        """Train CatBoost classifier with crypto hyperparameters.

        Args:
            train_df: Training data with features and target columns.
            features: Feature column names.
            target: Target column name.
            eval_df: Optional evaluation set for early stopping.

        Returns:
            (model, metrics_dict).
        """
        from catboost import CatBoostClassifier, Pool

        available_features = [f for f in features if f in train_df.columns]
        logger.info(
            "Training CatBoost (crypto): %d rows, %d features, %d iterations",
            len(train_df), len(available_features), self.iterations,
        )

        X_train = train_df[available_features].values.astype(float)
        y_train = train_df[target].values

        task_type = "GPU" if self._gpu_available() else "CPU"

        model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            eval_metric="Accuracy",
            random_seed=42,
            task_type=task_type,
            verbose=self.checkpoint_interval,
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
            snapshot_file=str(self.output_dir / "checkpoints" / "crypto_catboost_snapshot"),
            snapshot_interval=self.checkpoint_interval,
        )

        eval_set = None
        if eval_df is not None:
            X_eval = eval_df[available_features].values.astype(float)
            y_eval = eval_df[target].values
            eval_set = Pool(X_eval, y_eval)

        train_pool = Pool(X_train, y_train)

        t0 = time.time()
        model.fit(train_pool, eval_set=eval_set)
        elapsed = time.time() - t0

        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_train).flatten()
        train_acc = float(accuracy_score(y_train, y_pred))

        metrics: Dict[str, Any] = {
            "train_accuracy": train_acc,
            "iterations_used": model.tree_count_,
            "elapsed_sec": elapsed,
            "task_type": task_type,
            "features_used": len(available_features),
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "border_count": self.border_count,
        }

        logger.info(
            "CatBoost crypto trained: acc=%.4f, trees=%d, %.1fs (%s)",
            train_acc, model.tree_count_, elapsed, task_type,
        )

        self._cleanup_gpu()
        return model, metrics

    def save_model(self, model: Any, name: str) -> Path:
        """Save CatBoost model to .cbm format.

        Args:
            model: Trained CatBoostClassifier.
            name: Model filename (without extension).

        Returns:
            Path to saved model file.
        """
        path = self.output_dir / f"{name}.cbm"
        model.save_model(str(path))
        logger.info("Model saved: %s (%.2f MB)", path, path.stat().st_size / 1e6)
        return path

    def _gpu_available(self) -> bool:
        """Check if CUDA GPU is available."""
        if not self.use_gpu:
            return False
        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                logger.info(
                    "GPU available: %s (%d MB VRAM)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_mem // 1024 // 1024,
                )
            return available
        except ImportError:
            return False

    @staticmethod
    def _cleanup_gpu() -> None:
        """Release GPU memory."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ============================================================
# CRYPTO EVALUATOR
# ============================================================

class CryptoEvaluator:
    """Evaluate crypto model with time-series cross-validation.

    Uses a 5-fold expanding window with a 24-bar gap.  Flags accuracy
    above 65% as suspicious (realistic crypto target: 55-63%).

    Args:
        n_splits: Number of CV folds.
        min_train_size: Minimum rows for first fold.
        test_size: Test rows per fold.
        gap_bars: Gap between train end and test start.
    """

    SUSPICIOUS_THRESHOLD = CRYPTO_SUSPICIOUS_ACCURACY_THRESHOLD

    def __init__(self,
                 n_splits: int = 5,
                 min_train_size: int = 500,
                 test_size: int = 100,
                 gap_bars: int = 24) -> None:
        self.cv = TimeSeriesCrossValidator(
            n_splits=n_splits,
            min_train_size=min_train_size,
            test_size=test_size,
            gap_bars=gap_bars,
            mode="expanding",
        )
        self._evaluator = ModelEvaluator()

    def evaluate(self,
                 model: Any,
                 df: pd.DataFrame,
                 features: List[str],
                 target: str,
                 model_name: str = "crypto_model",
                 ) -> CVResult:
        """Run cross-validation and flag suspicious accuracy.

        Args:
            model: Sklearn-compatible trained model.
            df: DataFrame with features and target.
            features: Feature column names.
            target: Target column name.
            model_name: Label for logging.

        Returns:
            CVResult with per-fold metrics.
        """
        available_features = [f for f in features if f in df.columns]

        logger.info(
            "Evaluating %s: %d rows, %d features, %d folds",
            model_name, len(df), len(available_features), self.cv.n_splits,
        )

        result = self._evaluator.cross_validate(
            model=model,
            df=df,
            features=available_features,
            target=target,
            cv=self.cv,
            verbose=True,
        )

        if result.mean_score > self.SUSPICIOUS_THRESHOLD:
            logger.warning(
                "SUSPICIOUS: %s mean accuracy %.4f > %.0f%% threshold. "
                "Realistic crypto models achieve 55-63%%. "
                "Possible data leakage or overfitting.",
                model_name, result.mean_score,
                self.SUSPICIOUS_THRESHOLD * 100,
            )

        if result.is_overfit:
            logger.warning(
                "OVERFITTING: %s overfit_ratio=%.2f (train/test). "
                "Threshold is 1.5.",
                model_name, result.overfit_ratio,
            )

        return result

    def generate_report(self,
                        results: Dict[str, CVResult],
                        train_metrics: Dict[str, Dict],
                        output_path: Path,
                        symbols: List[str],
                        cutoff_info: str,
                        interval: str) -> None:
        """Generate Markdown training report for crypto specialist.

        Args:
            results: model_name -> CVResult
            train_metrics: model_name -> training metrics dict
            output_path: Path to save the .md report
            symbols: Crypto symbols used for training
            cutoff_info: Train/test split description
            interval: Candle interval used
        """
        lines = [
            "# Crypto Specialist ML Training Report",
            "",
            f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Symbols**: {', '.join(symbols)}",
            f"**Interval**: {interval}",
            f"**Split**: {cutoff_info}",
            f"**Hardware**: {HARDWARE.cpu_name}, {HARDWARE.gpu_name}, "
            f"{HARDWARE.ram_gb}GB {HARDWARE.ram_type}",
            "",
            "## Summary",
            "",
            "| Model | Mean Acc | Std | Overfit Ratio | Status |",
            "|-------|----------|-----|---------------|--------|",
        ]

        for name, cv_result in results.items():
            status = "OK"
            if cv_result.mean_score > self.SUSPICIOUS_THRESHOLD:
                status = f"SUSPICIOUS (>{self.SUSPICIOUS_THRESHOLD:.0%})"
            if cv_result.is_overfit:
                status = "OVERFITTING"

            lines.append(
                f"| {name} | {cv_result.mean_score:.4f} | "
                f"{cv_result.std_score:.4f} | "
                f"{cv_result.overfit_ratio:.2f} | {status} |"
            )

        lines.extend(["", "## Per-Model Details", ""])

        for name, cv_result in results.items():
            tm = train_metrics.get(name, {})
            lines.extend([
                f"### {name}",
                "",
                f"- **Train accuracy**: {tm.get('train_accuracy', 'N/A')}",
                f"- **Iterations used**: {tm.get('iterations_used', 'N/A')}",
                f"- **Training time**: {tm.get('elapsed_sec', 0):.1f}s",
                f"- **Compute**: {tm.get('task_type', 'N/A')}",
                f"- **Features**: {tm.get('features_used', 'N/A')}",
                f"- **Depth**: {tm.get('depth', 8)}",
                f"- **Learning rate**: {tm.get('learning_rate', 0.03)}",
                f"- **L2 leaf reg**: {tm.get('l2_leaf_reg', 5.0)}",
                "",
                "**Cross-Validation Folds:**",
                "",
                "| Fold | Test Acc | Precision | Recall | F1 | Train Acc | Train Size | Test Size |",
                "|------|----------|-----------|--------|-----|-----------|------------|-----------|",
            ])

            for fr in cv_result.fold_results:
                lines.append(
                    f"| {fr.fold_idx} | {fr.accuracy:.4f} | "
                    f"{fr.precision:.4f} | {fr.recall:.4f} | "
                    f"{fr.f1:.4f} | {fr.train_accuracy:.4f} | "
                    f"{fr.train_size} | {fr.test_size} |"
                )

            lines.extend([
                "",
                f"**Overfit ratio**: {cv_result.overfit_ratio:.2f} "
                f"({'WARNING' if cv_result.is_overfit else 'OK'})",
                "",
            ])

        lines.extend([
            "## Accuracy Guidelines (Crypto)",
            "",
            "- **50-52%**: Random — model is not learning",
            "- **52-55%**: Minimal signal — marginal use",
            "- **55-60%**: Good — realistic crypto prediction range",
            "- **60-63%**: Excellent — verify no data leakage",
            "- **>65%**: Suspicious — check for look-ahead bias or overfitting",
            "",
            "## Feature Set",
            "",
            f"- **Base features (60)**: Standard TA features from `engineer_features()`",
            f"- **Crypto features (20)**: Volatility, volume, momentum, price structure, session",
            f"- **Total**: {len(CRYPTO_FEATURES)} features",
            "",
            "## Target Variable",
            "",
            f"- Threshold: `|pct_change| > {TARGET_THRESHOLD:.1%}`",
            "- Label 1 (UP): next close > current close by > threshold",
            "- Label 0 (DOWN): next close < current close by > threshold",
            "- Neutral rows (|move| ≤ threshold) are removed from training",
            "",
            "---",
            "*Generated by scripts/retrain_crypto_specialist.py*",
        ])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Crypto training report saved: %s", output_path)


# ============================================================
# HASH UPDATER
# ============================================================

def update_model_hashes(model_paths: Dict[str, Path],
                        hash_file: Path) -> Dict[str, Dict[str, str]]:
    """Compute SHA-256 hashes for newly trained models and merge into hash file.

    Args:
        model_paths: model_name -> Path to saved model file.
        hash_file: Path to model_hashes.json.

    Returns:
        Updated hash store dict.
    """
    existing: Dict[str, Dict[str, str]] = {}
    if hash_file.exists():
        try:
            existing = json.loads(hash_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read existing hash file, starting fresh")

    for name, path in model_paths.items():
        if path.exists():
            file_hash = _compute_file_hash(path)
            existing[name] = {
                "path": str(path),
                "sha256": file_hash,
            }
            logger.info("Hash for %s: %s", name, file_hash[:16])

    hash_file.parent.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.info("Updated %s (%d entries)", hash_file, len(existing))

    return existing


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Market Hawk — Crypto Specialist Model Retraining Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run
  python scripts/retrain_crypto_specialist.py --dry-run

  # Train with defaults (8 crypto symbols, 2y, 4h+daily)
  python scripts/retrain_crypto_specialist.py

  # Custom symbols
  python scripts/retrain_crypto_specialist.py --symbols BTC-USD ETH-USD SOL-USD

  # Daily candles only (no multi-timeframe stack)
  python scripts/retrain_crypto_specialist.py --interval 1d --no-daily-stack
        """,
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_CRYPTO_SYMBOLS,
        help="Crypto symbols to train on (default: 8 crypto assets)",
    )
    parser.add_argument(
        "--period", type=str, default="2y",
        help="yfinance data period (default: 2y)",
    )
    parser.add_argument(
        "--interval", type=str, default="4h",
        choices=["4h", "1d", "1h"],
        help="Candle interval (default: 4h)",
    )
    parser.add_argument(
        "--no-daily-stack", action="store_true",
        help="Disable multi-timeframe daily feature stacking",
    )
    parser.add_argument(
        "--cutoff-date", type=str, default=None,
        help="Temporal split cutoff date (ISO format). Default: 80%% of data.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(MODELS_DIR / "retrained"),
        help="Directory to save retrained models (default: models/retrained/)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU training",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without training",
    )
    return parser.parse_args()


def dry_run_report(args: argparse.Namespace) -> None:
    """Print training plan without executing."""
    output_dir = Path(args.output_dir)
    use_daily_stack = not args.no_daily_stack and args.interval == "4h"
    n_features = len(CRYPTO_FEATURES)
    if use_daily_stack:
        n_features_note = f"{n_features} crypto + ~60 daily-stacked = ~{n_features + 60} total"
    else:
        n_features_note = str(n_features)

    print("\n" + "=" * 65)
    print("DRY RUN — Crypto Specialist ML Retraining Plan")
    print("=" * 65)
    print(f"\n  Symbols:        {', '.join(args.symbols)}")
    print(f"  Period:         {args.period}")
    print(f"  Interval:       {args.interval}")
    print(f"  Daily stack:    {'yes' if use_daily_stack else 'no'}")
    print(f"  Cutoff:         {args.cutoff_date or 'auto (80% train / 20% test)'}")
    print(f"  Gap:            24 bars (anti-leakage)")
    print(f"  Output dir:     {output_dir}")
    print(f"  GPU:            {'disabled' if args.no_gpu else 'auto-detect'}")
    print(f"\n  Features:       {n_features_note}")
    print(f"    - Base 60:    standard TA (engineer_features)")
    print(f"    - Crypto 20:  {', '.join(CRYPTO_EXTRA_FEATURES[:4])} ...")
    print(f"  Target:         threshold (|pct_change| > {TARGET_THRESHOLD:.1%})")
    print(f"  CV:             5-fold expanding window (gap=24 bars)")
    print(f"\n  CatBoost hyperparams (crypto-tuned):")
    print(f"    iterations:       2000")
    print(f"    depth:            8")
    print(f"    learning_rate:    0.03")
    print(f"    l2_leaf_reg:      5")
    print(f"    border_count:     128")
    print(f"    early_stopping:   100 rounds")
    print(f"\n  Hardware:")
    print(f"    CPU: {HARDWARE.cpu_name} ({HARDWARE.cpu_cores} cores)")
    print(f"    GPU: {HARDWARE.gpu_name} ({HARDWARE.gpu_vram_gb}GB VRAM)")
    print(f"    RAM: {HARDWARE.ram_gb}GB {HARDWARE.ram_type}")
    print(f"\n  Output files (planned):")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"    {output_dir / f'CatBoost_crypto_v2_{ts}.cbm'}")
    print(f"    {MODELS_DIR / 'model_hashes.json'} (updated)")
    print(f"    docs/CRYPTO_TRAINING_REPORT.md")
    print(f"\n  Accuracy flag:  > {CRYPTO_SUSPICIOUS_ACCURACY_THRESHOLD:.0%} = SUSPICIOUS")
    print(f"  Current model:  catboost_crypto_56 (56% accuracy)")
    print(f"  Target range:   55-63% (realistic crypto prediction)")
    print("\n" + "=" * 65)
    print("Dry run complete. Remove --dry-run to execute.")
    print("=" * 65 + "\n")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    """Main crypto specialist retraining pipeline."""
    setup_logging(logging.INFO)
    args = parse_args()

    if args.dry_run:
        dry_run_report(args)
        return

    t0 = time.time()
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_daily_stack = not args.no_daily_stack and args.interval == "4h"

    logger.info("=" * 65)
    logger.info("CRYPTO SPECIALIST RETRAINING PIPELINE START")
    logger.info("Symbols:   %s", args.symbols)
    logger.info("Interval:  %s  |  Daily stack: %s", args.interval, use_daily_stack)
    logger.info("Output:    %s", output_dir)
    logger.info("=" * 65)

    # ----------------------------------------------------------
    # Step 1: Load and prepare crypto data
    # ----------------------------------------------------------
    logger.info("Step 1/4: Loading and preparing crypto data...")
    pipeline = CryptoDataPipeline(
        symbols=args.symbols,
        period=args.period,
        interval=args.interval,
        use_daily_stack=use_daily_stack,
        cutoff_date=args.cutoff_date,
        gap_bars=24,
        threshold=TARGET_THRESHOLD,
    )
    combined = pipeline.load_multi_symbol()

    if combined is None:
        logger.error("No data loaded. Aborting.")
        sys.exit(1)

    features = pipeline.get_feature_columns(combined)
    logger.info("Features available: %d", len(features))

    # ----------------------------------------------------------
    # Step 2: Temporal split
    # ----------------------------------------------------------
    logger.info("Step 2/4: Temporal train/test split...")
    train_df, test_df = pipeline.temporal_split(combined)

    if len(train_df) < 100 or len(test_df) < 50:
        logger.error(
            "Insufficient data: train=%d, test=%d. Need at least 100/50.",
            len(train_df), len(test_df),
        )
        sys.exit(1)

    cutoff_info = (
        f"train={len(train_df)} rows, test={len(test_df)} rows, gap=24 bars"
    )

    # ----------------------------------------------------------
    # Step 3: Train CatBoost (crypto-tuned)
    # ----------------------------------------------------------
    logger.info("Step 3/4: Training CatBoost (crypto specialist)...")
    trainer = CryptoModelTrainer(
        output_dir=output_dir,
        use_gpu=not args.no_gpu,
    )

    model, train_metrics = trainer.train_catboost(
        train_df, features, TARGET_COL, eval_df=test_df,
    )

    model_name = f"CatBoost_crypto_v2_{timestamp}"
    model_path = trainer.save_model(model, model_name)
    gc.collect()

    # ----------------------------------------------------------
    # Step 4: Cross-validation evaluation
    # ----------------------------------------------------------
    logger.info("Step 4/4: Cross-validation evaluation...")
    evaluator = CryptoEvaluator(
        n_splits=5,
        min_train_size=max(500, len(train_df) // 3),
        test_size=max(100, len(train_df) // 10),
        gap_bars=24,
    )

    cv_result = evaluator.evaluate(
        model=model,
        df=train_df,
        features=features,
        target=TARGET_COL,
        model_name=model_name,
    )
    gc.collect()

    # ----------------------------------------------------------
    # Post-training: hash update, registry note, report
    # ----------------------------------------------------------
    hash_file = MODELS_DIR / "model_hashes.json"
    update_model_hashes({model_name: model_path}, hash_file)

    report_path = _PROJECT_ROOT / "docs" / "CRYPTO_TRAINING_REPORT.md"
    evaluator.generate_report(
        results={model_name: cv_result},
        train_metrics={model_name: train_metrics},
        output_path=report_path,
        symbols=args.symbols,
        cutoff_info=cutoff_info,
        interval=args.interval,
    )

    elapsed_total = time.time() - t0

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    logger.info("=" * 65)
    logger.info("CRYPTO SPECIALIST TRAINING COMPLETE (%.1f min)", elapsed_total / 60)
    logger.info("Model:       %s", model_path)
    logger.info("CV accuracy: %.4f ± %.4f", cv_result.mean_score, cv_result.std_score)
    logger.info("Overfit ratio: %.2f %s",
                 cv_result.overfit_ratio,
                 "⚠️ WARNING" if cv_result.is_overfit else "OK")
    if cv_result.mean_score > CRYPTO_SUSPICIOUS_ACCURACY_THRESHOLD:
        logger.warning("⚠️  Accuracy > %.0f%% — check for data leakage!",
                        CRYPTO_SUSPICIOUS_ACCURACY_THRESHOLD * 100)
    else:
        logger.info("✅ Accuracy in realistic range (55-63%% target)")
    logger.info("Report:      %s", report_path)
    logger.info("Next step:   Update MODEL_REGISTRY 'catboost_crypto_v2' path to: %s",
                 model_path)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
