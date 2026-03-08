"""
SCRIPT NAME: retrain_ml_models.py
====================================
Execution Location: market-hawk-mvp/scripts/
Purpose: ML model retraining pipeline with time-series cross-validation.
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-08

Retrains CatBoost and XGBoost models using:
    - OHLCV data from local dirs or yfinance
    - 60 engineered features (TA-Lib based)
    - Temporal train/test split with 5-day anti-leakage gap
    - TimeSeriesCrossValidator (5-fold expanding window)
    - Overfit detection (flag accuracy > 70% as SUSPICIOUS)
    - Checkpoint every 100 iterations
    - Memory management: gc.collect(), torch.cuda.empty_cache()

Usage:
    # Dry run (shows plan, no training)
    python scripts/retrain_ml_models.py --symbols AAPL MSFT --dry-run

    # Train with default cutoff (80% of data)
    python scripts/retrain_ml_models.py --symbols AAPL MSFT GOOGL

    # Train with specific cutoff date
    python scripts/retrain_ml_models.py --symbols AAPL --cutoff-date 2025-06-01

    # Custom output directory
    python scripts/retrain_ml_models.py --symbols AAPL --output-dir models/experiment_1
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Garanteaza ca project root e in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (
    HARDWARE, ML_CONFIG, MODELS_DIR, setup_logging,
)
from data.market_data_fetcher import (
    MarketDataFetcher,
    adjust_for_splits_dividends,
    detect_unadjusted_splits,
)
from agents.ml_signal_engine.catboost_predictor import (
    FEATURES_60,
    MODEL_REGISTRY,
    _compute_file_hash,
)
from ml.cross_validation import (
    CVResult,
    ModelEvaluator,
    TimeSeriesCrossValidator,
)

logger = logging.getLogger("market_hawk.retrain")

# Target column name
TARGET_COL = "price_up"

# Default symbols for training
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "AMD", "AMZN"]

# Accuracy threshold: above this is suspicious (realistic is 52-60%)
SUSPICIOUS_ACCURACY_THRESHOLD = 0.70


# ============================================================
# DATA PIPELINE
# ============================================================

class DataPipeline:
    """Load, engineer, validate, and split OHLCV data for ML training.

    Pipeline:
        1. Load OHLCV from yfinance (or local files via HistoricalDataLoader)
        2. Apply adjust_for_splits_dividends()
        3. Apply engineer_features() (60 features)
        4. Create binary target: price_up = Close[t+1] > Close[t]
        5. Temporal split: train = date < cutoff, test = date >= cutoff
        6. 5-day gap between train and test (anti-leakage)
        7. NO bfill, NO shuffle, NO future data

    Args:
        symbols: List of ticker symbols.
        period: yfinance period string (e.g., "2y", "5y").
        interval: Candle interval (e.g., "1d", "1h").
        cutoff_date: Date string for temporal split. If None, uses 80% of data.
        gap_days: Days between train end and test start (anti-leakage).
    """

    def __init__(self,
                 symbols: List[str],
                 period: str = "2y",
                 interval: str = "1d",
                 cutoff_date: Optional[str] = None,
                 gap_days: int = 5) -> None:
        self.symbols = symbols
        self.period = period
        self.interval = interval
        self.cutoff_date = cutoff_date
        self.gap_days = gap_days
        self._fetcher = MarketDataFetcher()

    def load_and_prepare(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load OHLCV, adjust, engineer features, create target.

        Returns:
            DataFrame with features + 'price_up' target column, or None on failure.
        """
        logger.info("Loading data for %s (period=%s, interval=%s)",
                     symbol, self.period, self.interval)

        df = self._fetcher.fetch_ohlcv(symbol, period=self.period,
                                        interval=self.interval)
        if df is None or df.empty:
            logger.warning("No data for %s, skipping", symbol)
            return None

        # Corporate action adjustment
        splits = detect_unadjusted_splits(df)
        if splits:
            logger.warning("%s: %d potential unadjusted splits detected",
                           symbol, len(splits))
        df = adjust_for_splits_dividends(df)

        # Validate OHLCV basics
        df = self._validate_ohlcv(df)
        if df.empty:
            logger.warning("%s: empty after OHLCV validation", symbol)
            return None

        # Engineer 60 features
        features_df = self._fetcher.engineer_features(df, symbol=symbol)

        # Create binary target: price_up = Close[t+1] > Close[t]
        # Shift(-1) looks at NEXT bar — target is future information
        features_df[TARGET_COL] = (
            features_df["Close"].shift(-1) > features_df["Close"]
        ).astype(int)

        # Drop last row (no target available)
        features_df = features_df.iloc[:-1]

        # Restore datetime index for temporal splitting
        if isinstance(df.index, pd.DatetimeIndex):
            features_df.index = df.index[:len(features_df)]

        logger.info("%s: %d rows, %d features, target distribution: %.1f%% up",
                     symbol, len(features_df), len(FEATURES_60),
                     features_df[TARGET_COL].mean() * 100)

        return features_df

    def load_multi_symbol(self) -> Optional[pd.DataFrame]:
        """Load and concatenate data for all symbols.

        Returns:
            Combined DataFrame with 'symbol' column, or None.
        """
        frames: List[pd.DataFrame] = []

        for symbol in self.symbols:
            df = self.load_and_prepare(symbol)
            if df is not None:
                df["_symbol"] = symbol
                frames.append(df)
            gc.collect()

        if not frames:
            logger.error("No data loaded for any symbol")
            return None

        combined = pd.concat(frames, ignore_index=False)
        combined = combined.sort_index()

        logger.info("Combined dataset: %d rows from %d symbols",
                     len(combined), len(frames))
        return combined

    def temporal_split(self, df: pd.DataFrame
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data temporally with gap.

        Returns:
            (train_df, test_df) with gap_days removed between them.
        """
        if self.cutoff_date:
            cutoff = pd.Timestamp(self.cutoff_date)
        else:
            # Default: 80% train, 20% test
            n = len(df)
            cutoff_idx = int(n * 0.80)
            if isinstance(df.index, pd.DatetimeIndex):
                cutoff = df.index[cutoff_idx]
            else:
                # Fallback: integer split
                train = df.iloc[:cutoff_idx]
                gap_rows = self.gap_days  # Approximate gap in bars
                test = df.iloc[cutoff_idx + gap_rows:]
                logger.info("Temporal split (integer): train=%d, gap=%d, test=%d",
                             len(train), gap_rows, len(test))
                return train, test

        # Gap: exclude gap_days after cutoff
        gap_end = cutoff + pd.Timedelta(days=self.gap_days)

        train = df[df.index < cutoff]
        test = df[df.index >= gap_end]

        logger.info("Temporal split: train=%d (< %s), gap=%d days, test=%d (>= %s)",
                     len(train), cutoff.date(), self.gap_days,
                     len(test), gap_end.date())

        if len(train) == 0:
            logger.error("Train set is empty! Cutoff too early.")
        if len(test) == 0:
            logger.error("Test set is empty! Cutoff too late.")

        return train, test

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

    def get_feature_columns(self) -> List[str]:
        """Return the 60 feature columns used for training."""
        return list(FEATURES_60)


# ============================================================
# MODEL TRAINER
# ============================================================

class ModelTrainer:
    """Train CatBoost and XGBoost classifiers with checkpointing.

    Args:
        output_dir: Directory to save checkpoints and final models.
        use_gpu: Whether to attempt GPU training (GTX 1070).
        max_depth: Tree depth.
        learning_rate: Gradient step size.
        iterations: Max boosting rounds.
        early_stopping_rounds: Stop after N rounds without improvement.
        checkpoint_interval: Save checkpoint every N iterations.
    """

    def __init__(self,
                 output_dir: Path,
                 use_gpu: bool = True,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 iterations: int = 1000,
                 early_stopping_rounds: int = 50,
                 checkpoint_interval: int = 100) -> None:
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.checkpoint_interval = checkpoint_interval

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

    def train_catboost(self,
                       train_df: pd.DataFrame,
                       features: List[str],
                       target: str,
                       eval_df: Optional[pd.DataFrame] = None,
                       ) -> Tuple[Any, Dict[str, float]]:
        """Train a CatBoost classifier.

        Args:
            train_df: Training data with features and target.
            features: Feature column names.
            target: Target column name.
            eval_df: Optional evaluation set for early stopping.

        Returns:
            (model, metrics_dict) where metrics_dict has train_accuracy.
        """
        from catboost import CatBoostClassifier, Pool

        available_features = [f for f in features if f in train_df.columns]
        logger.info("Training CatBoost: %d rows, %d features, %d iterations",
                     len(train_df), len(available_features), self.iterations)

        X_train = train_df[available_features].values
        y_train = train_df[target].values

        # GPU config: task_type='GPU' pe GTX 1070
        task_type = "GPU" if self._gpu_available() else "CPU"

        model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric="Accuracy",
            random_seed=42,
            task_type=task_type,
            verbose=self.checkpoint_interval,
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
            # Snapshot la fiecare checkpoint_interval iteratii
            snapshot_file=str(self.output_dir / "checkpoints" / "catboost_snapshot"),
            snapshot_interval=self.checkpoint_interval,
        )

        # Eval set for early stopping
        eval_set = None
        if eval_df is not None:
            X_eval = eval_df[available_features].values
            y_eval = eval_df[target].values
            eval_set = Pool(X_eval, y_eval)

        train_pool = Pool(X_train, y_train)

        t0 = time.time()
        model.fit(train_pool, eval_set=eval_set)
        elapsed = time.time() - t0

        # Train metrics
        y_pred = model.predict(X_train).flatten()
        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(y_train, y_pred)

        metrics = {
            "train_accuracy": train_acc,
            "iterations_used": model.tree_count_,
            "elapsed_sec": elapsed,
            "task_type": task_type,
            "features_used": len(available_features),
        }

        logger.info("CatBoost trained: acc=%.4f, trees=%d, %.1fs (%s)",
                     train_acc, model.tree_count_, elapsed, task_type)

        self._cleanup_gpu()
        return model, metrics

    def train_xgboost(self,
                      train_df: pd.DataFrame,
                      features: List[str],
                      target: str,
                      eval_df: Optional[pd.DataFrame] = None,
                      ) -> Tuple[Any, Dict[str, float]]:
        """Train an XGBoost classifier.

        Args:
            train_df: Training data with features and target.
            features: Feature column names.
            target: Target column name.
            eval_df: Optional evaluation set for early stopping.

        Returns:
            (model, metrics_dict).
        """
        import xgboost as xgb

        available_features = [f for f in features if f in train_df.columns]
        logger.info("Training XGBoost: %d rows, %d features, %d iterations",
                     len(train_df), len(available_features), self.iterations)

        X_train = train_df[available_features].values
        y_train = train_df[target].values

        # GPU: tree_method='gpu_hist' pe GTX 1070
        tree_method = "gpu_hist" if self._gpu_available() else "hist"

        model = xgb.XGBClassifier(
            n_estimators=self.iterations,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric="error",
            random_state=42,
            tree_method=tree_method,
            early_stopping_rounds=self.early_stopping_rounds,
            use_label_encoder=False,
            verbosity=1,
        )

        # Eval set for early stopping
        eval_set_list = None
        if eval_df is not None:
            X_eval = eval_df[available_features].values
            y_eval = eval_df[target].values
            eval_set_list = [(X_eval, y_eval)]

        t0 = time.time()
        model.fit(
            X_train, y_train,
            eval_set=eval_set_list,
            verbose=self.checkpoint_interval,
        )
        elapsed = time.time() - t0

        # Train metrics
        y_pred = model.predict(X_train)
        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(y_train, y_pred)

        metrics = {
            "train_accuracy": train_acc,
            "iterations_used": model.best_iteration if hasattr(model, "best_iteration") else self.iterations,
            "elapsed_sec": elapsed,
            "tree_method": tree_method,
            "features_used": len(available_features),
        }

        logger.info("XGBoost trained: acc=%.4f, best_iter=%s, %.1fs (%s)",
                     train_acc, metrics["iterations_used"], elapsed, tree_method)

        # Save checkpoint
        ckpt_path = self.output_dir / "checkpoints" / "xgboost_checkpoint.json"
        model.save_model(str(ckpt_path))

        self._cleanup_gpu()
        return model, metrics

    def save_model(self, model: Any, name: str, model_type: str) -> Path:
        """Save trained model to output directory.

        Args:
            model: Trained model object.
            name: Model filename (without extension).
            model_type: 'catboost' or 'xgboost'.

        Returns:
            Path to saved model file.
        """
        if model_type == "catboost":
            path = self.output_dir / f"{name}.cbm"
            model.save_model(str(path))
        else:
            import joblib
            path = self.output_dir / f"{name}.pkl"
            joblib.dump(model, path)

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
                logger.info("GPU available: %s (%d MB VRAM)",
                             torch.cuda.get_device_name(0),
                             torch.cuda.get_device_properties(0).total_mem // 1024 // 1024)
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
# EVALUATOR
# ============================================================

class Evaluator:
    """Evaluate retrained models with time-series cross-validation.

    Uses TimeSeriesCrossValidator (5-fold expanding window) and
    ModelEvaluator.cross_validate() from ml/cross_validation.py.

    Args:
        n_splits: Number of CV folds.
        min_train_size: Minimum training rows for first fold.
        test_size: Test rows per fold.
        gap_bars: Gap between train end and test start.
    """

    def __init__(self,
                 n_splits: int = 5,
                 min_train_size: int = 500,
                 test_size: int = 100,
                 gap_bars: int = 10) -> None:
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
                 model_name: str = "unknown",
                 ) -> CVResult:
        """Run cross-validation and check for overfitting.

        Args:
            model: Sklearn-compatible model.
            df: DataFrame with features and target.
            features: Feature column names.
            target: Target column name.
            model_name: Label for logging.

        Returns:
            CVResult with per-fold metrics.
        """
        available_features = [f for f in features if f in df.columns]

        logger.info("Evaluating %s: %d rows, %d features, %d folds",
                     model_name, len(df), len(available_features), self.cv.n_splits)

        result = self._evaluator.cross_validate(
            model=model,
            df=df,
            features=available_features,
            target=target,
            cv=self.cv,
            verbose=True,
        )

        # Flag suspicious accuracy
        if result.mean_score > SUSPICIOUS_ACCURACY_THRESHOLD:
            logger.warning(
                "SUSPICIOUS: %s mean accuracy %.4f > %.0f%% threshold. "
                "Realistic trading models achieve 52-60%%. "
                "Possible data leakage or overfitting.",
                model_name, result.mean_score,
                SUSPICIOUS_ACCURACY_THRESHOLD * 100,
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
                        cutoff_info: str) -> None:
        """Generate Markdown training report.

        Args:
            results: model_name -> CVResult
            train_metrics: model_name -> training metrics dict
            output_path: Path to save the .md report
            symbols: Symbols used for training
            cutoff_info: Train/test split description
        """
        lines = [
            "# ML Training Report",
            "",
            f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Symbols**: {', '.join(symbols)}",
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
            if cv_result.mean_score > SUSPICIOUS_ACCURACY_THRESHOLD:
                status = "SUSPICIOUS (>70%)"
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
                f"- **Compute**: {tm.get('task_type', tm.get('tree_method', 'N/A'))}",
                f"- **Features**: {tm.get('features_used', 'N/A')}",
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

        # Realistic accuracy guidance
        lines.extend([
            "## Accuracy Guidelines",
            "",
            "- **52-55%**: Realistic for daily price direction prediction",
            "- **55-60%**: Good model, verify with walk-forward optimization",
            "- **60-65%**: Excellent, but verify no data leakage",
            "- **65-70%**: Suspicious, check for look-ahead bias",
            "- **>70%**: Almost certainly overfitted or data leakage",
            "",
            "---",
            "*Generated by scripts/retrain_ml_models.py*",
        ])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Report saved: %s", output_path)


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
    # Load existing hashes
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
# MAIN PIPELINE
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Market Hawk ML Model Retraining Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/retrain_ml_models.py --symbols AAPL MSFT --dry-run
  python scripts/retrain_ml_models.py --symbols AAPL --cutoff-date 2025-06-01
  python scripts/retrain_ml_models.py --symbols AAPL MSFT GOOGL --output-dir models/experiment_1
        """,
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
        help="Ticker symbols to train on (default: 8 tech stocks)",
    )
    parser.add_argument(
        "--cutoff-date", type=str, default=None,
        help="Temporal split cutoff date (ISO format, e.g., 2025-06-01). "
             "Default: 80%% of data.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(MODELS_DIR / "retrained"),
        help="Directory to save retrained models (default: models/retrained/)",
    )
    parser.add_argument(
        "--period", type=str, default="2y",
        help="yfinance data period (default: 2y)",
    )
    parser.add_argument(
        "--interval", type=str, default="1d",
        help="Candle interval (default: 1d)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU training (use CPU only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without training",
    )
    parser.add_argument(
        "--skip-xgboost", action="store_true",
        help="Skip XGBoost training (CatBoost only)",
    )
    return parser.parse_args()


def dry_run_report(args: argparse.Namespace) -> None:
    """Print what the pipeline would do without executing."""
    output_dir = Path(args.output_dir)
    print("\n" + "=" * 60)
    print("DRY RUN — ML Retraining Plan")
    print("=" * 60)
    print(f"\n  Symbols:     {', '.join(args.symbols)}")
    print(f"  Period:      {args.period}")
    print(f"  Interval:    {args.interval}")
    print(f"  Cutoff:      {args.cutoff_date or 'auto (80% train / 20% test)'}")
    print(f"  Gap:         5 days (anti-leakage)")
    print(f"  Output dir:  {output_dir}")
    print(f"  GPU:         {'disabled' if args.no_gpu else 'auto-detect'}")
    print(f"  Models:      CatBoost{'' if args.skip_xgboost else ' + XGBoost'}")
    print(f"\n  Features:    {len(FEATURES_60)} (TA-Lib based)")
    print(f"  Target:      binary (price_up = Close[t+1] > Close[t])")
    print(f"  CV:          5-fold expanding window (gap=10 bars)")

    print(f"\n  Training params:")
    print(f"    max_depth:             6")
    print(f"    learning_rate:         0.05")
    print(f"    iterations:            1000")
    print(f"    early_stopping:        50 rounds")
    print(f"    checkpoint_interval:   100 iterations")

    print(f"\n  Hardware:")
    print(f"    CPU: {HARDWARE.cpu_name} ({HARDWARE.cpu_cores} cores)")
    print(f"    GPU: {HARDWARE.gpu_name} ({HARDWARE.gpu_vram_gb}GB VRAM)")
    print(f"    RAM: {HARDWARE.ram_gb}GB {HARDWARE.ram_type}")

    print(f"\n  Output files (planned):")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"    {output_dir / f'catboost_{ts}.cbm'}")
    if not args.skip_xgboost:
        print(f"    {output_dir / f'xgboost_{ts}.pkl'}")
    print(f"    {MODELS_DIR / 'model_hashes.json'} (updated)")
    print(f"    docs/ML_TRAINING_REPORT.md")

    print(f"\n  Existing models (NOT overwritten):")
    for name, entry in MODEL_REGISTRY.items():
        exists = Path(entry["path"]).exists()
        status = "EXISTS" if exists else "MISSING"
        print(f"    {name}: acc={entry['accuracy']} [{status}]")

    print(f"\n  Accuracy flag: > {SUSPICIOUS_ACCURACY_THRESHOLD:.0%} = SUSPICIOUS")
    print("\n" + "=" * 60)
    print("Dry run complete. Remove --dry-run to execute.")
    print("=" * 60 + "\n")


def main() -> None:
    """Main retraining pipeline."""
    setup_logging(logging.INFO)
    args = parse_args()

    if args.dry_run:
        dry_run_report(args)
        return

    t0 = time.time()
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("ML RETRAINING PIPELINE START")
    logger.info("Symbols: %s", args.symbols)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)

    # Optional tqdm import
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # Step 1: Load data
    logger.info("Step 1/4: Loading and preparing data...")
    pipeline = DataPipeline(
        symbols=args.symbols,
        period=args.period,
        interval=args.interval,
        cutoff_date=args.cutoff_date,
        gap_days=5,
    )
    combined = pipeline.load_multi_symbol()
    if combined is None:
        logger.error("No data loaded. Aborting.")
        sys.exit(1)

    features = pipeline.get_feature_columns()

    # Step 2: Temporal split
    logger.info("Step 2/4: Temporal train/test split...")
    train_df, test_df = pipeline.temporal_split(combined)

    if len(train_df) < 100 or len(test_df) < 50:
        logger.error("Insufficient data: train=%d, test=%d. Need at least 100/50.",
                      len(train_df), len(test_df))
        sys.exit(1)

    cutoff_info = (f"train={len(train_df)} rows, test={len(test_df)} rows, "
                   f"gap=5 days")

    # Step 3: Train models
    logger.info("Step 3/4: Training models...")
    trainer = ModelTrainer(
        output_dir=output_dir,
        use_gpu=not args.no_gpu,
    )

    model_paths: Dict[str, Path] = {}
    train_metrics: Dict[str, Dict] = {}
    trained_models: Dict[str, Any] = {}

    # CatBoost
    logger.info("--- Training CatBoost ---")
    cb_model, cb_metrics = trainer.train_catboost(
        train_df, features, TARGET_COL, eval_df=test_df,
    )
    cb_name = f"catboost_retrained_{timestamp}"
    cb_path = trainer.save_model(cb_model, cb_name, "catboost")
    model_paths[cb_name] = cb_path
    train_metrics[cb_name] = cb_metrics
    trained_models[cb_name] = cb_model
    gc.collect()

    # XGBoost
    if not args.skip_xgboost:
        logger.info("--- Training XGBoost ---")
        xgb_model, xgb_metrics = trainer.train_xgboost(
            train_df, features, TARGET_COL, eval_df=test_df,
        )
        xgb_name = f"xgboost_retrained_{timestamp}"
        xgb_path = trainer.save_model(xgb_model, xgb_name, "xgboost")
        model_paths[xgb_name] = xgb_path
        train_metrics[xgb_name] = xgb_metrics
        trained_models[xgb_name] = xgb_model
        gc.collect()

    # Step 4: Evaluate with cross-validation
    logger.info("Step 4/4: Cross-validation evaluation...")
    evaluator = Evaluator(
        n_splits=5,
        min_train_size=max(500, len(train_df) // 3),
        test_size=max(100, len(train_df) // 10),
        gap_bars=10,
    )

    cv_results: Dict[str, CVResult] = {}
    for name, model in trained_models.items():
        cv_result = evaluator.evaluate(
            model=model,
            df=train_df,
            features=features,
            target=TARGET_COL,
            model_name=name,
        )
        cv_results[name] = cv_result
        gc.collect()

    # Generate report
    report_path = _PROJECT_ROOT / "docs" / "ML_TRAINING_REPORT.md"
    evaluator.generate_report(
        results=cv_results,
        train_metrics=train_metrics,
        output_path=report_path,
        symbols=args.symbols,
        cutoff_info=cutoff_info,
    )

    # Update model hashes
    hash_file = MODELS_DIR / "model_hashes.json"
    update_model_hashes(model_paths, hash_file)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("RETRAINING COMPLETE in %.1fs", elapsed)
    logger.info("Models saved to: %s", output_dir)
    logger.info("Report: %s", report_path)
    logger.info("Hashes: %s", hash_file)
    logger.info("=" * 60)

    # Final cleanup
    for model in trained_models.values():
        del model
    gc.collect()
    ModelTrainer._cleanup_gpu()


if __name__ == "__main__":
    main()
