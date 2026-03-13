#!/usr/bin/env python3
"""
Spark-enhanced ML retraining pipeline for Market Hawk MVP.

Replaces the sequential ``DataPipeline.load_multi_symbol()`` loop with a
distributed Spark feature-engineering pass, then uses the existing
``ModelTrainer`` and ``Evaluator`` classes for training and evaluation.

Usage::

    # Dry-run: show plan without executing
    python scripts/spark_retrain_pipeline.py --symbols AAPL MSFT --dry-run

    # Standard run (2-year daily data, auto 80/20 split)
    python scripts/spark_retrain_pipeline.py --symbols AAPL MSFT GOOGL

    # Custom cutoff date
    python scripts/spark_retrain_pipeline.py --symbols AAPL --cutoff-date 2025-06-01

    # GPU disabled, CatBoost only
    python scripts/spark_retrain_pipeline.py --symbols AAPL --no-gpu --skip-xgboost

Hardware: Intel i7-9700F (8 cores), NVIDIA GTX 1070 8 GB VRAM, 64 GB DDR4.
"""

import argparse
import gc
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Ensure project root is on sys.path when executed directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import HARDWARE, ML_CONFIG, MODELS_DIR, setup_logging
from agents.ml_signal_engine.catboost_predictor import (
    FEATURES_60,
    MODEL_REGISTRY,
)
from scripts.retrain_ml_models import (
    ModelTrainer,
    Evaluator,
    update_model_hashes,
    SUSPICIOUS_ACCURACY_THRESHOLD,
    TARGET_COL,
    DEFAULT_SYMBOLS,
)

logger = logging.getLogger("market_hawk.spark.retrain")


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments — mirrors ``scripts/retrain_ml_models.py``."""
    parser = argparse.ArgumentParser(
        description="Spark-enhanced ML retraining pipeline for Market Hawk MVP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Ticker symbols to train on",
    )
    parser.add_argument(
        "--cutoff-date",
        default=None,
        help="Temporal split cutoff date (YYYY-MM-DD). "
             "Default: auto 80%% train / 20%% test.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MODELS_DIR),
        help="Directory for saved model files",
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="yfinance period string (e.g. '2y', '5y')",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Candle interval (e.g. '1d', '1h')",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU training (CPU only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Train CatBoost only (skip XGBoost)",
    )
    return parser.parse_args()


# ============================================================
# DRY-RUN
# ============================================================

def _dry_run_report(args: argparse.Namespace) -> None:
    """Print the training plan without executing anything."""
    output_dir = Path(args.output_dir)
    print("\n" + "=" * 60)
    print("DRY RUN — Spark ML Retraining Plan")
    print("=" * 60)
    print(f"\n  Symbols:     {', '.join(args.symbols)}")
    print(f"  Period:      {args.period}")
    print(f"  Interval:    {args.interval}")
    print(f"  Cutoff:      {args.cutoff_date or 'auto (80% train / 20% test)'}")
    print(f"  Gap:         5 days (anti-leakage)")
    print(f"  Output dir:  {output_dir}")
    print(f"  GPU:         {'disabled' if args.no_gpu else 'auto-detect'}")
    print(
        f"  Models:      CatBoost"
        f"{'' if args.skip_xgboost else ' + XGBoost'}"
    )
    print(f"\n  Data loader: SparkFeaturePipeline (distributed)")
    print(f"  Spark:       local[8] — 8 CPU cores for feature engineering")
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
    print(f"    GPU: {HARDWARE.gpu_name} ({HARDWARE.gpu_vram_gb} GB VRAM)")
    print(f"    RAM: {HARDWARE.ram_gb} GB {HARDWARE.ram_type}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n  Output files (planned):")
    print(f"    {output_dir / f'catboost_spark_{ts}.cbm'}")
    if not args.skip_xgboost:
        print(f"    {output_dir / f'xgboost_spark_{ts}.pkl'}")
    print(f"    {MODELS_DIR / 'model_hashes.json'} (updated)")
    print(f"    docs/ML_TRAINING_REPORT_SPARK.md")

    print(f"\n  Existing models (NOT overwritten):")
    for name, entry in MODEL_REGISTRY.items():
        exists = Path(entry["path"]).exists()
        status = "EXISTS" if exists else "MISSING"
        print(f"    {name}: acc={entry['accuracy']} [{status}]")

    print(f"\n  Accuracy flag: > {SUSPICIOUS_ACCURACY_THRESHOLD:.0%} = SUSPICIOUS")
    print("\n" + "=" * 60)
    print("Dry run complete. Remove --dry-run to execute.")
    print("=" * 60 + "\n")


# ============================================================
# TEMPORAL SPLIT (mirrors DataPipeline.temporal_split)
# ============================================================

def _temporal_split(
    df: pd.DataFrame,
    cutoff_date: Optional[str],
    gap_days: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* temporally with an anti-leakage gap.

    Mirrors the logic of ``DataPipeline.temporal_split()`` from
    ``scripts/retrain_ml_models.py``.

    Args:
        df:           Combined pandas DataFrame sorted by date index.
        cutoff_date:  ISO date string for the split boundary.  If
                      ``None``, uses 80 % of rows as the train set.
        gap_days:     Number of calendar days to exclude between train
                      end and test start (default: 5).

    Returns:
        ``(train_df, test_df)`` tuple.
    """
    if cutoff_date:
        cutoff = pd.Timestamp(cutoff_date)
    else:
        n = len(df)
        cutoff_idx = int(n * 0.80)
        if isinstance(df.index, pd.DatetimeIndex):
            cutoff = df.index[cutoff_idx]
        else:
            train = df.iloc[:cutoff_idx]
            test = df.iloc[cutoff_idx + gap_days:]
            logger.info(
                "Temporal split (integer): train=%d, gap=%d, test=%d",
                len(train), gap_days, len(test),
            )
            return train, test

    gap_end = cutoff + pd.Timedelta(days=gap_days)
    train = df[df.index < cutoff]
    test = df[df.index >= gap_end]

    logger.info(
        "Temporal split: train=%d (< %s), gap=%d days, test=%d (>= %s)",
        len(train), cutoff.date(), gap_days, len(test), gap_end.date(),
    )
    if len(train) == 0:
        logger.error("Train set is empty — cutoff date too early.")
    if len(test) == 0:
        logger.error("Test set is empty — cutoff date too late.")

    return train, test


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    """Main Spark retraining pipeline."""
    setup_logging(logging.INFO)
    args = _parse_args()

    if args.dry_run:
        _dry_run_report(args)
        return

    from pyspark.sql import functions as F  # type: ignore[import]
    from pyspark.sql import Window  # type: ignore[import]
    from config.spark_config import stop_spark
    from data.spark_feature_pipeline import SparkFeaturePipeline

    t0 = time.time()
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("SPARK ML RETRAINING PIPELINE START")
    logger.info("Symbols: %s", args.symbols)
    logger.info("Output:  %s", output_dir)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Distributed feature engineering via Spark
    # ------------------------------------------------------------------
    logger.info("Step 1/4: Distributed feature engineering (Spark)...")
    pipeline = SparkFeaturePipeline()

    sdf = pipeline.process_symbols(
        symbols=args.symbols,
        period=args.period,
        interval=args.interval,
    )

    # ------------------------------------------------------------------
    # Step 2: Create binary target with Spark Window lead()
    # Anti look-ahead: lead(Close, 1) gives the NEXT bar's close price;
    # we use it only as the TARGET label, not as a feature.
    # ------------------------------------------------------------------
    logger.info("Step 2/4: Creating binary target (Spark lead)...")

    w_sym_date = Window.partitionBy("symbol").orderBy("date")
    sdf = sdf.withColumn(
        "_next_close",
        F.lead("Close", 1).over(w_sym_date),
    )
    sdf = sdf.withColumn(
        TARGET_COL,
        (F.col("_next_close") > F.col("Close")).cast("int"),
    )
    # Drop rows where target is null (last row of each symbol)
    sdf = sdf.filter(F.col(TARGET_COL).isNotNull())
    sdf = sdf.drop("_next_close")

    # ------------------------------------------------------------------
    # Step 3: Collect to pandas for model training
    # ------------------------------------------------------------------
    logger.info("Step 3/4: Collecting Spark → pandas for model training...")
    combined: pd.DataFrame = pipeline.to_pandas(sdf)
    del sdf
    gc.collect()

    # Set date as index for temporal split (matching DataPipeline behaviour)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.set_index("date").sort_index()

    features = list(FEATURES_60)

    # Validate feature availability
    missing_features = [f for f in features if f not in combined.columns]
    if missing_features:
        logger.warning(
            "%d FEATURES_60 columns not found in data (will be skipped): %s",
            len(missing_features), missing_features,
        )
        features = [f for f in features if f in combined.columns]

    # Temporal split
    train_df, test_df = _temporal_split(
        combined,
        cutoff_date=args.cutoff_date,
        gap_days=5,
    )

    if len(train_df) < 100 or len(test_df) < 50:
        logger.error(
            "Insufficient data: train=%d, test=%d. Need at least 100/50.",
            len(train_df), len(test_df),
        )
        stop_spark()
        sys.exit(1)

    cutoff_info = (
        f"train={len(train_df)} rows, test={len(test_df)} rows, gap=5 days"
    )

    # ------------------------------------------------------------------
    # Step 4: Train models (reuse existing ModelTrainer + Evaluator)
    # ------------------------------------------------------------------
    logger.info("Step 4/4: Training models...")
    trainer = ModelTrainer(
        output_dir=output_dir,
        use_gpu=not args.no_gpu,
    )

    model_paths: Dict[str, Path] = {}
    train_metrics: Dict[str, Dict[str, Any]] = {}
    trained_models: Dict[str, Any] = {}

    # CatBoost
    logger.info("--- Training CatBoost ---")
    cb_model, cb_metrics = trainer.train_catboost(
        train_df, features, TARGET_COL, eval_df=test_df,
    )
    cb_name = f"catboost_spark_{timestamp}"
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
        xgb_name = f"xgboost_spark_{timestamp}"
        xgb_path = trainer.save_model(xgb_model, xgb_name, "xgboost")
        model_paths[xgb_name] = xgb_path
        train_metrics[xgb_name] = xgb_metrics
        trained_models[xgb_name] = xgb_model
        gc.collect()

    # Cross-validation evaluation
    logger.info("Step 4b: Cross-validation evaluation...")
    evaluator = Evaluator(
        n_splits=5,
        min_train_size=max(500, len(train_df) // 3),
        test_size=max(100, len(train_df) // 10),
        gap_bars=10,
    )

    cv_results = {}
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

    # Report
    report_path = _PROJECT_ROOT / "docs" / "ML_TRAINING_REPORT_SPARK.md"
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
    logger.info("SPARK RETRAINING COMPLETE in %.1fs", elapsed)
    logger.info("Models saved to: %s", output_dir)
    logger.info("Report: %s", report_path)
    logger.info("Hashes: %s", hash_file)
    logger.info("=" * 60)

    # Final cleanup
    for model in trained_models.values():
        del model
    gc.collect()

    stop_spark()


if __name__ == "__main__":
    main()
