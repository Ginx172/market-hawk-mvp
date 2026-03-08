# ML Model Retraining Guide

## Overview

`scripts/retrain_ml_models.py` retrains CatBoost and XGBoost classifiers using
OHLCV market data with 60 engineered technical features. Models are evaluated
with time-series cross-validation (no data shuffling, no future leakage).

## Quick Start

```bash
# 1. Dry run (shows plan without training)
python scripts/retrain_ml_models.py --symbols AAPL MSFT --dry-run

# 2. Train with defaults (8 tech stocks, 2y data, daily candles)
python scripts/retrain_ml_models.py

# 3. Train specific symbols with cutoff date
python scripts/retrain_ml_models.py --symbols AAPL GOOGL --cutoff-date 2025-06-01

# 4. CPU-only training (no GPU)
python scripts/retrain_ml_models.py --symbols AAPL --no-gpu

# 5. CatBoost only (skip XGBoost)
python scripts/retrain_ml_models.py --symbols AAPL --skip-xgboost
```

## Pipeline Steps

### Step 1: Data Loading (`DataPipeline`)

- Downloads OHLCV via yfinance (or loads from local CSV/Parquet)
- Applies `adjust_for_splits_dividends()` (backward price adjustment)
- Runs `detect_unadjusted_splits()` (warns about suspicious gaps)
- Engineers 60 TA features via `MarketDataFetcher.engineer_features()`
- Creates binary target: `price_up = Close[t+1] > Close[t]`

### Step 2: Temporal Split

- **Train**: all data before cutoff date (or first 80%)
- **Gap**: 5 calendar days removed between train and test (anti-leakage)
- **Test**: all data after gap end
- No shuffling, no random splits

### Step 3: Model Training (`ModelTrainer`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_depth | 6 | Prevents overfitting on small datasets |
| learning_rate | 0.05 | Slow learning, better generalization |
| iterations | 1000 | Max rounds (early stopping may cut short) |
| early_stopping_rounds | 50 | Stop if no improvement in 50 rounds |
| checkpoint_interval | 100 | Save snapshot every 100 trees |

GPU training is auto-detected (GTX 1070 with CUDA). Falls back to CPU.

### Step 4: Cross-Validation Evaluation (`Evaluator`)

Uses `TimeSeriesCrossValidator` (5-fold expanding window) from `ml/cross_validation.py`:

- Fold 1: train on oldest data, test on next segment
- Fold 2-5: train set grows (expanding window)
- 10-bar gap between train/test per fold
- Reports per fold: accuracy, precision, recall, F1, train accuracy
- Computes `overfit_ratio = train_accuracy / test_accuracy`

## Output Files

```
models/retrained/
  catboost_retrained_YYYYMMDD_HHMMSS.cbm    # CatBoost model
  xgboost_retrained_YYYYMMDD_HHMMSS.pkl     # XGBoost model
  checkpoints/
    catboost_snapshot                          # CatBoost training checkpoint
    xgboost_checkpoint.json                   # XGBoost training checkpoint

models/model_hashes.json                      # Updated SHA-256 hashes
docs/ML_TRAINING_REPORT.md                    # Evaluation report
```

Existing models are NEVER overwritten. New models are saved alongside.

## Accuracy Guidelines

| Range | Interpretation |
|-------|---------------|
| 52-55% | Realistic for daily price direction |
| 55-60% | Good model, validate with walk-forward |
| 60-65% | Excellent, verify no data leakage |
| 65-70% | Suspicious, audit feature pipeline |
| > 70% | Almost certainly overfitted or leaking |

The script flags any model with test accuracy > 70% as **SUSPICIOUS**.
The `overfit_ratio` (train/test accuracy) > 1.5 triggers an **OVERFITTING** warning.

## Anti-Leakage Safeguards

1. **Temporal split only** -- no random train/test split
2. **5-day gap** between train and test sets
3. **10-bar gap** in cross-validation folds
4. **No bfill** in feature engineering (forward-fill only)
5. **No shuffle** anywhere in the pipeline
6. **Target uses shift(-1)** -- explicitly future information, clearly labeled

## Hardware Requirements

Tested on:
- CPU: Intel i7-9700F (8 cores)
- GPU: NVIDIA GTX 1070 (8GB VRAM)
- RAM: 64GB DDR4-2666

Memory management:
- `gc.collect()` after each symbol load and model train
- `torch.cuda.empty_cache()` after GPU training
- CatBoost snapshot checkpoints for resume after interruption

## Command Reference

```
usage: retrain_ml_models.py [-h] [--symbols SYMBOLS [SYMBOLS ...]]
                            [--cutoff-date CUTOFF_DATE]
                            [--output-dir OUTPUT_DIR]
                            [--period PERIOD] [--interval INTERVAL]
                            [--no-gpu] [--dry-run] [--skip-xgboost]

Options:
  --symbols          Ticker symbols (default: AAPL MSFT GOOGL NVDA META TSLA AMD AMZN)
  --cutoff-date      ISO date for train/test split (default: 80% of data)
  --output-dir       Where to save models (default: models/retrained/)
  --period           yfinance period (default: 2y)
  --interval         Candle interval (default: 1d)
  --no-gpu           Force CPU-only training
  --dry-run          Show plan without executing
  --skip-xgboost     Train CatBoost only
```

## Existing Model Registry

Models in `agents/ml_signal_engine/catboost_predictor.py :: MODEL_REGISTRY`:

| Model | Type | Accuracy | Notes |
|-------|------|----------|-------|
| catboost_v2 | .cbm | TBD | Primary production model |
| catboost_clean_75 | .pkl | 75% | Rebuilt, needs validation |
| catboost_ultra_75 | .pkl | 75% | Ultra-tuned |
| catboost_commodities_82 | .pkl | 82% | Commodities specialist |
| catboost_indices_76 | .pkl | 76% | Indices specialist |
| catboost_crypto_56 | .pkl | 56% | Needs improvement |
| catboost_80plus | .pkl | 100% | Likely overfitted |
| xgboost_80plus | .pkl | 99.96% | Likely overfitted |
| extratrees_80plus | .pkl | 98.4% | Needs validation |

All stored on G: drive. SHA-256 hashes in `models/model_hashes.json`.
