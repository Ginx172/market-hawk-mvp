# Crypto ML Improvement Plan

## Problem Statement

The `catboost_crypto_56` model achieves only **56% accuracy** — the worst in the
model registry. Root cause analysis identified five key problems:

| # | Root Cause | Severity |
|---|-----------|---------|
| 1 | Training data uses only stock symbols (AAPL, MSFT, GOOGL…) — no crypto | 🔴 Critical |
| 2 | Same 60 generic features for all asset types | 🔴 Critical |
| 3 | Single timeframe (1d) — crypto markets are 24/7 | 🟡 Major |
| 4 | Simplistic binary target (`Close[t+1] > Close[t]`) — no noise filter | 🟡 Major |
| 5 | No crypto-specific indicators (volatility ratios, OBV momentum, sessions) | 🟡 Major |

---

## Solution Architecture

### New Files

| File | Purpose |
|------|---------|
| `scripts/retrain_crypto_specialist.py` | Main training pipeline (crypto-specialist) |
| `data/crypto_feature_engineer.py` | 20 additional crypto-specific features |
| `docs/CRYPTO_ML_IMPROVEMENT_PLAN.md` | This document |

### Modified Files

| File | Change |
|------|--------|
| `agents/ml_signal_engine/catboost_predictor.py` | Added `CRYPTO_EXTRA_FEATURES`, `CRYPTO_FEATURES`, updated `MODEL_REGISTRY` |

---

## Feature Engineering

### Base features (60) — unchanged

Produced by `MarketDataFetcher.engineer_features()`. See `data/market_data_fetcher.py`
for the full list.

### Crypto-specific features (20) — new

Produced by `data/crypto_feature_engineer.py → CryptoFeatureEngineer.add_features()`.

#### Volatility (4)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `crypto_volatility_ratio` | `std(returns,10) / std(returns,50)` | Detects vol regime changes |
| `crypto_range_pct` | `(High−Low) / Close × 100` | Intraday range as % |
| `crypto_atr_pct` | `ATR(14) / Close × 100` | Normalised ATR |
| `crypto_volatility_zscore` | `(vol_20 − mean_vol) / std_vol` | Z-score of current vol |

#### Volume (4)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `crypto_volume_ratio_8h` | `volume / rolling_mean(volume, 8)` | Short-term volume surge |
| `crypto_volume_ratio_24h` | `volume / rolling_mean(volume, 24)` | Daily volume comparison |
| `crypto_volume_trend` | `mean(vol,5) / mean(vol,20)` | Volume momentum |
| `crypto_obv_momentum` | `OBV.diff(5) / abs(OBV).rolling(20).mean()` | OBV momentum |

#### Momentum (4)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `crypto_momentum_3d` | `returns.rolling(3×ppd).sum()` | 3-day cumulative return |
| `crypto_momentum_7d` | `returns.rolling(7×ppd).sum()` | 7-day cumulative return |
| `crypto_rsi_divergence` | `RSI.diff(5)` | RSI rate of change |
| `crypto_macd_histogram_momentum` | `MACD_hist.diff(3)` | MACD acceleration |

#### Price structure (4)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `crypto_distance_from_high_20` | `(Close − High.rolling(20).max()) / Close` | Distance from 20-period high |
| `crypto_distance_from_low_20` | `(Close − Low.rolling(20).min()) / Close` | Distance from 20-period low |
| `crypto_bb_position` | `(Close − BB_lower) / (BB_upper − BB_lower)` | Position in Bollinger Band |
| `crypto_mean_reversion_signal` | `(Close − SMA_20) / std(Close,20)` | Z-score from mean |

#### Session / cyclical (4)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `crypto_hour_sin` | `sin(2π × hour / 24)` | Cyclical hour encoding |
| `crypto_hour_cos` | `cos(2π × hour / 24)` | Cyclical hour encoding |
| `crypto_is_weekend` | `day_of_week >= 5` | Weekend — less institutional activity |
| `crypto_day_of_week_sin` | `sin(2π × day_of_week / 7)` | Cyclical day encoding |

---

## Target Variable

### Old (binary, noisy)

```python
target = int(Close[t+1] > Close[t])  # Any move, even noise
```

### New (threshold-based)

```python
pct_change = (Close[t+1] - Close[t]) / Close[t]
if pct_change > 0.003:    target = 1  # UP
elif pct_change < -0.003: target = 0  # DOWN
else:                     target = -1  # NEUTRAL — row removed
```

**Effect**: Removes ~20-40% of rows (market noise), leaving cleaner directional signal.
The threshold (0.3%) was chosen as approximately 1/3 of the average crypto daily move.

---

## Multi-Timeframe Feature Stacking

When using 4h candles (`--interval 4h`), each 4h bar is enriched with
**daily-aggregated features** from the same data:

```
4h OHLCV  ──►  engineer_features(60)  ──►  4h feature set
    │
    ▼ resample to 1D
Daily OHLCV ──►  engineer_features(60)  ──► daily_ prefixed features
    │
    ▼ forward-fill (no look-ahead)
4h bars receive daily_ features for their calendar date
```

This gives the model both intraday and daily context per bar.

---

## CatBoost Hyperparameters

| Parameter | Old (`catboost_crypto_56`) | New (`catboost_crypto_v2`) | Reason |
|-----------|--------------------------|--------------------------|--------|
| `iterations` | 1000 | **2000** | More boosting rounds for complex patterns |
| `depth` | 6 | **8** | Deeper trees for crypto feature interactions |
| `learning_rate` | 0.05 | **0.03** | Slower learning → better generalisation |
| `l2_leaf_reg` | default (3) | **5** | Stronger regularisation against overfitting |
| `border_count` | default (254) | **128** | Fewer bins, faster training, less overfitting |
| `early_stopping_rounds` | 50 | **100** | Give model more chance to recover |

---

## Training Data

| Aspect | Old | New |
|--------|-----|-----|
| Symbols | AAPL, MSFT, GOOGL, NVDA, META, TSLA, AMD, AMZN | **BTC-USD, ETH-USD, SOL-USD, BNB-USD, ADA-USD, XRP-USD, AVAX-USD, DOT-USD** |
| Interval | 1d | **4h** (+ daily stacking) |
| Period | 2y | 2y |
| Features | 60 | **80** (60 base + 20 crypto) |
| Target | binary | **threshold 0.3%** |

---

## Cross-Validation

- **Method**: 5-fold expanding window (`TimeSeriesCrossValidator`)
- **Gap**: 24 bars between train end and test start (~1 day for 4h data)
- **Anti-leakage**: No shuffle, strict temporal ordering, forward-fill only

---

## Accuracy Expectations

| Range | Interpretation |
|-------|----------------|
| 50-52% | Random — model is not learning |
| 52-55% | Minimal signal — marginal use |
| **55-63%** | **Target range — realistic for crypto** |
| 63-65% | Excellent — verify no data leakage |
| >65% | **Suspicious — check for look-ahead bias** |

Note: The 56% baseline was trained on wrong data (stocks). Even without
the new features, retraining on actual crypto data alone should push
accuracy to ~56-58%. The additional 20 features and improved target
variable are expected to bring it to the 58-63% range.

---

## Usage

```bash
# Dry run — see what will be done
python scripts/retrain_crypto_specialist.py --dry-run

# Full training with defaults
python scripts/retrain_crypto_specialist.py

# Custom symbols
python scripts/retrain_crypto_specialist.py --symbols BTC-USD ETH-USD SOL-USD

# Daily candles only
python scripts/retrain_crypto_specialist.py --interval 1d --no-daily-stack

# CPU only
python scripts/retrain_crypto_specialist.py --no-gpu
```

---

## Model Registry Integration

After training, the new model is saved as:
```
models/retrained/CatBoost_crypto_v2_{YYYYMMDD_HHMMSS}.cbm
```

Update the `catboost_crypto_v2` entry in `MODEL_REGISTRY`
(`agents/ml_signal_engine/catboost_predictor.py`) with the actual path.
The SHA-256 hash is automatically written to `models/model_hashes.json`.

---

*Created: 2026-03-13 | Author: Market Hawk AI Coding Agent*
