"""
SCRIPT NAME: cross_validation.py
====================================
Execution Location: market-hawk-mvp/ml/
Purpose: Time-series-aware cross-validation framework for ML models.
         Never shuffles data — preserves temporal ordering.
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-07

Provides:
    - TimeSeriesCrossValidator: Expanding or Sliding window CV splits
    - ModelEvaluator: cross_validate() with per-fold metrics and overfit detection
    - CVResult: aggregate metrics dataclass

Usage:
    cv = TimeSeriesCrossValidator(n_splits=5, min_train_size=500, test_size=100, gap_bars=10)
    evaluator = ModelEvaluator()
    result = evaluator.cross_validate(model, df, feature_cols, target_col, cv)
    print(result.mean_score, result.overfit_ratio)
"""

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger("market_hawk.ml.cross_validation")


# ============================================================
# FOLD METRICS
# ============================================================

@dataclass
class FoldResult:
    """Metrics for a single CV fold."""
    fold_idx: int
    train_size: int
    test_size: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_accuracy: float  # For overfit detection


@dataclass
class CVResult:
    """Aggregated cross-validation results."""
    n_folds: int
    fold_results: List[FoldResult] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def fold_scores(self) -> List[Dict[str, float]]:
        """Per-fold metric dicts."""
        return [
            {
                "accuracy": f.accuracy,
                "precision": f.precision,
                "recall": f.recall,
                "f1": f.f1,
                "train_accuracy": f.train_accuracy,
            }
            for f in self.fold_results
        ]

    @property
    def mean_score(self) -> float:
        """Mean test accuracy across folds."""
        if not self.fold_results:
            return 0.0
        return float(np.mean([f.accuracy for f in self.fold_results]))

    @property
    def std_score(self) -> float:
        """Std of test accuracy across folds."""
        if len(self.fold_results) < 2:
            return 0.0
        return float(np.std([f.accuracy for f in self.fold_results]))

    @property
    def overfit_ratio(self) -> float:
        """Mean train_accuracy / mean test_accuracy.

        > 1.5 suggests overfitting.
        """
        if not self.fold_results:
            return 0.0
        mean_train = float(np.mean([f.train_accuracy for f in self.fold_results]))
        mean_test = self.mean_score
        if mean_test <= 0:
            return float("inf")
        return mean_train / mean_test

    @property
    def is_overfit(self) -> bool:
        """True if overfit_ratio > 1.5."""
        return self.overfit_ratio > 1.5

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"CV Results: {self.n_folds} folds, {self.elapsed_sec:.1f}s",
            f"  Test accuracy:  {self.mean_score:.4f} +/- {self.std_score:.4f}",
            f"  Overfit ratio:  {self.overfit_ratio:.2f}"
            f" {'(WARNING: overfitting)' if self.is_overfit else '(OK)'}",
        ]
        for f in self.fold_results:
            lines.append(
                f"  Fold {f.fold_idx}: acc={f.accuracy:.4f} prec={f.precision:.4f} "
                f"rec={f.recall:.4f} f1={f.f1:.4f} | train_acc={f.train_accuracy:.4f} "
                f"[train={f.train_size}, test={f.test_size}]"
            )
        return "\n".join(lines)


# ============================================================
# TIME SERIES CROSS VALIDATOR
# ============================================================

class TimeSeriesCrossValidator:
    """Time-series-aware cross-validation splitter.

    Two modes:
        - Expanding Window (default): train set grows with each fold.
          Fold 1 trains on [0:min_train], Fold 2 on [0:min_train+step], etc.
        - Sliding Window: train window is fixed size and slides forward.

    A gap between train and test prevents information leakage from
    indicators that use look-ahead (e.g., smoothed values near boundary).

    Args:
        n_splits: Number of CV folds.
        min_train_size: Minimum training samples for the first fold.
        test_size: Number of test samples per fold.
        gap_bars: Gap between train end and test start (leak protection).
        mode: 'expanding' or 'sliding'.
    """

    def __init__(self,
                 n_splits: int = 5,
                 min_train_size: int = 500,
                 test_size: int = 100,
                 gap_bars: int = 10,
                 mode: str = "expanding") -> None:
        if mode not in ("expanding", "sliding"):
            raise ValueError(f"mode must be 'expanding' or 'sliding', got '{mode}'")
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if min_train_size < 1 or test_size < 1:
            raise ValueError("min_train_size and test_size must be >= 1")

        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap_bars = max(0, gap_bars)
        self.mode = mode

    def split(self, df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) for each fold.

        Args:
            df: DataFrame to split (uses integer indices).

        Yields:
            Tuples of (train_indices, test_indices) as numpy arrays.

        Raises:
            ValueError: If data is too small for the requested splits.
        """
        n = len(df)
        required = self.min_train_size + self.gap_bars + self.test_size
        if n < required:
            raise ValueError(
                f"Data too small ({n} rows) for min_train={self.min_train_size} + "
                f"gap={self.gap_bars} + test={self.test_size} = {required}"
            )

        # Total space needed for all test windows (excluding the first fold's train set)
        available = n - self.min_train_size - self.gap_bars - self.test_size
        if available < 0:
            raise ValueError("Not enough data for even 1 fold")

        step = max(1, available // max(1, self.n_splits - 1)) if self.n_splits > 1 else 0

        for fold in range(self.n_splits):
            if self.mode == "expanding":
                train_start = 0
                train_end = self.min_train_size + fold * step
            else:  # sliding
                train_start = fold * step
                train_end = train_start + self.min_train_size

            test_start = train_end + self.gap_bars
            test_end = test_start + self.test_size

            if test_end > n:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            # Sanity: no overlap
            assert train_idx[-1] < test_idx[0], (
                f"Fold {fold}: train/test overlap detected"
            )

            yield train_idx, test_idx

    def describe(self, n_rows: int) -> str:
        """Describe the split plan for a dataset of given size."""
        lines = [f"TimeSeriesCrossValidator(mode={self.mode}, n_splits={self.n_splits}, "
                 f"min_train={self.min_train_size}, test={self.test_size}, "
                 f"gap={self.gap_bars})"]
        dummy = pd.DataFrame(np.zeros((n_rows, 1)), columns=["x"])
        for i, (train_idx, test_idx) in enumerate(self.split(dummy)):
            lines.append(
                f"  Fold {i}: train[{train_idx[0]}:{train_idx[-1]+1}] ({len(train_idx)}) "
                f"-> gap({self.gap_bars}) -> test[{test_idx[0]}:{test_idx[-1]+1}] ({len(test_idx)})"
            )
        return "\n".join(lines)


# ============================================================
# MODEL EVALUATOR
# ============================================================

class ModelEvaluator:
    """Evaluate ML models with time-series cross-validation.

    Supports any model with sklearn-compatible fit/predict interface
    (CatBoost, XGBoost, LightGBM, sklearn classifiers).
    """

    def cross_validate(self,
                       model: Any,
                       df: pd.DataFrame,
                       features: List[str],
                       target: str,
                       cv: TimeSeriesCrossValidator,
                       verbose: bool = True) -> CVResult:
        """Run time-series cross-validation.

        Args:
            model: Sklearn-compatible model with fit() and predict().
                   A deep copy is made for each fold.
            df: DataFrame with feature columns and target column.
            features: List of feature column names.
            target: Target column name.
            cv: TimeSeriesCrossValidator instance.
            verbose: Log progress per fold.

        Returns:
            CVResult with per-fold and aggregate metrics.
        """
        t0 = time.time()
        fold_results: List[FoldResult] = []

        missing = [c for c in features + [target] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

        X = df[features].values
        y = df[target].values

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df)):
            fold_t0 = time.time()

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Deep copy model to avoid state leakage between folds
            fold_model = copy.deepcopy(model)

            # Fit — handle CatBoost params (use_best_model needs eval_set)
            fit_kwargs: Dict[str, Any] = {}
            if hasattr(fold_model, "get_params"):
                params = fold_model.get_params()
                if "verbose" in params:
                    fit_kwargs["verbose"] = False
                # CatBoost: use_best_model=True requires eval_set; pass it
                if params.get("use_best_model"):
                    fit_kwargs["eval_set"] = (X_test, y_test)
                elif "use_best_model" in params:
                    # Explicitly False — no eval_set needed
                    pass
                elif hasattr(fold_model, "_init_params"):
                    # CatBoost default has use_best_model=True; disable it
                    # to avoid crash when no eval_set is provided
                    fold_model.set_params(use_best_model=False)

                # XGBoost: early_stopping_rounds requires eval_set
                # Format: list of tuples [(X, y)], not a bare tuple
                if params.get("early_stopping_rounds") and "eval_set" not in fit_kwargs:
                    fit_kwargs["eval_set"] = [(X_test, y_test)]

            fold_model.fit(X_train, y_train, **fit_kwargs)

            # Predict
            y_pred_test = fold_model.predict(X_test)
            y_pred_train = fold_model.predict(X_train)

            # Flatten predictions if needed (CatBoost returns 2D sometimes)
            if hasattr(y_pred_test, "flatten"):
                y_pred_test = y_pred_test.flatten()
            if hasattr(y_pred_train, "flatten"):
                y_pred_train = y_pred_train.flatten()

            # Metrics (zero_division=0 for edge cases)
            test_acc = accuracy_score(y_test, y_pred_test)
            test_prec = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
            test_rec = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
            train_acc = accuracy_score(y_train, y_pred_train)

            fr = FoldResult(
                fold_idx=fold_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                train_start=int(train_idx[0]),
                train_end=int(train_idx[-1]),
                test_start=int(test_idx[0]),
                test_end=int(test_idx[-1]),
                accuracy=test_acc,
                precision=test_prec,
                recall=test_rec,
                f1=test_f1,
                train_accuracy=train_acc,
            )
            fold_results.append(fr)

            fold_sec = time.time() - fold_t0
            if verbose:
                overfit_flag = " [OVERFIT]" if train_acc > 0 and test_acc > 0 and train_acc / test_acc > 1.5 else ""
                logger.info(
                    "  Fold %d/%d: acc=%.4f prec=%.4f f1=%.4f | "
                    "train_acc=%.4f | train=%d test=%d (%.1fs)%s",
                    fold_idx + 1, cv.n_splits,
                    test_acc, test_prec, test_f1,
                    train_acc, len(train_idx), len(test_idx),
                    fold_sec, overfit_flag,
                )

            # Free fold model
            del fold_model

        elapsed = time.time() - t0
        result = CVResult(
            n_folds=len(fold_results),
            fold_results=fold_results,
            elapsed_sec=elapsed,
        )

        logger.info("CV complete: mean_acc=%.4f +/- %.4f | overfit_ratio=%.2f | %.1fs",
                     result.mean_score, result.std_score, result.overfit_ratio, elapsed)

        if result.is_overfit:
            logger.warning("OVERFITTING DETECTED: train/test ratio=%.2f (threshold=1.5)",
                           result.overfit_ratio)

        return result


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("\n" + "=" * 60)
    print("TIME SERIES CROSS-VALIDATION — Demo")
    print("=" * 60)

    # Generate synthetic classification data
    np.random.seed(42)
    n = 2000
    X = np.random.randn(n, 10)
    # Simple rule: class 1 if feature 0 + feature 1 > 0
    y = (X[:, 0] + X[:, 1] + np.random.randn(n) * 0.5 > 0).astype(int)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["target"] = y

    feature_cols = [f"f{i}" for i in range(10)]

    # Test expanding window CV
    print("\n--- Expanding Window CV ---")
    cv = TimeSeriesCrossValidator(n_splits=5, min_train_size=500, test_size=200, gap_bars=10)
    print(cv.describe(n))

    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)

    evaluator = ModelEvaluator()
    result = evaluator.cross_validate(model, df, feature_cols, "target", cv)
    print("\n" + result.summary())

    # Test sliding window CV
    print("\n--- Sliding Window CV ---")
    cv_slide = TimeSeriesCrossValidator(
        n_splits=5, min_train_size=500, test_size=200, gap_bars=10, mode="sliding"
    )
    print(cv_slide.describe(n))
    result2 = evaluator.cross_validate(model, df, feature_cols, "target", cv_slide)
    print("\n" + result2.summary())

    print("\nDone.")
