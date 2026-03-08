"""
SCRIPT NAME: catboost_predictor.py
====================================
Execution Location: market-hawk-mvp/agents/ml_signal_engine/
Purpose: ML Signal Engine — Multi-Model Trading Signal Generator
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01
Last Modified: 2026-03-01

Loads trained ML models from G drive model repository.
Supports CatBoost (.cbm), sklearn/xgboost/lightgbm (.pkl) with associated scalers.

AVAILABLE MODELS (from G drive):
    - catboost_v2_advanced (.cbm) — CatBoost native format
    - CatBoost_CLEAN_acc75.pkl — Rebuilt, 75% accuracy
    - CatBoost_ULTRA_acc75.pkl — Ultra-tuned, 75%
    - PER_CATEGORY_MODELS/ — Per-asset models (commodities 82%, indices 76%)
    - Modele_AI_80_Plus/ — Suite of 10 models (needs validation for overfitting)

60 features used across all models (TA-Lib based).
"""

import gc
import hashlib
import hmac
import json
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from pathlib import Path

import joblib

logger = logging.getLogger("market_hawk.ml_signal")

# Directoare trusted din care se pot incarca modele .pkl
# Orice path in afara acestor directoare va fi refuzat
_TRUSTED_MODEL_DIRS: List[Path] = []


def _init_trusted_dirs() -> None:
    """Populate trusted dirs from config (lazy, called once)."""
    if _TRUSTED_MODEL_DIRS:
        return
    try:
        from config.settings import ML_CONFIG
        base = Path(ML_CONFIG.models_base_dir)
        if base.exists():
            _TRUSTED_MODEL_DIRS.append(base.resolve())
    except Exception:
        pass
    # Fallback: allow paths already declared in MODEL_REGISTRY
    for entry in MODEL_REGISTRY.values():
        p = Path(entry["path"]).parent.resolve()
        if p not in _TRUSTED_MODEL_DIRS:
            _TRUSTED_MODEL_DIRS.append(p)


def _is_trusted_path(file_path: Path) -> bool:
    """Verify that file_path is inside a trusted model directory.

    Rejects symlinks and paths containing '..' after resolution.
    """
    _init_trusted_dirs()

    # Reject symlinks (could point outside trusted dirs)
    if file_path.is_symlink():
        logger.warning("Rejected symlink path: %s", file_path)
        return False

    resolved = file_path.resolve()

    # Reject if resolved path still contains '..' components
    if ".." in resolved.parts:
        logger.warning("Rejected path with traversal components: %s", file_path)
        return False

    is_trusted = any(
        resolved == trusted or trusted in resolved.parents
        for trusted in _TRUSTED_MODEL_DIRS
    )
    if not is_trusted:
        logger.warning("Rejected untrusted path: %s (resolved: %s)", file_path, resolved)
    return is_trusted


# ============================================================
# SHA-256 HASH VERIFICATION
# ============================================================

_HASH_FILE = Path(__file__).parent.parent.parent / "models" / "model_hashes.json"


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file, reading in 8KB chunks for memory efficiency."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def _verify_model_hash(model_name: str, model_path: Path) -> bool:
    """Verify SHA-256 hash of a model file against stored hashes.

    Returns True if hash matches or if no hash file/entry exists (backward compatible).
    Returns False if hash mismatch (possible tampering).
    """
    if not _HASH_FILE.exists():
        logger.warning("Hash file %s not found — skipping verification for %s "
                       "(run --generate-hashes to create)", _HASH_FILE, model_name)
        return True

    try:
        with open(_HASH_FILE, "r") as f:
            hash_store = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read hash file %s: %s — skipping verification", _HASH_FILE, e)
        return True

    if model_name not in hash_store:
        logger.warning("No hash entry for model '%s' — skipping verification "
                       "(run --generate-hashes to update)", model_name)
        return True

    expected_hash = hash_store[model_name].get("sha256", "")
    if not expected_hash:
        logger.warning("Empty hash for model '%s' — skipping verification", model_name)
        return True

    actual_hash = _compute_file_hash(model_path)

    if hmac.compare_digest(actual_hash, expected_hash):
        logger.info("Hash verified for %s", model_name)
        return True

    logger.error("HASH MISMATCH for model '%s': expected=%s, actual=%s — "
                 "possible file tampering, refusing to load", model_name, expected_hash, actual_hash)
    return False


def generate_model_hashes() -> Dict[str, Dict[str, str]]:
    """Scan MODEL_REGISTRY, compute SHA-256 hashes, and save to models/model_hashes.json."""
    hash_store: Dict[str, Dict[str, str]] = {}

    for name, entry in MODEL_REGISTRY.items():
        model_path = Path(entry["path"])
        if model_path.exists():
            file_hash = _compute_file_hash(model_path)
            hash_store[name] = {
                "path": str(model_path),
                "sha256": file_hash,
            }
            print(f"  {name}: {file_hash[:16]}... ({model_path.name})")

            # Also hash scaler if present
            if entry.get("scaler"):
                scaler_path = Path(entry["scaler"])
                if scaler_path.exists():
                    scaler_hash = _compute_file_hash(scaler_path)
                    hash_store[f"{name}__scaler"] = {
                        "path": str(scaler_path),
                        "sha256": scaler_hash,
                    }
                    print(f"  {name}__scaler: {scaler_hash[:16]}... ({scaler_path.name})")
        else:
            print(f"  {name}: SKIPPED (file not found: {model_path})")

    _HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_HASH_FILE, "w") as f:
        json.dump(hash_store, f, indent=2)

    print(f"\nSaved {len(hash_store)} hashes to {_HASH_FILE}")
    return hash_store


# ============================================================
# MODEL REGISTRY — All available models
# ============================================================

MODEL_REGISTRY = {
    # === Most reliable (realistic accuracy) ===
    "catboost_v2": {
        "path": r"G:\......................AI_Models\catboost_v2_advanced_20250724_011145.cbm",
        "type": "catboost_native",
        "scaler": None,
        "accuracy": "TBD — needs walk-forward validation",
        "notes": "CatBoost native .cbm format",
    },
    "catboost_clean_75": {
        "path": r"G:\......................AI_Models\REBUILT_MODELS\CatBoost_CLEAN_acc75.pkl",
        "type": "pickle",
        "scaler": None,
        "accuracy": 0.75,
        "notes": "Rebuilt clean, 75% accuracy",
    },
    "catboost_ultra_75": {
        "path": r"G:\......................AI_Models\ULTRA_TUNED_MODELS\CatBoost_ULTRA_acc75.pkl",
        "type": "pickle",
        "scaler": None,
        "accuracy": 0.75,
        "notes": "Ultra-tuned, 75% accuracy",
    },

    # === Per-category models ===
    "catboost_commodities_82": {
        "path": r"G:\......................AI_Models\PER_CATEGORY_MODELS\CatBoost_commodities_acc82.pkl",
        "type": "pickle",
        "scaler": None,
        "accuracy": 0.82,
        "notes": "Commodities specialist",
    },
    "catboost_indices_76": {
        "path": r"G:\......................AI_Models\PER_CATEGORY_MODELS\CatBoost_indices_acc76.pkl",
        "type": "pickle",
        "scaler": None,
        "accuracy": 0.76,
        "notes": "Indices specialist",
    },
    "catboost_crypto_56": {
        "path": r"G:\......................AI_Models\PER_CATEGORY_MODELS\CatBoost_crypto_acc56.pkl",
        "type": "pickle",
        "scaler": None,
        "accuracy": 0.56,
        "notes": "Crypto — needs improvement",
    },

    # === 80+ Suite (use with caution — may be overfitted) ===
    "catboost_80plus": {
        "path": r"G:\......................AI_Models\Modele_AI_80_Plus\CatBoost_model.pkl",
        "type": "pickle",
        "scaler": r"G:\......................AI_Models\Modele_AI_80_Plus\CatBoost_scaler.pkl",
        "accuracy": 1.0,
        "notes": "⚠️ 100% accuracy — likely overfitted, needs walk-forward validation",
    },
    "xgboost_80plus": {
        "path": r"G:\......................AI_Models\Modele_AI_80_Plus\XGBoost_model.pkl",
        "type": "pickle",
        "scaler": r"G:\......................AI_Models\Modele_AI_80_Plus\XGBoost_scaler.pkl",
        "accuracy": 0.9996,
        "notes": "⚠️ 99.96% — likely overfitted",
    },
    "extratrees_80plus": {
        "path": r"G:\......................AI_Models\Modele_AI_80_Plus\ExtraTrees_model.pkl",
        "type": "pickle",
        "scaler": r"G:\......................AI_Models\Modele_AI_80_Plus\ExtraTrees_scaler.pkl",
        "accuracy": 0.9842,
        "notes": "98.4% — needs validation",
    },
}

# Features used by all 80_Plus models (60 features)
FEATURES_60 = [
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
    "hour", "day_of_week",
    "is_london_session", "is_ny_session", "is_asia_session",
    "source_file", "Adj Close",
    "price_change", "price_change_pct", "price_range",
    "rsi", "macd", "signal",
    "ma_5", "ma_10", "ma_20", "volume_ma",
]


class MLSignalEngine:
    """
    ML Signal Engine Agent — Multi-model trading signal generator.

    Can load any model from the model registry. Supports ensemble voting
    across multiple models for more robust signals.

    Usage:
        engine = MLSignalEngine()
        engine.load_model("catboost_clean_75")
        signal = engine.predict("AAPL", {"features": feature_array})

        # Or load multiple models for ensemble
        engine.load_model("catboost_clean_75")
        engine.load_model("catboost_indices_76")
        signal = engine.ensemble_predict("AAPL", {"features": feature_array})
    """

    def __init__(self, default_model: str = "catboost_v2"):
        self._models = {}        # model_name -> loaded model
        self._scalers = {}       # model_name -> loaded scaler
        self._default_model = default_model
        self._initialized = False

    def load_model(self, model_name: str) -> bool:
        """Load a model from the registry."""
        if model_name not in MODEL_REGISTRY:
            logger.error("Model '%s' not in registry. Available: %s",
                         model_name, list(MODEL_REGISTRY.keys()))
            return False

        entry = MODEL_REGISTRY[model_name]
        model_path = Path(entry["path"])

        if not model_path.exists():
            logger.error("Model file not found: %s", model_path)
            return False

        # SHA-256 hash verification before loading
        if not _verify_model_hash(model_name, model_path):
            logger.error("Refused to load model '%s' — hash verification failed", model_name)
            return False

        try:
            if entry["type"] == "catboost_native":
                from catboost import CatBoostClassifier
                model = CatBoostClassifier()
                model.load_model(str(model_path))
            elif entry["type"] == "pickle":
                if not _is_trusted_path(model_path):
                    logger.error("Refused to load model from untrusted path: %s", model_path)
                    return False
                obj = joblib.load(model_path)
                # Handle dict-wrapped models (e.g., {'model': CatBoostClassifier, 'features': [...]})
                if isinstance(obj, dict) and 'model' in obj:
                    model = obj['model']
                    logger.info("  Unwrapped dict model: %s (features: %d)",
                                 type(model).__name__,
                                 len(obj.get('features', [])))
                else:
                    model = obj
            else:
                logger.error("Unknown model type: %s", entry["type"])
                return False

            self._models[model_name] = model

            # Load scaler if available
            if entry.get("scaler"):
                scaler_path = Path(entry["scaler"])
                if scaler_path.exists():
                    if not _is_trusted_path(scaler_path):
                        logger.error("Refused to load scaler from untrusted path: %s", scaler_path)
                    else:
                        self._scalers[model_name] = joblib.load(scaler_path)
                        logger.info("Loaded scaler for %s", model_name)

            self._initialized = True
            logger.info("✅ Model loaded: %s (accuracy: %s) — %s",
                         model_name, entry["accuracy"], entry["notes"])
            return True

        except Exception:
            logger.exception("Failed to load model %s", model_name)
            return False

    def list_models(self) -> Dict[str, Dict]:
        """List all available models in the registry."""
        return {
            name: {
                "accuracy": entry["accuracy"],
                "notes": entry["notes"],
                "loaded": name in self._models,
                "path_exists": Path(entry["path"]).exists(),
            }
            for name, entry in MODEL_REGISTRY.items()
        }

    def predict(self, symbol: str, context: Dict = None,
                model_name: str = None) -> Dict[str, Any]:
        """
        Generate trading signal using a specific model.

        Args:
            symbol: Trading symbol
            context: Must contain 'features' key with feature array
            model_name: Which model to use (default: first loaded)

        Returns:
            Dict with recommendation, confidence, reasoning
        """
        context = context or {}

        # Select model
        if model_name and model_name in self._models:
            model = self._models[model_name]
            scaler = self._scalers.get(model_name)
        elif self._models:
            model_name = list(self._models.keys())[0]
            model = self._models[model_name]
            scaler = self._scalers.get(model_name)
        else:
            # Try to load default model
            if not self.load_model(self._default_model):
                return {
                    "recommendation": "HOLD",
                    "confidence": 0.0,
                    "reasoning": f"No model loaded for {symbol}",
                    "metadata": {"mode": "no_model"}
                }
            model = self._models[self._default_model]
            model_name = self._default_model
            scaler = self._scalers.get(self._default_model)

        if "features" not in context:
            return {
                "recommendation": "HOLD",
                "confidence": 0.0,
                "reasoning": f"No features provided for {symbol}",
                "metadata": {"mode": "no_features", "model": model_name}
            }

        try:
            import numpy as np
            import pandas as pd
            features_raw = np.array(context["features"]).reshape(1, -1)

            # CatBoost native models need DataFrame with column names
            if hasattr(model, 'feature_names_') and model.feature_names_:
                feature_names = model.feature_names_
                # Trim or pad features to match model expectations
                n_expected = len(feature_names)
                if features_raw.shape[1] >= n_expected:
                    features_raw = features_raw[:, :n_expected]
                else:
                    # Pad with zeros
                    pad = np.zeros((1, n_expected - features_raw.shape[1]))
                    features_raw = np.concatenate([features_raw, pad], axis=1)
                features = pd.DataFrame(features_raw, columns=feature_names)
            else:
                features = features_raw

            # Apply scaler if available
            if scaler is not None:
                features = scaler.transform(features)

            # Predict
            prediction = model.predict(features)
            if hasattr(prediction, 'flatten'):
                prediction = prediction.flatten()[0]

            # Get probabilities if available
            confidence = 0.5
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(features)
                if hasattr(probas, 'flatten'):
                    probabilities = probas.flatten().tolist()
                confidence = float(max(probas.flatten()))

            # Map prediction to action
            pred_int = int(prediction)
            action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            # Binary classification (0/1)
            if pred_int in [0, 1] and len(action_map) > 2:
                action_map = {0: "SELL", 1: "BUY"}
            action = action_map.get(pred_int, "HOLD")

            return {
                "recommendation": action,
                "confidence": confidence,
                "reasoning": (f"{model_name}: {action} for {symbol} "
                              f"(confidence: {confidence:.2%})"),
                "metadata": {
                    "model": model_name,
                    "accuracy": MODEL_REGISTRY[model_name]["accuracy"],
                    "probabilities": probabilities,
                    "prediction_raw": pred_int,
                }
            }

        except Exception as e:
            logger.exception("Prediction failed (%s, %s)", model_name, symbol)
            return {
                "recommendation": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Prediction error: {str(e)}",
                "metadata": {"model": model_name, "error": str(e)}
            }

    def ensemble_predict(self, symbol: str, context: Dict = None) -> Dict[str, Any]:
        """
        Ensemble prediction across all loaded models.
        Uses majority voting weighted by model accuracy.
        """
        if not self._models:
            return self.predict(symbol, context)

        votes = []
        for model_name in self._models:
            result = self.predict(symbol, context, model_name=model_name)
            accuracy = MODEL_REGISTRY.get(model_name, {}).get("accuracy", 0.5)
            if isinstance(accuracy, str):
                accuracy = 0.5
            votes.append({
                "model": model_name,
                "recommendation": result["recommendation"],
                "confidence": result["confidence"],
                "weight": accuracy,
            })

        # Weighted voting
        direction_scores = {"BUY": 0, "SELL": 0, "HOLD": 0}
        total_weight = 0
        for vote in votes:
            direction_scores[vote["recommendation"]] += vote["weight"] * vote["confidence"]
            total_weight += vote["weight"]

        if total_weight > 0:
            for k in direction_scores:
                direction_scores[k] /= total_weight

        best_action = max(direction_scores, key=direction_scores.get)
        best_confidence = direction_scores[best_action]

        return {
            "recommendation": best_action,
            "confidence": best_confidence,
            "reasoning": f"Ensemble ({len(votes)} models): {best_action} "
                         f"(confidence: {best_confidence:.2%})",
            "metadata": {
                "votes": votes,
                "direction_scores": direction_scores,
                "models_used": len(votes),
            }
        }

    # Brain-compatible interface
    def analyze(self, symbol: str, context: Dict = None) -> Dict:
        """Brain-compatible: uses ensemble if multiple models loaded."""
        if len(self._models) > 1:
            return self.ensemble_predict(symbol, context)
        return self.predict(symbol, context)

    def cleanup(self):
        """Release memory."""
        self._models.clear()
        self._scalers.clear()
        self._initialized = False
        gc.collect()
        logger.info("ML Signal Engine cleanup complete")


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    if "--generate-hashes" in sys.argv:
        print("\n" + "=" * 60)
        print("ML SIGNAL ENGINE — Generate Model Hashes (SHA-256)")
        print("=" * 60 + "\n")
        generate_model_hashes()
        sys.exit(0)

    print("\n" + "=" * 60)
    print("ML SIGNAL ENGINE — Model Registry Test")
    print("=" * 60)

    engine = MLSignalEngine()

    # List all models
    print("\nModel Registry:")
    for name, info in engine.list_models().items():
        status = "EXISTS" if info["path_exists"] else "MISSING"
        loaded = " [LOADED]" if info["loaded"] else ""
        print(f"  {name}: acc={info['accuracy']} -- {status}{loaded}")
        print(f"    {info['notes']}")

    # Try to load catboost_v2
    print("\n" + "-" * 60)
    print("Loading catboost_v2...")
    if engine.load_model("catboost_v2"):
        print("Model loaded successfully")
    else:
        print("Could not load -- trying catboost_clean_75...")
        engine.load_model("catboost_clean_75")

    print("\nTest complete")
    engine.cleanup()
