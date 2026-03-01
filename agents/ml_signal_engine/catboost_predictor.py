"""
SCRIPT NAME: catboost_predictor.py
====================================
Execution Location: K:\_DEV_MVP_2026\Market_Hawk_3\agents\ml_signal_engine\
Purpose: ML Signal Engine — CatBoost Model Wrapper (76.47% accuracy)
Creation Date: 2026-03-01
"""

import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger("market_hawk.ml_signal")


class MLSignalEngine:
    """
    ML Signal Engine Agent — CatBoost-based trading signal generator.
    Wraps the walk-forward validated CatBoost model (76.47% accuracy).
    """

    def __init__(self, config=None):
        from config.settings import ML_CONFIG
        self.config = config or ML_CONFIG
        self._model = None
        self._initialized = False

    def initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                logger.warning("Model not found: %s — demo mode", model_path)
                self._initialized = True
                return True

            from catboost import CatBoostClassifier
            self._model = CatBoostClassifier()
            self._model.load_model(str(model_path))
            self._initialized = True
            logger.info("ML Signal Engine loaded from %s", model_path)
            return True

        except ImportError:
            logger.warning("CatBoost not installed — demo mode")
            self._initialized = True
            return True
        except Exception as e:
            logger.error("Failed to load ML model: %s", str(e))
            return False

    def predict(self, symbol: str, context: Dict = None) -> Dict[str, Any]:
        """Generate trading signal for a symbol."""
        if not self._initialized:
            self.initialize()

        context = context or {}

        if self._model is not None and "features" in context:
            try:
                import numpy as np
                features = np.array(context["features"]).reshape(1, -1)
                prediction = self._model.predict(features)[0]
                probabilities = self._model.predict_proba(features)[0]
                confidence = float(max(probabilities))

                action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
                action = action_map.get(int(prediction), "HOLD")

                return {
                    "recommendation": action,
                    "confidence": confidence,
                    "reasoning": f"CatBoost: {action} with {confidence:.2%} confidence for {symbol}",
                    "metadata": {"probabilities": probabilities.tolist(),
                                 "model": "catboost_v2",
                                 "walk_forward_accuracy": 0.7647}
                }
            except Exception as e:
                logger.error("Prediction failed for %s: %s", symbol, str(e))

        return {
            "recommendation": "HOLD",
            "confidence": 0.0,
            "reasoning": f"No features available for {symbol} (demo mode)",
            "metadata": {"mode": "demo"}
        }

    def analyze(self, symbol: str, context: Dict = None) -> Dict:
        """Brain-compatible alias."""
        return self.predict(symbol, context)
