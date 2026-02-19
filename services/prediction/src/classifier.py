"""
LightGBM Classifier (adapted for microservice)
================================================
Loads model from MLflow, performs TF-IDF + LightGBM prediction.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

from services.prediction.src.mlflow_utils import MLflowManager
from services.shared.preprocessing import preprocess_text

_LOGGER = logging.getLogger(__name__)


class LightGBMClassifier:
    """
    LightGBM + TF-IDF classifier loaded from MLflow Registry.
    """

    def __init__(self, mlflow_mgr: MLflowManager) -> None:
        self._mlflow_mgr = mlflow_mgr
        self._model = None
        self._tfidf = None
        self._word_tfidf = None
        self._char_tfidf = None
        self._has_char_tfidf = False
        self._label_encoder = None
        self._preprocessor = None
        self._is_loaded = False
        self._model_version = "unknown"
        self._model_stage = "unknown"
        self._model_run_id = None
        self._metadata = {}

    def load(self, stage: str = "Production") -> bool:
        """Load model from MLflow Registry."""
        try:
            if not self._mlflow_mgr.init():
                _LOGGER.error("MLflow initialization failed")
                return False

            model_data = self._mlflow_mgr.load_model_by_stage(stage)
            if model_data is None:
                _LOGGER.error("No model found in MLflow stage: %s", stage)
                return False

            self._model = model_data["model"]

            vectorizers = model_data["vectorizers"]
            if isinstance(vectorizers, dict) and "word_tfidf" in vectorizers:
                self._word_tfidf = vectorizers["word_tfidf"]
                self._char_tfidf = vectorizers["char_tfidf"]
                self._has_char_tfidf = True
                self._tfidf = None
            else:
                self._tfidf = vectorizers
                self._has_char_tfidf = False

            self._label_encoder = model_data["encoder"]
            self._preprocessor = model_data.get("preprocessor")
            self._metadata = model_data["metadata"]
            self._model_version = f"mlflow-v{model_data['version']}"
            self._model_stage = str(model_data.get("stage") or "unknown")
            self._model_run_id = model_data.get("run_id")
            self._is_loaded = True

            _LOGGER.info(
                "Loaded model: version=%s, stage=%s, classes=%d",
                model_data["version"], model_data["stage"], self.num_classes
            )
            return True

        except Exception as e:
            _LOGGER.error("Failed to load from MLflow: %s", e)
            return False

    def reload(self, stage: str = "Production") -> tuple[bool, str, str]:
        """
        Hot reload model. Returns (success, old_version, new_version).
        """
        old_version = self._model_version

        # Reset
        self._is_loaded = False
        self._model = None
        self._tfidf = None
        self._word_tfidf = None
        self._char_tfidf = None
        self._has_char_tfidf = False
        self._label_encoder = None
        self._model_stage = "unknown"
        self._model_run_id = None

        success = self.load(stage)
        return success, old_version, self._model_version

    def predict(self, tech_raw_text: str, solving: str = "") -> dict:
        """
        Predict using LightGBM.

        Returns dict: label, confidence, inference_time_ms, all_probas
        """
        start_time = time.time()

        if not self._is_loaded:
            return {
                "label": "",
                "confidence": 0.0,
                "inference_time_ms": 0.0,
                "all_probas": {},
            }

        try:
            cleaned_text = preprocess_text(tech_raw_text, solving)
            if not cleaned_text:
                return {
                    "label": "",
                    "confidence": 0.0,
                    "inference_time_ms": (time.time() - start_time) * 1000,
                    "all_probas": {},
                }

            # Vectorize
            if self._has_char_tfidf:
                from scipy import sparse
                X_word = self._word_tfidf.transform([cleaned_text])
                X_char = self._char_tfidf.transform([cleaned_text])
                X = sparse.hstack([X_word, X_char])
            else:
                X = self._tfidf.transform([cleaned_text])

            # Predict — use predict_proba for probability matrix
            if hasattr(self._model, 'predict_proba'):
                probas = self._model.predict_proba(X)[0]
            else:
                probas = self._model.predict(X)[0]
            predicted_idx = np.argmax(probas)
            confidence = probas[predicted_idx]

            # Decode label
            if isinstance(self._label_encoder, dict):
                predicted_label = self._label_encoder.get(predicted_idx, "UNKNOWN")
                all_probas = {self._label_encoder.get(i, f"class_{i}"): float(p)
                              for i, p in enumerate(probas)}
            else:
                predicted_label = self._label_encoder.inverse_transform([predicted_idx])[0]
                all_probas = {cls: float(p)
                              for cls, p in zip(self._label_encoder.classes_, probas)}

            inference_time = (time.time() - start_time) * 1000

            return {
                "label": predicted_label,
                "confidence": round(float(confidence), 4),
                "inference_time_ms": round(inference_time, 2),
                "all_probas": all_probas,
            }

        except Exception as e:
            _LOGGER.exception("Prediction failed: %s", e)
            return {
                "label": "",
                "confidence": 0.0,
                "inference_time_ms": (time.time() - start_time) * 1000,
                "all_probas": {},
            }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_stage(self) -> str:
        return self._model_stage

    @property
    def model_run_id(self) -> str | None:
        return self._model_run_id

    @property
    def num_classes(self) -> int:
        if self._label_encoder:
            if isinstance(self._label_encoder, dict):
                return len(self._label_encoder)
            return len(self._label_encoder.classes_)
        return 0

    @property
    def classes(self) -> list[str]:
        if self._label_encoder:
            if isinstance(self._label_encoder, dict):
                return list(self._label_encoder.values())
            return list(self._label_encoder.classes_)
        return []

    @property
    def metadata(self) -> dict:
        return self._metadata
