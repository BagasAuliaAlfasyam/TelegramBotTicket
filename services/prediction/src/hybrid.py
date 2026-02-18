"""
Hybrid Cascade Classifier
===========================
LightGBM primary → Gemini fallback when confidence is low.

Knowledge Distillation pattern:
- Gemini = "Teacher" (slow, expensive, smart)
- LightGBM = "Student" (fast, free, learns from teacher)
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from services.prediction.src.classifier import LightGBMClassifier
from services.prediction.src.gemini_classifier import GeminiClassifier
from services.prediction.src.mlflow_utils import MLflowConfig, MLflowManager
from services.shared.config import PredictionServiceConfig
from services.shared.models import (
    PredictionResult,
    PredictionSource,
    PredictionStatus,
)

_LOGGER = logging.getLogger(__name__)


class HybridClassifier:
    """
    Cascade classifier: LightGBM → Gemini fallback.

    Flow:
    1. LightGBM predicts (fast, free)
    2. If confidence >= cascade_threshold: use LightGBM result
    3. If confidence < cascade_threshold AND Gemini enabled:
       a. Ask Gemini for classification
       b. If Gemini confident: use Gemini result, mark source="gemini"
       c. If Gemini fails: fall back to LightGBM result
    4. Results logged with source tag for retraining analytics
    """

    def __init__(self, config: PredictionServiceConfig) -> None:
        self._config = config
        self._cascade_threshold = config.gemini_cascade_threshold

        # Initialize LightGBM
        mlflow_config = MLflowConfig(
            tracking_uri=config.mlflow_tracking_uri,
            experiment_name=config.mlflow_experiment_name,
            model_name=config.mlflow_model_name,
            s3_endpoint_url=config.mlflow_s3_endpoint_url,
            bucket_name=config.mlflow_bucket_name,
            tracking_username=config.mlflow_tracking_username,
            tracking_password=config.mlflow_tracking_password,
        )
        self._mlflow_mgr = MLflowManager(mlflow_config)
        self._lgbm = LightGBMClassifier(self._mlflow_mgr)

        # Initialize Gemini (optional)
        self._gemini: GeminiClassifier | None = None
        if config.gemini_enabled and config.gemini_api_key:
            self._gemini = GeminiClassifier(
                api_key=config.gemini_api_key,
                model_name=config.gemini_model_name,
                timeout=config.gemini_timeout,
            )

        # Threshold (2-tier)
        self._threshold_auto = config.threshold_auto

    def load_model(self, stage: str = "Production") -> bool:
        """Load LightGBM model and sync labels to Gemini."""
        success = self._lgbm.load(stage)

        if success and self._gemini:
            # Sync label list to Gemini
            self._gemini.set_labels(self._lgbm.classes)
            _LOGGER.info("Synced %d labels to Gemini", len(self._lgbm.classes))

        return success

    def reload_model(self, stage: str = "Production") -> tuple[bool, str, str]:
        """Hot reload LightGBM model and resync Gemini labels."""
        success, old_ver, new_ver = self._lgbm.reload(stage)

        if success and self._gemini:
            self._gemini.set_labels(self._lgbm.classes)

        return success, old_ver, new_ver

    def predict(self, tech_raw_text: str, solving: str = "") -> PredictionResult:
        """
        Cascade prediction: LightGBM → Gemini fallback.
        """
        total_start = time.time()

        # Step 1: LightGBM prediction (always first)
        lgbm_result = self._lgbm.predict(tech_raw_text, solving)
        lgbm_confidence = lgbm_result["confidence"]
        lgbm_label = lgbm_result["label"]

        # Step 2: Decide if we need Gemini
        gemini_label = None
        gemini_confidence = None
        final_label = lgbm_label
        final_confidence = lgbm_confidence
        source = PredictionSource.LIGHTGBM

        if (lgbm_confidence < self._cascade_threshold
            and self._gemini is not None
            and self._gemini.is_ready
            and lgbm_label):  # Only cascade if LightGBM gave some prediction

            _LOGGER.info(
                "LightGBM confidence %.1f%% < threshold %.1f%%, asking Gemini...",
                lgbm_confidence * 100,
                self._cascade_threshold * 100,
            )

            gemini_result = self._gemini.predict(tech_raw_text, solving)

            if gemini_result and gemini_result["confidence"] > 0.5:
                gemini_label = gemini_result["label"]
                gemini_confidence = gemini_result["confidence"]

                # Strategy: Use Gemini if it's more confident
                if gemini_confidence > lgbm_confidence:
                    final_label = gemini_label
                    final_confidence = gemini_confidence
                    source = PredictionSource.GEMINI
                    _LOGGER.info(
                        "Using Gemini: %s (%.1f%%) > LightGBM: %s (%.1f%%)",
                        gemini_label, gemini_confidence * 100,
                        lgbm_label, lgbm_confidence * 100
                    )
                else:
                    # Both agree or LightGBM is still better
                    if gemini_label == lgbm_label:
                        # Both agree! Boost confidence (never lower than original)
                        boosted = (lgbm_confidence + gemini_confidence) / 2 * 1.1
                        final_confidence = min(max(lgbm_confidence, boosted), 0.99)
                        source = PredictionSource.HYBRID
                        _LOGGER.info(
                            "Both agree on %s! Boosted confidence: %.1f%%",
                            final_label, final_confidence * 100
                        )
                    else:
                        _LOGGER.info(
                            "Disagreement: LightGBM=%s, Gemini=%s. Keeping LightGBM.",
                            lgbm_label, gemini_label
                        )

        # Step 3: Determine status
        status = self._get_status(final_confidence)

        total_time = (time.time() - total_start) * 1000

        return PredictionResult(
            predicted_symtomps=final_label,
            ml_confidence=round(final_confidence, 4),
            prediction_status=status,
            inference_time_ms=round(total_time, 2),
            source=source,
            gemini_label=gemini_label,
            gemini_confidence=round(gemini_confidence, 4) if gemini_confidence else None,
        )

    def _get_status(self, confidence: float) -> PredictionStatus:
        if confidence >= self._threshold_auto:
            return PredictionStatus.AUTO
        return PredictionStatus.REVIEW

    @property
    def is_loaded(self) -> bool:
        return self._lgbm.is_loaded

    def get_info(self) -> dict:
        """Get combined model info."""
        info = {
            "version": self._lgbm.model_version,
            "is_loaded": self._lgbm.is_loaded,
            "num_classes": self._lgbm.num_classes,
            "classes": self._lgbm.classes,
            "thresholds": {
                "AUTO": self._threshold_auto,
                "REVIEW": "< " + str(self._threshold_auto),
            },
            "cascade_threshold": self._cascade_threshold,
            "gemini_enabled": self._gemini is not None and self._gemini.is_ready,
            "training_samples": self._lgbm.metadata.get("n_samples"),
            "training_accuracy": self._lgbm.metadata.get("f1_macro"),
            "trained_at": self._lgbm.metadata.get("trained_at"),
        }

        if self._gemini:
            info["gemini"] = self._gemini.get_info()

        # MLflow status
        info["mlflow"] = self._mlflow_mgr.get_status()

        return info
