"""
ML Classifier Module
=====================
Load model LightGBM + TF-IDF untuk klasifikasi tiket.
Uses domain-aware preprocessing for consistency.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

# Optional LightGBM import
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

from src.ml.preprocessing import preprocess_text

if TYPE_CHECKING:
    from src.core.config import Config

_LOGGER = logging.getLogger(__name__)

# Confidence thresholds
THRESHOLD_AUTO = 0.80        # >= 80% = AUTO (langsung pakai, isi Symtomps)
THRESHOLD_HIGH = 0.70        # 70-80% = HIGH_REVIEW
THRESHOLD_MEDIUM = 0.50      # 50-70% = MEDIUM_REVIEW
                             # < 50% = MANUAL


@dataclass
class PredictionResult:
    """Hasil prediksi ML classifier."""
    predicted_symtomps: str
    ml_confidence: float
    prediction_status: str  # AUTO, HIGH_REVIEW, MEDIUM_REVIEW, MANUAL
    inference_time_ms: float


class MLClassifier:
    """
    Wrapper untuk ML model classification.
    
    Uses consistent preprocessing with training pipeline.
    """
    
    def __init__(self, config: "Config") -> None:
        """
        Initialize classifier dengan load model artifacts.
        
        Args:
            config: Configuration object containing model_dir and model_version
        """
        self._model_dir = config.model_dir
        self._version = self._resolve_version(config.model_version, config.model_dir)
        self._model = None
        self._tfidf = None
        self._word_tfidf = None
        self._char_tfidf = None
        self._has_char_tfidf = False
        self._label_encoder = None
        self._is_loaded = False
        self._model_version = "unknown"
        self._metadata = {}
        
        self._load_model()
    
    def _resolve_version(self, version: str, model_dir: Path) -> str:
        """
        Resolve version string to actual version.
        
        If version is "auto", read from current_version.txt.
        Otherwise return as-is.
        """
        if version.lower() == "auto":
            current_version_file = model_dir / "current_version.txt"
            if current_version_file.exists():
                resolved = current_version_file.read_text().strip()
                _LOGGER.info(f"Auto-detected model version: {resolved}")
                return resolved
            else:
                _LOGGER.warning("current_version.txt not found, falling back to v1")
                return "v1"
        return version
    
    def _load_model(self) -> None:
        """
        Load semua model artifacts dari disk.
        
        Supports two formats:
        1. New: Versioned folders (models/v3/lgb_model.txt)
        2. Legacy: Flat files (models/lgb_model_v2.txt)
        """
        if not HAS_LIGHTGBM:
            _LOGGER.warning("LightGBM not installed, classifier disabled")
            return
        
        try:
            # Determine model path - check versioned folder first
            version = self._version
            version_dir = self._model_dir / version
            
            if version_dir.exists() and version_dir.is_dir():
                # New format: versioned folder
                _LOGGER.info(f"Loading model from versioned folder: {version_dir}")
                # Try .bin first, fallback to .txt
                model_path = version_dir / "lgb_model.bin"
                if not model_path.exists():
                    model_path = version_dir / "lgb_model.txt"
                tfidf_path = version_dir / "tfidf_vectorizer.pkl"
                encoder_path = version_dir / "label_encoder.pkl"
                metadata_path = version_dir / "metadata.json"
            else:
                # Legacy format: flat files with version suffix
                _LOGGER.info(f"Loading model from legacy format: {self._model_dir}")
                model_path = self._model_dir / f"lgb_model_{version}.txt"
                tfidf_path = self._model_dir / f"tfidf_vectorizer_{version}.pkl"
                encoder_path = self._model_dir / f"label_encoder_{version}.pkl"
                metadata_path = self._model_dir / f"model_metadata_{version}.json"
            
            # Load LightGBM model
            if not model_path.exists():
                _LOGGER.error("Model file not found: %s", model_path)
                return
            self._model = lgb.Booster(model_file=str(model_path))
            
            # Load TF-IDF vectorizer
            if not tfidf_path.exists():
                _LOGGER.error("TF-IDF file not found: %s", tfidf_path)
                return
            with open(tfidf_path, 'rb') as f:
                tfidf_data = pickle.load(f)
            
            # Support both old (single TF-IDF) and new (word+char combined) formats
            if isinstance(tfidf_data, dict) and 'word_tfidf' in tfidf_data:
                # New format: word + char TF-IDF
                self._word_tfidf = tfidf_data['word_tfidf']
                self._char_tfidf = tfidf_data['char_tfidf']
                self._has_char_tfidf = True
                self._tfidf = None  # Not used in new format
                _LOGGER.info("Loaded combined word+char TF-IDF vectorizers")
            else:
                # Old format: single TF-IDF
                self._tfidf = tfidf_data
                self._word_tfidf = None
                self._char_tfidf = None
                self._has_char_tfidf = False
            
            # Load label encoder
            if not encoder_path.exists():
                _LOGGER.error("Label encoder file not found: %s", encoder_path)
                return
            with open(encoder_path, 'rb') as f:
                self._label_encoder = pickle.load(f)
            
            # Load metadata untuk version info
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self._metadata = json.load(f)
                    self._model_version = self._metadata.get('version', version)
            
            self._is_loaded = True
            
            # Check label encoder type (dict or sklearn LabelEncoder)
            if isinstance(self._label_encoder, dict):
                num_classes = len(self._label_encoder)
            else:
                num_classes = len(self._label_encoder.classes_)
            
            _LOGGER.info(
                "ML Classifier loaded successfully. Version: %s, Classes: %d",
                self._model_version,
                num_classes
            )
            
        except Exception as e:
            _LOGGER.exception("Failed to load ML model: %s", e)
            self._is_loaded = False
    
    def reload(self, new_version: Optional[str] = None) -> bool:
        """
        Hot reload model dari disk tanpa restart.
        
        Args:
            new_version: Optional new version to load (e.g., 'v3'). 
                        If None, reload current version.
        
        Returns:
            True if reload successful, False otherwise
        """
        old_version = self._model_version
        
        if new_version:
            self._version = new_version
        
        _LOGGER.info("Reloading ML model (current: %s, target: %s)...", 
                     old_version, self._version)
        
        # Reset state
        self._is_loaded = False
        self._model = None
        self._tfidf = None
        self._word_tfidf = None
        self._char_tfidf = None
        self._has_char_tfidf = False
        self._label_encoder = None
        
        # Reload
        self._load_model()
        
        if self._is_loaded:
            _LOGGER.info("Model reloaded successfully: %s → %s", 
                        old_version, self._model_version)
            return True
        else:
            _LOGGER.error("Model reload failed!")
            return False
    
    def _get_prediction_status(self, confidence: float) -> str:
        """Determine prediction status based on confidence threshold."""
        if confidence >= THRESHOLD_AUTO:
            return "AUTO"
        elif confidence >= THRESHOLD_HIGH:
            return "HIGH_REVIEW"
        elif confidence >= THRESHOLD_MEDIUM:
            return "MEDIUM_REVIEW"
        else:
            return "MANUAL"
    
    def predict(self, tech_raw_text: str, solving: str = "") -> PredictionResult:
        """
        Predict symptom category dari tech message dan solving text.
        
        Args:
            tech_raw_text: Raw text dari teknisi
            solving: Text solving dari ops (optional, untuk better accuracy)
        
        Returns:
            PredictionResult dengan predicted label, confidence, dan status
        """
        start_time = time.time()
        
        # Default result jika model belum loaded
        if not self._is_loaded:
            _LOGGER.warning("ML model not loaded, returning empty prediction")
            return PredictionResult(
                predicted_symtomps="",
                ml_confidence=0.0,
                prediction_status="MANUAL",
                inference_time_ms=0.0
            )
        
        try:
            # Preprocess text using domain-aware preprocessor
            cleaned_text = preprocess_text(tech_raw_text, solving)
            
            if not cleaned_text:
                return PredictionResult(
                    predicted_symtomps="",
                    ml_confidence=0.0,
                    prediction_status="MANUAL",
                    inference_time_ms=(time.time() - start_time) * 1000
                )
            
            # Vectorize - support both old and new TF-IDF formats
            if self._has_char_tfidf:
                # New format: word + char TF-IDF
                from scipy import sparse
                X_word = self._word_tfidf.transform([cleaned_text])
                X_char = self._char_tfidf.transform([cleaned_text])
                X = sparse.hstack([X_word, X_char])
            else:
                # Old format: single TF-IDF
                X = self._tfidf.transform([cleaned_text])
            
            # Predict
            probas = self._model.predict(X)[0]
            predicted_idx = np.argmax(probas)
            confidence = probas[predicted_idx]
            
            # Decode label - support both dict and sklearn LabelEncoder
            if isinstance(self._label_encoder, dict):
                predicted_label = self._label_encoder.get(predicted_idx, "UNKNOWN")
            else:
                predicted_label = self._label_encoder.inverse_transform([predicted_idx])[0]
            
            # Get status
            status = self._get_prediction_status(confidence)
            
            inference_time = (time.time() - start_time) * 1000
            
            _LOGGER.debug(
                "Prediction: %s (%.2f%%) - %s [%.1fms]",
                predicted_label, confidence * 100, status, inference_time
            )
            
            return PredictionResult(
                predicted_symtomps=predicted_label,
                ml_confidence=round(confidence, 4),
                prediction_status=status,
                inference_time_ms=round(inference_time, 2)
            )
            
        except Exception as e:
            _LOGGER.exception("Prediction failed: %s", e)
            return PredictionResult(
                predicted_symtomps="",
                ml_confidence=0.0,
                prediction_status="MANUAL",
                inference_time_ms=(time.time() - start_time) * 1000
            )
    
    @property
    def is_loaded(self) -> bool:
        """Check apakah model sudah loaded."""
        return self._is_loaded
    
    @property
    def model_version(self) -> str:
        """Get model version."""
        return self._model_version
    
    @property
    def num_classes(self) -> int:
        """Get jumlah classes."""
        if self._label_encoder:
            if isinstance(self._label_encoder, dict):
                return len(self._label_encoder)
            return len(self._label_encoder.classes_)
        return 0
    
    def get_model_info(self) -> dict:
        """Get info model untuk reporting."""
        return {
            "version": self._model_version,
            "is_loaded": self._is_loaded,
            "num_classes": self.num_classes,
            "thresholds": {
                "AUTO": THRESHOLD_AUTO,
                "HIGH_REVIEW": THRESHOLD_HIGH,
                "MEDIUM_REVIEW": THRESHOLD_MEDIUM,
            }
        }
    
    def get_metadata(self) -> dict:
        """Get full model metadata for admin commands."""
        # Get classes list
        if self._label_encoder:
            if isinstance(self._label_encoder, dict):
                classes = list(self._label_encoder.values())
            else:
                classes = list(self._label_encoder.classes_)
        else:
            classes = []
        
        metadata = {
            "version": self._model_version,
            "is_loaded": self._is_loaded,
            "num_classes": self.num_classes,
            "threshold_auto": THRESHOLD_AUTO,
            "threshold_high": THRESHOLD_HIGH,
            "threshold_medium": THRESHOLD_MEDIUM,
            "classes": classes,
        }
        
        # Merge with file metadata
        if self._metadata:
            metadata.update({
                "training_samples": self._metadata.get("training_samples", "N/A"),
                "training_accuracy": self._metadata.get("training_accuracy", 0) * 100,
                "trained_at": self._metadata.get("trained_at", "N/A"),
            })
        
        return metadata
