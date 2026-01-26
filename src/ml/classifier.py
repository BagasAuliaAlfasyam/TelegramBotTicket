"""
ML Classifier Module
=====================
Modul untuk klasifikasi tiket menggunakan model LightGBM + TF-IDF.

File ini bertanggung jawab untuk:
- Load model ML dari disk (LightGBM booster + TF-IDF vectorizer)
- Preprocessing text menggunakan domain-aware preprocessor
- Prediksi kategori Symtomps dari text tiket
- Hot reload model tanpa restart bot

Model Flow:
    1. Terima text dari teknisi + solving dari ops
    2. Preprocess: lowercase, normalisasi abbreviation, merge text
    3. Vectorize: TF-IDF word (1-3 ngram) + char (3-5 ngram)
    4. Predict: LightGBM booster predict probabilities
    5. Return: label dengan confidence tertinggi + status

Confidence Thresholds:
    - AUTO (>= 80%): Langsung pakai, isi Symtomps otomatis
    - HIGH_REVIEW (70-80%): Perlu review tapi kemungkinan benar
    - MEDIUM_REVIEW (50-70%): Review prioritas medium
    - MANUAL (< 50%): Harus review manual, prediksi tidak reliable

Model Versioning:
    - Support versioned folders: models/v1/, models/v2/
    - Auto-detect versi dari current_version.txt
    - Hot reload via reload() method
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
    """
    Container untuk hasil prediksi ML classifier.
    
    Digunakan untuk mengembalikan hasil prediksi beserta metadata
    yang berguna untuk logging, review, dan monitoring.
    
    Attributes:
        predicted_symtomps: Label kategori yang diprediksi (contoh: "KENDALA ONT/NTE")
        ml_confidence: Confidence score 0.0 - 1.0 (contoh: 0.85 = 85%)
        prediction_status: Status berdasarkan threshold (AUTO/HIGH_REVIEW/MEDIUM_REVIEW/MANUAL)
        inference_time_ms: Waktu prediksi dalam milliseconds
    
    Example:
        PredictionResult(
            predicted_symtomps="KENDALA ONT/NTE",
            ml_confidence=0.8523,
            prediction_status="AUTO",
            inference_time_ms=15.3
        )
    """
    predicted_symtomps: str
    ml_confidence: float
    prediction_status: str  # AUTO, HIGH_REVIEW, MEDIUM_REVIEW, MANUAL
    inference_time_ms: float


class MLClassifier:
    """
    Wrapper untuk ML model classification tiket IT Support.
    
    Class ini mengelola seluruh lifecycle model ML:
    - Load artifacts dari disk (model, vectorizer, encoder)
    - Preprocessing text dengan domain-aware rules
    - Prediksi kategori dengan confidence scoring
    - Hot reload untuk update model tanpa restart
    
    Attributes:
        _model_dir: Path ke folder models/
        _version: Versi model yang aktif (v1, v2, dst)
        _model: LightGBM Booster object
        _word_tfidf: TF-IDF vectorizer untuk word n-grams
        _char_tfidf: TF-IDF vectorizer untuk char n-grams
        _label_encoder: Mapping index -> label
        _is_loaded: Flag apakah model sudah loaded
        _model_version: String versi model
        _metadata: Metadata dari training (F1 score, samples, dll)
    
    Model Format Support:
        - New: Versioned folder (models/v3/lgb_model.txt)
        - Legacy: Flat files (models/lgb_model_v2.txt)
    
    TF-IDF Format Support:
        - New: Word + Char combined (dictionary dengan 2 vectorizers)
        - Old: Single TF-IDF (untuk backward compatibility)
    """
    
    def __init__(self, config: "Config") -> None:
        """
        Inisialisasi classifier dan load model artifacts dari disk.
        
        Saat init:
        1. Resolve versi model (auto atau spesifik)
        2. Load LightGBM model, TF-IDF vectorizer, label encoder
        3. Load metadata untuk info training
        
        Args:
            config: Object konfigurasi yang berisi model_dir dan model_version
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
        Resolve version string ke versi aktual.
        
        Kalau version = "auto":
            Baca dari current_version.txt (contoh: "v3")
        Kalau version spesifik:
            Return langsung (contoh: "v2")
        
        Args:
            version: String versi ("auto" atau "v1", "v2", dst)
            model_dir: Path ke folder models/
            
        Returns:
            String versi aktual (contoh: "v3")
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
        
        Artifacts yang di-load:
        1. lgb_model.txt/.bin - LightGBM Booster model
        2. tfidf_vectorizer.pkl - TF-IDF vectorizer (word + char)
        3. label_encoder.pkl - Mapping index ke label
        4. metadata.json - Info training (version, F1, samples)
        
        Format Support:
        - New: Versioned folder (models/v3/lgb_model.txt)
        - Legacy: Flat files (models/lgb_model_v2.txt)
        
        TF-IDF Format:
        - New: Dictionary {word_tfidf, char_tfidf} untuk combined features
        - Old: Single TfidfVectorizer object
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
        Hot reload model dari disk tanpa perlu restart bot.
        
        Berguna untuk:
        - Update ke model baru setelah retrain
        - Rollback ke versi sebelumnya
        - Refresh model saat ada perubahan
        
        Flow:
        1. Reset semua state (model, tfidf, encoder)
        2. Load ulang dari disk dengan versi baru/current
        3. Return status sukses/gagal
        
        Args:
            new_version: Versi baru yang akan di-load (contoh: 'v3')
                        Kalau None, reload versi current
        
        Returns:
            True jika reload sukses, False jika gagal
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
        """
        Tentukan status prediksi berdasarkan confidence threshold.
        
        Thresholds:
        - >= 80%: AUTO (langsung pakai)
        - 70-80%: HIGH_REVIEW (kemungkinan benar, perlu konfirmasi)
        - 50-70%: MEDIUM_REVIEW (prioritas review medium)
        - < 50%: MANUAL (harus review, prediksi tidak reliable)
        
        Args:
            confidence: Score confidence 0.0 - 1.0
            
        Returns:
            String status: AUTO, HIGH_REVIEW, MEDIUM_REVIEW, atau MANUAL
        """
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
        Prediksi kategori Symtomps dari text tiket teknisi dan solving ops.
        
        Ini adalah FUNGSI UTAMA untuk klasifikasi tiket.
        
        Flow:
        1. Preprocess text (lowercase, normalisasi, merge)
        2. Vectorize dengan TF-IDF (word + char n-grams)
        3. Predict dengan LightGBM booster
        4. Return label dengan confidence tertinggi
        
        Args:
            tech_raw_text: Raw text dari message teknisi (caption/text)
            solving: Text solving dari ops (opsional, untuk akurasi lebih baik)
        
        Returns:
            PredictionResult berisi:
            - predicted_symtomps: Label kategori (contoh: \"KENDALA ONT/NTE\")
            - ml_confidence: Confidence 0.0-1.0
            - prediction_status: AUTO/HIGH_REVIEW/MEDIUM_REVIEW/MANUAL
            - inference_time_ms: Waktu prediksi dalam ms
        
        Kalau model belum loaded: return empty prediction dengan status MANUAL
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
        """Cek apakah model sudah berhasil di-load dari disk."""
        return self._is_loaded
    
    @property
    def model_version(self) -> str:
        """Get versi model yang aktif (contoh: 'v1', 'v2')."""
        return self._model_version
    
    @property
    def num_classes(self) -> int:
        """Get jumlah kategori/class yang bisa diprediksi."""
        if self._label_encoder:
            if isinstance(self._label_encoder, dict):
                return len(self._label_encoder)
            return len(self._label_encoder.classes_)
        return 0
    
    def get_model_info(self) -> dict:
        """
        Get info ringkas model untuk reporting.
        
        Returns:
            Dictionary berisi version, is_loaded, num_classes, thresholds
        """
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
        """
        Get metadata lengkap model untuk admin commands.
        
        Digunakan oleh /modelstatus untuk menampilkan:
        - Versi model aktif
        - Jumlah classes dan list nama classes
        - Thresholds (AUTO, HIGH, MEDIUM)
        - Training info (samples, accuracy, trained_at)
        
        Returns:
            Dictionary dengan semua metadata model
        """
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
