#!/usr/bin/env python
"""
Retrain Script - Pipeline Compatible
=====================================
Script untuk retraining model ML yang match dengan notebook pipeline.

Features:
- Domain-aware preprocessing (ITSupportTextPreprocessor)
- Word TF-IDF (1-3 n-grams) + Char TF-IDF (3-5 n-grams)
- LightGBM with optimized params (from Optuna tuning)
- Optional probability calibration
- Incremental versioning (v2 → v3, etc.)

Usage:
    # Manual retrain dengan data baru dari Sheets
    python scripts/retrain.py
    
    # Dengan threshold check (hanya retrain jika reviewed >= 100)
    python scripts/retrain.py --check-threshold 100
    
    # Force retrain (skip threshold check)
    python scripts/retrain.py --force

Pipeline Flow:
    1. Load Master data + Reviewed data dari ML_Tracking (jika ada)
    2. Preprocess dengan ITSupportTextPreprocessor
    3. TF-IDF Vectorization (word 1-3gram + char 3-5gram)
    4. Train LightGBM dengan params optimal
    5. Calibration (optional)
    6. Export artifacts ke models/
    7. Notify admin via Telegram (optional)
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

# Optional imports
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    gspread = None
    Credentials = None
    HAS_GSPREAD = False

# Setup path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
_LOGGER = logging.getLogger(__name__)


# =============================================================================
# ITSupportTextPreprocessor (exact copy from notebook)
# =============================================================================

class ITSupportTextPreprocessor(BaseEstimator, TransformerMixin):
    """Domain-aware text preprocessor for IT support tickets."""
    
    ABBREVIATIONS = {
        'moban': 'mohon bantuan', 'tks': 'terima kasih', 'thx': 'terima kasih',
        'pls': 'please', 'plz': 'please', 'yg': 'yang', 'utk': 'untuk',
        'dgn': 'dengan', 'tdk': 'tidak', 'gak': 'tidak', 'ga': 'tidak',
        'gk': 'tidak', 'krn': 'karena', 'blm': 'belum', 'sdh': 'sudah',
        'udh': 'sudah', 'lg': 'lagi', 'jg': 'juga', 'bs': 'bisa',
        'trs': 'terus', 'klo': 'kalau', 'kl': 'kalau', 'sm': 'sama',
        'dr': 'dari', 'ke': 'ke', 'sy': 'saya', 'ak': 'aku', 'min': 'admin',
    }
    
    IT_TERMS = {
        'otp', 'totp', 'mfa', '2fa', 'auth', 'authenticator',
        'login', 'logout', 'password', 'pwd', 'pass', 'reset', 'locked', 'unlock',
        'wo', 'sc', 'inet', 'orbit', 'modem', 'ont', 'onu', 'router',
        'provisioning', 'aktivasi', 'instalasi', 'migrasi',
        'ibooster', 'mytech', 'myi', 'lensa', 'starclick', 'mytens',
        'teknisi', 'user', 'email', 'nik', 'laborcode', 'nopel',
        'witel', 'regional', 'mitra', 'vendor',
        'error', 'failed', 'gagal', 'timeout', 'reject', 'pending',
        'app', 'apk', 'update', 'version', 'install', 'uninstall',
        'sync', 'sinkron', 'refresh', 'clear', 'cache',
        'bai', 'reopen', 'close', 'done', 'selesai',
    }
    
    def __init__(self, merge_columns: bool = True, text_col: str = 'tech raw text', 
                 solving_col: str = 'solving'):
        self.merge_columns = merge_columns
        self.text_col = text_col
        self.solving_col = solving_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.merge_columns and self.solving_col in X.columns:
                texts = X.apply(self._merge_and_clean, axis=1)
            else:
                texts = X[self.text_col].apply(self._clean_text)
        else:
            texts = pd.Series(X).apply(self._clean_text)
        return texts.values
    
    def _merge_and_clean(self, row) -> str:
        tech_text = str(row.get(self.text_col, '')) if pd.notna(row.get(self.text_col)) else ''
        solving = str(row.get(self.solving_col, '')) if pd.notna(row.get(self.solving_col)) else ''
        
        tech_text = self._clean_text(tech_text)
        solving = self._clean_text(solving)
        
        if tech_text and solving:
            combined = f"{tech_text} [SEP] {solving}"
        elif tech_text:
            combined = tech_text
        elif solving:
            combined = f"[SEP] {solving}"
        else:
            combined = ''
        
        return combined.strip()
    
    def _clean_text(self, text: str) -> str:
        if pd.isna(text) or text == '' or text == 'nan':
            return ''
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'\+?62\d{8,13}', ' ', text)
        text = re.sub(r'08\d{8,13}', ' ', text)
        text = re.sub(r'\S+@\S+\.\S+', ' ', text)
        text = re.sub(r'(wo|sc)[-\s]?(\d+)', r'\1_code', text)
        text = re.sub(r'(nik|laborcode)\s*[:\-]?\s*\d+', r'\1', text)
        
        for abbr, full in self.ABBREVIATIONS.items():
            text = re.sub(rf'\b{abbr}\b', full, text)
        
        text = re.sub(r'\b\d+\b', ' ', text)
        text = re.sub(r'[^\w\s\[\]]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


# =============================================================================
# Label normalization (from notebook)
# =============================================================================

LABEL_MAPPING = {
    'USER_AUTH_DEVID_REQUIRED': 'USER AUTH DEVID REQUIRED',
    'USER_AUTH_FAILED': 'USER AUTH FAILED',
    'USER_AUTH_LOCKED': 'USER AUTH LOCKED',
    'USER_AUTH_NOT_ACTIVE': 'USER AUTH NOT ACTIVE',
}


# =============================================================================
# LightGBM optimal params (from Optuna tuning)
# =============================================================================

OPTIMAL_LGBM_PARAMS = {
    'n_estimators': 300,
    'max_depth': 7,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


# =============================================================================
# Main Retraining Class
# =============================================================================

class ModelRetrainer:
    """
    Retrainer yang match dengan notebook pipeline.
    """
    
    def __init__(
        self,
        config: Config,
        master_data_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.config = config
        self.master_data_path = master_data_path or Path(
            PROJECT_ROOT.parent / "Analyst" / "artifacts" / "master_data.csv"
        )
        self.output_dir = output_dir or config.model_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Artifacts
        self.preprocessor = None
        self.word_tfidf = None
        self.char_tfidf = None
        self.label_encoder = None
        self.model = None
        self.calibrated_model = None
        
        # Stats
        self.metrics = {}
        self.new_version = self._get_next_version()
    
    def _get_next_version(self) -> str:
        """
        Get next version number.
        
        Logic:
        1. Check current_version.txt (if exists)
        2. Check versions.json (if exists)
        3. Check existing version folders (v1, v2, v3...)
        4. If nothing found, start with v1
        """
        # Method 1: Check current_version.txt
        current_version_file = self.output_dir / "current_version.txt"
        if current_version_file.exists():
            current = current_version_file.read_text().strip()
            match = re.match(r'v(\d+)', current)
            if match:
                return f"v{int(match.group(1)) + 1}"    
        
        # Method 2: Check versions.json
        versions_file = self.output_dir / "versions.json"
        if versions_file.exists():
            try:
                import json
                with open(versions_file, 'r') as f:
                    versions = json.load(f)
                if versions.get('current'):
                    current = versions['current']
                    match = re.match(r'v(\d+)', current)
                    if match:
                        return f"v{int(match.group(1)) + 1}"
            except Exception:
                pass
        
        # Method 3: Check existing version folders
        existing_versions = []
        for path in self.output_dir.glob('v*'):
            if path.is_dir():
                match = re.match(r'v(\d+)', path.name)
                if match:
                    existing_versions.append(int(match.group(1)))
        
        if existing_versions:
            return f"v{max(existing_versions) + 1}"
        
        # Method 4: Start fresh with v1
        return "v1"
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load training data from ML_Tracking sheet.
        
        ML_Tracking is the single source of truth for training data.
        It should be synced from Logs sheet using sync_training_data.py
        
        Columns expected: tech_message_id, timestamp, tech_raw_text, solving, Symtomps
        """
        _LOGGER.info("Loading training data from ML_Tracking sheet...")
        
        df = None
        
        # Primary: Load from ML_Tracking sheet (single source of truth)
        if HAS_GSPREAD:
            try:
                df = self._load_from_ml_tracking()
                if df is not None and len(df) > 0:
                    _LOGGER.info(f"  Loaded from ML_Tracking: {len(df)} rows")
            except Exception as e:
                _LOGGER.warning(f"  Could not load from ML_Tracking: {e}")
        
        # Fallback: Load from CSV if ML_Tracking failed
        if df is None or len(df) == 0:
            if self.master_data_path.exists():
                _LOGGER.info("  Fallback: Loading from CSV...")
                df = pd.read_csv(self.master_data_path)
                _LOGGER.info(f"  Loaded from CSV: {len(df)} rows")
            else:
                raise ValueError("No training data found! Run sync_training_data.py first.")
        
        # Normalize columns
        df = self._normalize_dataframe(df)
        
        _LOGGER.info(f"Total training data: {len(df)} rows, {df['Symtomps'].nunique()} classes")
        
        return df
    
    def _load_from_ml_tracking(self) -> Optional[pd.DataFrame]:
        """Load training data from ML_Tracking sheet."""
        if not HAS_GSPREAD:
            return None
        
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Try different credential paths
        cred_paths = [
            self.config.google_service_account_json,
            PROJECT_ROOT / 'white-set-293710-9cca41a1afd6.json',
            PROJECT_ROOT / 'service_account.json',
        ]
        
        cred_file = None
        for path in cred_paths:
            if path.exists():
                cred_file = path
                break
        
        if cred_file is None:
            raise FileNotFoundError(f"No credentials file found")
        
        _LOGGER.info(f"  Using credentials: {cred_file.name}")
        
        credentials = Credentials.from_service_account_file(
            str(cred_file), 
            scopes=scopes
        )
        client = gspread.authorize(credentials)
        
        # Use config spreadsheet name or fallback
        spreadsheet_name = self.config.google_spreadsheet_name
        if not spreadsheet_name:
            spreadsheet_name = 'Log_Tiket_MyTech_ML_Test'
        
        spreadsheet = client.open(spreadsheet_name)
        worksheet = spreadsheet.worksheet("ML_Tracking")
        
        data = worksheet.get_all_values()
        if len(data) <= 1:
            _LOGGER.warning("  ML_Tracking sheet is empty!")
            return None
        
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # ========================================
        # FILTER BY REVIEW STATUS
        # Only use APPROVED or CORRECTED data for training
        # - APPROVED: prediction was correct
        # - CORRECTED: prediction was wrong, label has been fixed
        # - pending/SKIPPED: not reviewed or ignored
        # ========================================
        if 'review_status' in df.columns:
            valid_statuses = ['APPROVED', 'CORRECTED']
            before_count = len(df)
            df = df[df['review_status'].isin(valid_statuses)]
            after_count = len(df)
            _LOGGER.info(f"  Filtered by review_status (APPROVED/CORRECTED): {after_count}/{before_count} rows")
            
            if after_count == 0:
                _LOGGER.warning("  No approved/corrected data found in ML_Tracking!")
                return None
        else:
            _LOGGER.warning("  review_status column not found - using all data (legacy mode)")
        
        # Rename columns to match expected format
        column_mapping = {
            'tech_raw_text': 'tech raw text',
        }
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Select only training columns
        training_cols = []
        for col in ['tech raw text', 'tech_raw_text', 'solving', 'Symtomps']:
            if col in df.columns:
                training_cols.append(col)
        
        if 'Symtomps' not in df.columns:
            _LOGGER.warning("  Symtomps column not found in ML_Tracking!")
            return None
        
        return df[training_cols]
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names and clean data.
        
        IMPORTANT: Data dari gspread (Google Sheets) tidak memiliki NaN!
        - Cell kosong menjadi empty string ''
        - df.info() tidak akan menunjukkan null
        - Perlu convert empty string → NaN untuk konsistensi
        """
        # Rename columns if needed
        column_mapping = {
            'tech_raw_text': 'tech raw text',
            'reviewed_symtomps': 'Symtomps',
            'predicted_symtomps': 'Symtomps',  # fallback
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure required columns
        required = ['tech raw text', 'Symtomps']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Fill missing solving column
        if 'solving' not in df.columns:
            df['solving'] = ''
        
        # ========== EXPLICIT CLEANING (handle gspread empty strings) ==========
        # Replace empty strings & common null-like values with NaN
        null_values = ['', ' ', 'nan', 'None', 'NaN', 'null', 'NULL', 'NA', 'N/A']
        
        for col in ['tech raw text', 'solving', 'Symtomps']:
            if col in df.columns:
                # Convert to string first (in case of mixed types)
                df[col] = df[col].astype(str)
                # Replace null-like values with NaN
                df[col] = df[col].replace(null_values, np.nan)
                # Also handle whitespace-only strings
                df[col] = df[col].apply(lambda x: np.nan if pd.notna(x) and str(x).strip() == '' else x)
        
        # Log status before cleaning
        _LOGGER.info("Data status before cleaning:")
        _LOGGER.info(f"  Total rows: {len(df)}")
        _LOGGER.info(f"  tech raw text NaN/empty: {df['tech raw text'].isna().sum()}")
        _LOGGER.info(f"  solving NaN/empty: {df['solving'].isna().sum()}")
        _LOGGER.info(f"  Symtomps NaN/empty: {df['Symtomps'].isna().sum()}")
        
        # Fill NaN for text columns (they can be empty, we'll combine them)
        df['tech raw text'] = df['tech raw text'].fillna('')
        df['solving'] = df['solving'].fillna('')
        
        # Drop rows with empty Symtomps (target column MUST have value)
        before = len(df)
        df = df.dropna(subset=['Symtomps'])
        after = len(df)
        if before - after > 0:
            _LOGGER.info(f"  Dropped {before - after} rows with empty Symtomps")
        
        # Extra safety: strip whitespace from Symtomps and remove empty
        df['Symtomps'] = df['Symtomps'].str.strip()
        df = df[df['Symtomps'] != '']
        
        _LOGGER.info(f"  Final valid rows: {len(df)}")
        
        # Normalize labels (merge similar classes)
        df['Symtomps'] = df['Symtomps'].replace(LABEL_MAPPING)
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> tuple:
        """Preprocess and vectorize data."""
        _LOGGER.info("Preprocessing...")
        
        # 1. Text preprocessing
        self.preprocessor = ITSupportTextPreprocessor(
            merge_columns=True,
            text_col='tech raw text',
            solving_col='solving'
        )
        X_text = self.preprocessor.transform(df)
        
        # 2. Label encoding
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['Symtomps'])
        
        _LOGGER.info(f"  Classes: {len(self.label_encoder.classes_)}")
        
        # 3. TF-IDF Word (1-3 n-grams)
        self.word_tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            max_features=10000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        X_word = self.word_tfidf.fit_transform(X_text)
        _LOGGER.info(f"  Word TF-IDF features: {X_word.shape[1]}")
        
        # 4. TF-IDF Char (3-5 n-grams)
        self.char_tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=5000,
            min_df=2,
            max_df=0.95,
        )
        X_char = self.char_tfidf.fit_transform(X_text)
        _LOGGER.info(f"  Char TF-IDF features: {X_char.shape[1]}")
        
        # 5. Combine features
        X = sparse.hstack([X_word, X_char])
        _LOGGER.info(f"  Total features: {X.shape[1]}")
        
        return X, y
    
    def train(self, X, y, calibrate: bool = True) -> None:
        """Train LightGBM model."""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required for training")
        
        _LOGGER.info("Training LightGBM...")
        
        # Create model with optimal params
        self.model = lgb.LGBMClassifier(**OPTIMAL_LGBM_PARAMS)
        
        # Cross-validation for metrics
        # Note: n_jobs=1 to avoid multiprocessing issues on Windows/Python 3.13
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(self.model, X, y, cv=cv, n_jobs=1)
        
        macro_f1 = f1_score(y, y_pred_cv, average='macro')
        weighted_f1 = f1_score(y, y_pred_cv, average='weighted')
        
        self.metrics['macro_f1'] = float(macro_f1)
        self.metrics['weighted_f1'] = float(weighted_f1)
        
        _LOGGER.info(f"  CV Macro F1: {macro_f1:.4f}")
        _LOGGER.info(f"  CV Weighted F1: {weighted_f1:.4f}")
        
        # Fit on full data
        self.model.fit(X, y)
        _LOGGER.info("  Model fitted on full data")
        
        # Calibration
        if calibrate:
            _LOGGER.info("Calibrating probabilities...")
            class_counts = pd.Series(y).value_counts()
            min_samples = class_counts.min()
            
            if min_samples >= 5:
                cv_folds = min(5, min_samples)
                self.calibrated_model = CalibratedClassifierCV(
                    self.model,
                    method='isotonic',
                    cv=cv_folds
                )
                self.calibrated_model.fit(X, y)
                _LOGGER.info(f"  Calibrated with {cv_folds}-fold CV")
            else:
                _LOGGER.warning("  Skipping calibration (insufficient samples)")
                self.calibrated_model = self.model
    
    def save_artifacts(self) -> Path:
        """
        Save all model artifacts to versioned folder.
        
        Structure:
            models/
            ├── v3/
            │   ├── lgb_model.txt
            │   ├── tfidf_vectorizer.pkl
            │   ├── label_encoder.pkl
            │   ├── preprocessor.pkl
            │   └── metadata.json
            └── current_version.txt
        """
        version = self.new_version
        
        # Create versioned folder
        version_dir = self.output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        _LOGGER.info(f"Saving artifacts to {version_dir}...")
        
        # Save LightGBM model (binary format for smaller size & faster load)
        model_path = version_dir / "lgb_model.bin"
        self.model.booster_.save_model(str(model_path))
        _LOGGER.info(f"  ✅ lgb_model.bin")
        
        # Save TF-IDF vectorizer (combined word+char)
        tfidf_path = version_dir / "tfidf_vectorizer.pkl"
        combined_tfidf = {
            'word_tfidf': self.word_tfidf,
            'char_tfidf': self.char_tfidf,
        }
        with open(tfidf_path, 'wb') as f:
            pickle.dump(combined_tfidf, f)
        _LOGGER.info(f"  ✅ tfidf_vectorizer.pkl")
        
        # Save label encoder
        encoder_path = version_dir / "label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        _LOGGER.info(f"  ✅ label_encoder.pkl")
        
        # Save preprocessor
        preprocessor_path = version_dir / "preprocessor.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        _LOGGER.info(f"  ✅ preprocessor.pkl")
        
        # Save metadata
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'classes': list(self.label_encoder.classes_),
            'n_classes': len(self.label_encoder.classes_),
            'features': {
                'word_tfidf': self.word_tfidf.get_feature_names_out().shape[0],
                'char_tfidf': self.char_tfidf.get_feature_names_out().shape[0],
            },
            'metrics': self.metrics,
            'lgbm_params': OPTIMAL_LGBM_PARAMS,
        }
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        _LOGGER.info(f"  ✅ metadata.json")
        
        # Update current_version.txt
        current_version_file = self.output_dir / "current_version.txt"
        with open(current_version_file, 'w') as f:
            f.write(version)
        _LOGGER.info(f"  ✅ current_version.txt → {version}")
        
        # Also save versions history
        self._update_versions_history(version, metadata)
        
        return version_dir
    
    def _update_versions_history(self, version: str, metadata: dict) -> None:
        """Update versions.json with new version info."""
        versions_file = self.output_dir / "versions.json"
        
        if versions_file.exists():
            with open(versions_file, 'r') as f:
                versions = json.load(f)
        else:
            versions = {'versions': []}
        
        # Add new version entry
        version_entry = {
            'version': version,
            'created_at': metadata['created_at'],
            'n_classes': metadata['n_classes'],
            'metrics': metadata['metrics'],
        }
        
        # Remove if exists (update case)
        versions['versions'] = [v for v in versions['versions'] if v['version'] != version]
        versions['versions'].append(version_entry)
        
        # Sort by version descending
        versions['versions'].sort(key=lambda x: x['version'], reverse=True)
        versions['current'] = version
        
        with open(versions_file, 'w') as f:
            json.dump(versions, f, indent=2)
        _LOGGER.info(f"  ✅ versions.json updated")
    
    def run(self, calibrate: bool = True) -> dict:
        """Run full retraining pipeline."""
        _LOGGER.info("="*60)
        _LOGGER.info(f"RETRAINING PIPELINE - Target Version: {self.new_version}")
        _LOGGER.info("="*60)
        
        start_time = datetime.now()
        
        # 1. Load data
        df = self.load_training_data()
        
        # 2. Preprocess
        X, y = self.preprocess(df)
        
        # 3. Train
        self.train(X, y, calibrate=calibrate)
        
        # 4. Save
        output_path = self.save_artifacts()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        _LOGGER.info("="*60)
        _LOGGER.info("RETRAINING COMPLETE!")
        _LOGGER.info("="*60)
        _LOGGER.info(f"  Version: {self.new_version}")
        _LOGGER.info(f"  Classes: {len(self.label_encoder.classes_)}")
        _LOGGER.info(f"  Macro F1: {self.metrics['macro_f1']:.4f}")
        _LOGGER.info(f"  Duration: {duration:.1f}s")
        _LOGGER.info(f"  Output: {output_path}")
        
        return {
            'version': self.new_version,
            'classes': len(self.label_encoder.classes_),
            'metrics': self.metrics,
            'duration_seconds': duration,
            'output_dir': str(output_path),
        }


# =============================================================================
# Threshold Check
# =============================================================================

def check_retrain_threshold(config: Config, threshold: int = 100) -> tuple[bool, int]:
    """
    Check apakah reviewed data sudah mencapai threshold.
    
    Returns:
        (should_retrain, reviewed_count)
    """
    if not HAS_GSPREAD:
        _LOGGER.warning("gspread not available, cannot check threshold")
        return False, 0
    
    try:
        client = gspread.service_account(
            filename=str(config.google_service_account_json)
        )
        spreadsheet = client.open(config.google_spreadsheet_name)
        worksheet = spreadsheet.worksheet("ML_Tracking")
        
        data = worksheet.get_all_values()
        if len(data) <= 1:
            return False, 0
        
        # Count reviewed rows
        reviewed_count = sum(
            1 for row in data[1:] 
            if len(row) > 8 and row[8] == 'reviewed'
        )
        
        should_retrain = reviewed_count >= threshold
        
        return should_retrain, reviewed_count
        
    except Exception as e:
        _LOGGER.warning(f"Failed to check threshold: {e}")
        return False, 0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Retrain ML model with notebook pipeline")
    parser.add_argument(
        '--check-threshold', 
        type=int, 
        default=0,
        help='Only retrain if reviewed count >= threshold (0 = always retrain)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retrain, skip threshold check'
    )
    parser.add_argument(
        '--master-data',
        type=str,
        default=None,
        help='Path to master training data CSV'
    )
    parser.add_argument(
        '--no-calibrate',
        action='store_true',
        help='Skip probability calibration'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Check threshold if specified
    if args.check_threshold > 0 and not args.force:
        should_retrain, reviewed_count = check_retrain_threshold(
            config, args.check_threshold
        )
        _LOGGER.info(f"Reviewed count: {reviewed_count} / Threshold: {args.check_threshold}")
        
        if not should_retrain:
            _LOGGER.info("Threshold not reached, skipping retrain")
            return
    
    # Run retraining
    master_path = Path(args.master_data) if args.master_data else None
    
    retrainer = ModelRetrainer(
        config=config,
        master_data_path=master_path,
    )
    
    result = retrainer.run(calibrate=not args.no_calibrate)
    
    print("\n" + "="*60)
    print("🎉 RETRAINING SUCCESSFUL!")
    print("="*60)
    print(f"\n📦 New model: {result['version']}")
    print(f"📊 Classes: {result['classes']}")
    print(f"📈 Macro F1: {result['metrics']['macro_f1']:.4f}")
    print(f"⏱️ Duration: {result['duration_seconds']:.1f}s")
    print(f"\n➡️ Next steps:")
    print(f"   1. Update .env.local: MODEL_VERSION={result['version']}")
    print(f"   2. Restart bot: python scripts/run_all.py")


if __name__ == "__main__":
    main()
