"""
Text Preprocessing — Single Source of Truth
=============================================
Canonical ITSupportTextPreprocessor used by ALL services.
Eliminates the duplicate preprocessor problem (was in 3 places).
"""
from __future__ import annotations

import re
from typing import Optional

try:
    import pandas as pd
except ImportError:
    pd = None  # Optional — only needed for batch_preprocess

try:
    import numpy as np
except ImportError:
    np = None  # Optional — only needed for batch_preprocess


class ITSupportTextPreprocessor:
    """
    Preprocessor teks khusus untuk tiket IT support Indonesia.
    
    Single source of truth — used by:
    - Prediction API (inference)
    - Training Pipeline (training)
    """
    
    ABBREVIATIONS = {
        'moban': 'mohon bantuan',
        'tks': 'terima kasih',
        'thx': 'terima kasih',
        'pls': 'please',
        'plz': 'please',
        'yg': 'yang',
        'utk': 'untuk',
        'dgn': 'dengan',
        'tdk': 'tidak',
        'gak': 'tidak',
        'ga': 'tidak',
        'gk': 'tidak',
        'krn': 'karena',
        'blm': 'belum',
        'sdh': 'sudah',
        'udh': 'sudah',
        'lg': 'lagi',
        'jg': 'juga',
        'bs': 'bisa',
        'trs': 'terus',
        'klo': 'kalau',
        'kl': 'kalau',
        'sm': 'sama',
        'dr': 'dari',
        'sy': 'saya',
        'ak': 'aku',
        'min': 'admin',
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
    
    def __init__(self, use_separator: bool = True):
        self.use_separator = use_separator
    
    def preprocess(self, tech_raw_text: str, solving: str = "") -> str:
        tech_text = self._clean_text(tech_raw_text)
        solving_text = self._clean_text(solving)
        
        if self.use_separator:
            if tech_text and solving_text:
                return f"{tech_text} [SEP] {solving_text}"
            elif tech_text:
                return tech_text
            elif solving_text:
                return f"[SEP] {solving_text}"
            else:
                return ""
        else:
            return f"{tech_text} {solving_text}".strip()
    
    def _clean_text(self, text: str) -> str:
        if not text or str(text).lower() == 'nan':
            return ""
        if pd is not None and pd.isna(text):
            return ""
        
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
    
    def batch_preprocess(self, df,
                         text_col: str = 'tech raw text',
                         solving_col: str = 'solving'):
        results = []
        for _, row in df.iterrows():
            tech_text = row.get(text_col, "") or ""
            solving = row.get(solving_col, "") or ""
            results.append(self.preprocess(tech_text, solving))
        return np.array(results)


# Singleton
_preprocessor = ITSupportTextPreprocessor()


def preprocess_text(tech_raw_text: str, solving: str = "") -> str:
    """Convenience function using singleton preprocessor."""
    return _preprocessor.preprocess(tech_raw_text, solving)
