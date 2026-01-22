"""
Text Preprocessing Module for IT Support Tickets
=================================================
Domain-aware text preprocessing for Indonesian IT support tickets.
Consistent preprocessing for both inference and training.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


class ITSupportTextPreprocessor:
    """
    Domain-aware text preprocessor for IT support tickets.
    
    Key design decisions:
    1. Preserve IT-specific terms (OTP, TOTP, WO, SC, etc.)
    2. Normalize common abbreviations (moban -> mohon bantuan)
    3. Handle code-switched text (Indonesian + English IT terms)
    4. Smart merge of tech_raw_text + solving columns
    """
    
    # Domain-specific abbreviation mappings
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
    
    # IT-specific terms to preserve (these are important signals!)
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
        """
        Initialize preprocessor.
        
        Args:
            use_separator: Whether to use [SEP] token between tech_text and solving
        """
        self.use_separator = use_separator
    
    def preprocess(self, tech_raw_text: str, solving: str = "") -> str:
        """
        Preprocess and merge text columns.
        
        Args:
            tech_raw_text: Raw text from technician
            solving: Solving text from ops
            
        Returns:
            Cleaned and merged text
        """
        # Clean both texts
        tech_text = self._clean_text(tech_raw_text)
        solving_text = self._clean_text(solving)
        
        # Merge with separator
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
        """
        Clean single text with domain-aware processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or pd.isna(text) or str(text).lower() == 'nan':
            return ""
        
        text = str(text).lower()
        
        # 1. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
        
        # 2. Remove mentions but keep context
        text = re.sub(r'@\w+', ' ', text)
        
        # 3. Preserve hashtags content (remove #)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # 4. Remove phone numbers (Indonesian format)
        text = re.sub(r'\+?62\d{8,13}', ' ', text)
        text = re.sub(r'08\d{8,13}', ' ', text)
        
        # 5. Remove emails
        text = re.sub(r'\S+@\S+\.\S+', ' ', text)
        
        # 6. Normalize WO/SC codes (keep type, remove specific ID)
        text = re.sub(r'(wo|sc)[-\s]?(\d+)', r'\1_code', text)
        
        # 7. Normalize NIK/laborcode patterns
        text = re.sub(r'(nik|laborcode)\s*[:\-]?\s*\d+', r'\1', text)
        
        # 8. Expand abbreviations
        for abbr, full in self.ABBREVIATIONS.items():
            text = re.sub(rf'\b{abbr}\b', full, text)
        
        # 9. Remove standalone numbers (but keep numbers in context)
        text = re.sub(r'\b\d+\b', ' ', text)
        
        # 10. Remove special characters but keep underscores and [SEP]
        text = re.sub(r'[^\w\s\[\]]', ' ', text)
        
        # 11. Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def batch_preprocess(self, df: pd.DataFrame, 
                         text_col: str = 'tech raw text',
                         solving_col: str = 'solving') -> np.ndarray:
        """
        Batch preprocess a DataFrame.
        
        Args:
            df: DataFrame with text columns
            text_col: Name of tech text column
            solving_col: Name of solving column
            
        Returns:
            Array of preprocessed texts
        """
        results = []
        for _, row in df.iterrows():
            tech_text = row.get(text_col, "") or ""
            solving = row.get(solving_col, "") or ""
            results.append(self.preprocess(tech_text, solving))
        return np.array(results)


# Singleton instance for convenience
_preprocessor = ITSupportTextPreprocessor()


def preprocess_text(tech_raw_text: str, solving: str = "") -> str:
    """
    Convenience function to preprocess text.
    
    Args:
        tech_raw_text: Raw text from technician
        solving: Solving text from ops
        
    Returns:
        Preprocessed text
    """
    return _preprocessor.preprocess(tech_raw_text, solving)


def clean_text_simple(text: str) -> str:
    """
    Simple text cleaning (for backward compatibility).
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    return _preprocessor._clean_text(text)
