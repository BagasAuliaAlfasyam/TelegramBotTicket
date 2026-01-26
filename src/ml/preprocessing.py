"""
Text Preprocessing Module for IT Support Tickets
=================================================

Modul preprocessing teks khusus untuk tiket IT support Indonesia.
Menyediakan preprocessing yang konsisten untuk inference dan training.

Fitur Utama:
    - Preserve istilah IT (OTP, TOTP, WO, SC, dll)
    - Expand singkatan umum (moban → mohon bantuan)
    - Handle code-switching Indonesia + English IT terms
    - Smart merge teks teknisi + solving dengan [SEP] separator
    - Normalisasi kode tiket, NIK, nomor telepon

Author: Bagas Aulia Alfasyam
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


class ITSupportTextPreprocessor:
    """
    Preprocessor teks khusus untuk tiket IT support.
    
    Keputusan desain utama:
        1. Preserve istilah IT penting (OTP, TOTP, WO, SC, modem, dll)
        2. Normalisasi singkatan umum (moban → mohon bantuan)
        3. Handle code-switching Indonesia + English IT terms
        4. Smart merge tech_raw_text + solving dengan separator [SEP]
    
    Attributes:
        ABBREVIATIONS: Dict mapping singkatan → bentuk lengkap
        IT_TERMS: Set istilah IT yang harus dipertahankan
        use_separator: Apakah gunakan [SEP] token sebagai pemisah
    
    Contoh:
        >>> preprocessor = ITSupportTextPreprocessor()
        >>> preprocessor.preprocess("moban reset pwd", "sudah reset")
        "mohon bantuan reset password [SEP] sudah reset"
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
        Inisialisasi preprocessor.
        
        Args:
            use_separator: Apakah gunakan token [SEP] antara tech_text dan solving.
                          Default True untuk memberikan signal batas antar bagian.
        """
        self.use_separator = use_separator
    
    def preprocess(self, tech_raw_text: str, solving: str = "") -> str:
        """
        Preprocess dan merge kolom teks.
        
        Proses:
            1. Bersihkan kedua teks dengan _clean_text()
            2. Gabungkan dengan separator [SEP] jika use_separator=True
            3. Handle kasus salah satu teks kosong
        
        Args:
            tech_raw_text: Teks mentah dari teknisi
            solving: Teks solving dari ops (optional)
        
        Returns:
            str: Teks yang sudah dibersihkan dan digabung
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
        Bersihkan satu teks dengan processing domain-aware.
        
        Tahapan cleaning:
            1. Remove URLs, mentions, emails
            2. Preserve konten hashtag (hapus #)
            3. Remove nomor telepon Indonesia (08xx, +62xx)
            4. Normalisasi kode WO/SC ke placeholder
            5. Normalisasi NIK/laborcode
            6. Expand singkatan dari ABBREVIATIONS
            7. Remove angka standalone
            8. Remove special characters
            9. Collapse whitespace
        
        Args:
            text: Teks mentah yang akan dibersihkan
        
        Returns:
            str: Teks yang sudah dibersihkan
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
        Batch preprocess DataFrame.
        
        Memproses seluruh DataFrame sekaligus, lebih efisien untuk
        training atau batch prediction.
        
        Args:
            df: DataFrame dengan kolom teks
            text_col: Nama kolom teks teknisi (default: 'tech raw text')
            solving_col: Nama kolom solving (default: 'solving')
        
        Returns:
            np.ndarray: Array teks yang sudah dipreprocess
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
    Fungsi convenience untuk preprocess teks.
    
    Menggunakan singleton preprocessor untuk kemudahan.
    Cocok untuk inference single text.
    
    Args:
        tech_raw_text: Teks mentah dari teknisi
        solving: Teks solving dari ops (optional)
    
    Returns:
        str: Teks yang sudah dipreprocess
    """
    return _preprocessor.preprocess(tech_raw_text, solving)


def clean_text_simple(text: str) -> str:
    """
    Cleaning teks sederhana (backward compatibility).
    
    Fungsi ini dipertahankan untuk kompatibilitas dengan kode lama
    yang hanya membutuhkan cleaning tanpa merge.
    
    Args:
        text: Teks yang akan dibersihkan
    
    Returns:
        str: Teks yang sudah dibersihkan
    """
    return _preprocessor._clean_text(text)
