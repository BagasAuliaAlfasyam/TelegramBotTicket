"""
Core Configuration Module
==========================

Management konfigurasi terpusat menggunakan environment variables.

Mendukung:
    - .env file untuk konfigurasi default
    - .env.local untuk override development lokal
    - System environment variables

Prioritas loading (tertinggi ke terendah):
    1. .env.local (jika ada)
    2. .env
    3. System environment variables

Author: Bagas Aulia Alfasyam
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

_LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Konfigurasi aplikasi yang diload dari environment variables.
    
    Semua setting aplikasi dikumpulkan di class ini untuk memudahkan
    management dan validasi. Mendukung .env.local untuk override lokal.
    
    Attributes:
        telegram_collecting_bot_token: Token bot untuk collecting tiket
        telegram_reporting_bot_token: Token bot untuk reporting/admin
        telegram_ops_chat_id: Chat ID grup ops
        telegram_tech_chat_id: Chat ID grup teknisi
        telegram_admin_chat_id: Chat ID untuk notifikasi admin
        google_service_account_json: Path ke file credentials Google
        google_spreadsheet_name: Nama spreadsheet Google Sheets
        google_worksheet_name: Nama worksheet (default: Logs)
        aws_access_key_id: AWS/MinIO access key
        aws_secret_access_key: AWS/MinIO secret key
        s3_bucket_name: Nama bucket S3
        s3_region: Region S3 (default: ap-southeast-1)
        s3_endpoint_url: Custom endpoint untuk MinIO
        s3_public_url: Base URL publik untuk file
        model_dir: Direktori model ML
        model_version: Versi model ("auto" = baca dari current_version.txt)
        threshold_auto: Threshold confidence untuk AUTO (default: 0.90)
        threshold_high: Threshold untuk HIGH REVIEW (default: 0.85)
        threshold_medium: Threshold untuk MEDIUM REVIEW (default: 0.70)
        timezone: Timezone aplikasi (default: Asia/Jakarta)
        debug: Mode debug
        admin_user_ids: List ID user admin
    """
    
    # Telegram Bot tokens
    telegram_collecting_bot_token: str = ""
    telegram_reporting_bot_token: str = ""
    
    # Chat IDs
    telegram_ops_chat_id: int = 0
    telegram_tech_chat_id: int = 0
    telegram_admin_chat_id: int = 0
    
    # Google Sheets
    google_service_account_json: Path = field(default_factory=lambda: Path("service_account.json"))
    google_spreadsheet_name: str = ""
    google_worksheet_name: str = "Logs"
    
    # AWS S3 / MinIO
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket_name: str = ""
    s3_region: str = "ap-southeast-1"
    s3_endpoint_url: str = ""  # For MinIO or S3-compatible storage
    s3_public_url: str = ""    # Public URL for accessing uploaded files
    
    # ML Model
    model_dir: Path = field(default_factory=lambda: Path("models"))
    model_version: str = "auto"  # "auto" = read from current_version.txt
    
    # ML Thresholds
    threshold_auto: float = 0.90
    threshold_high: float = 0.85
    threshold_medium: float = 0.70
    
    # General
    timezone: str = "Asia/Jakarta"
    debug: bool = False
    
    # Admin user IDs (comma-separated in env)
    admin_user_ids: list[int] = field(default_factory=list)
    
    @classmethod
    def from_env(cls, env_path: Optional[Path] = None) -> "Config":
        """
        Load konfigurasi dari environment variables.
        
        Prioritas loading:
            1. .env.local (jika ada) - untuk override development
            2. .env - konfigurasi default
            3. System environment variables
        
        Args:
            env_path: Optional path ke file .env custom
        
        Returns:
            Config: Instance konfigurasi yang sudah diload
        
        Note:
            Untuk MODEL_VERSION="auto", versi akan dibaca dari
            current_version.txt saat MLClassifier di-initialize.
        """
        # Determine base path
        if env_path:
            base_path = env_path.parent
        else:
            base_path = Path(__file__).parent.parent.parent
        
        # Load .env files (local overrides default)
        env_local = base_path / ".env.local"
        env_default = base_path / ".env"
        
        if env_local.exists():
            load_dotenv(env_local, override=True)
            _LOGGER.debug("Loaded .env.local from %s", env_local)
        elif env_default.exists():
            load_dotenv(env_default)
            _LOGGER.debug("Loaded .env from %s", env_default)
        
        # Parse admin/allowed user IDs (support both ADMIN_USER_IDS and ALLOWED_OPS_USER_IDS)
        admin_ids_str = os.getenv("ADMIN_USER_IDS", "") or os.getenv("ALLOWED_OPS_USER_IDS", "")
        admin_ids = []
        if admin_ids_str:
            try:
                admin_ids = [int(x.strip()) for x in admin_ids_str.split(",") if x.strip()]
            except ValueError:
                _LOGGER.warning("Invalid user IDs format: %s", admin_ids_str)
        
        # Build config
        config = cls(
            telegram_collecting_bot_token=os.getenv("TELEGRAM_BOT_TOKEN_COLLECTING", os.getenv("TELEGRAM_BOT_TOKEN", "")),
            telegram_reporting_bot_token=os.getenv("TELEGRAM_BOT_TOKEN_REPORTING", ""),
            telegram_ops_chat_id=int(os.getenv("TARGET_GROUP_COLLECTING", "0") or "0"),
            telegram_tech_chat_id=int(os.getenv("TECH_CHAT_ID", "0") or "0"),
            telegram_admin_chat_id=int(os.getenv("TARGET_GROUP_REPORTING", os.getenv("ADMIN_CHAT_ID", "0")) or "0"),
            google_service_account_json=Path(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")),
            google_spreadsheet_name=os.getenv("GOOGLE_SPREADSHEET_NAME", ""),
            google_worksheet_name=os.getenv("GOOGLE_WORKSHEET_NAME", "Logs"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            s3_bucket_name=os.getenv("AWS_S3_BUCKET", os.getenv("S3_BUCKET_NAME", "")),
            s3_region=os.getenv("AWS_S3_REGION", os.getenv("S3_REGION", "ap-southeast-1")),
            s3_endpoint_url=os.getenv("AWS_S3_ENDPOINT", os.getenv("S3_ENDPOINT_URL", "")),
            s3_public_url=os.getenv("AWS_S3_PUBLIC_BASE_URL", os.getenv("S3_PUBLIC_URL", "")),
            model_dir=Path(os.getenv("MODEL_DIR", "models")),
            model_version=os.getenv("MODEL_VERSION", "auto"),  # auto = read current_version.txt
            threshold_auto=float(os.getenv("ML_THRESHOLD_AUTO", "0.90")),
            threshold_high=float(os.getenv("ML_THRESHOLD_HIGH", "0.85")),
            threshold_medium=float(os.getenv("ML_THRESHOLD_MEDIUM", "0.70")),
            timezone=os.getenv("TIMEZONE", "Asia/Jakarta"),
            debug=os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"),
            admin_user_ids=admin_ids,
        )
        
        return config
    
    def validate(self) -> list[str]:
        """
        Validasi konfigurasi dan return list error.
        
        Mengecek apakah konfigurasi wajib sudah diisi:
            - TELEGRAM_BOT_TOKEN_COLLECTING
            - GOOGLE_SPREADSHEET_NAME
            - Service account file exists
        
        Returns:
            list[str]: List pesan error (kosong jika valid)
        """
        errors = []
        
        if not self.telegram_collecting_bot_token:
            errors.append("TELEGRAM_BOT_TOKEN_COLLECTING is required")
        
        if not self.google_spreadsheet_name:
            errors.append("GOOGLE_SPREADSHEET_NAME is required")
        
        if not self.google_service_account_json.exists():
            errors.append(f"Service account file not found: {self.google_service_account_json}")
        
        return errors
    
    def __repr__(self) -> str:
        """
        Representasi aman yang menyembunyikan secrets.
        
        Hanya menampilkan informasi non-sensitif seperti
        nama spreadsheet, versi model, dan timezone.
        """
        return (
            f"Config("
            f"spreadsheet={self.google_spreadsheet_name!r}, "
            f"model={self.model_version}, "
            f"timezone={self.timezone})"
        )


def setup_logging(debug: bool = False) -> None:
    """
    Setup konfigurasi logging.
    
    Mengatur format log dan level berdasarkan mode debug.
    Juga meredam noise dari library eksternal (httpx, telegram, gspread).
    
    Args:
        debug: Jika True, gunakan level DEBUG. Jika False, gunakan INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("gspread").setLevel(logging.WARNING)
