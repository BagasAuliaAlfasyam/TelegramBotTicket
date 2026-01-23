"""
Core Configuration Module
==========================
Centralized configuration management using environment variables.
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
    Application configuration loaded from environment variables.
    
    Supports .env.local for local development overrides.
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
        Load configuration from environment variables.
        
        Priority:
        1. .env.local (if exists)
        2. .env
        3. System environment variables
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
        Validate configuration and return list of errors.
        
        Returns:
            List of error messages (empty if valid)
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
        """Safe representation that hides secrets."""
        return (
            f"Config("
            f"spreadsheet={self.google_spreadsheet_name!r}, "
            f"model={self.model_version}, "
            f"timezone={self.timezone})"
        )


def setup_logging(debug: bool = False) -> None:
    """
    Setup logging configuration.
    
    Args:
        debug: Enable debug level logging
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
