"""
Service-Specific Configuration
================================
Each microservice only loads the config it needs.
"""
from __future__ import annotations

import logging
import os
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_LOGGER = logging.getLogger(__name__)

# ============ Trace ID Context ============

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


class _TraceIdFilter(logging.Filter):
    """Injects current trace_id from ContextVar into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get("-")
        return True


def _load_env():
    """Load .env files with priority: .env.local > .env"""
    base_path = Path(__file__).parent.parent.parent  # TelegramBotMyTech root
    env_local = base_path / ".env.local"
    env_default = base_path / ".env"

    if env_local.exists():
        load_dotenv(env_local, override=True)
    elif env_default.exists():
        load_dotenv(env_default)


# Load env on import
_load_env()


@dataclass
class PredictionServiceConfig:
    """Config for ML Prediction API service."""
    # MLflow
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "ticket-classifier"
    mlflow_model_name: str = "ticket-classifier"
    mlflow_s3_endpoint_url: str = ""
    mlflow_bucket_name: str = "mlflow-artifacts"
    mlflow_tracking_username: str = ""
    mlflow_tracking_password: str = ""
    mlflow_public_url: str = ""

    # Gemini
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-2.0-flash"
    gemini_enabled: bool = True
    gemini_cascade_threshold: float = 0.75  # Below this, ask Gemini
    gemini_timeout: float = 10.0  # seconds

    # ML Threshold (2-tier: AUTO >= 0.75, REVIEW < 0.75)
    threshold_auto: float = 0.75

    # Service
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = False

    @classmethod
    def from_env(cls) -> PredictionServiceConfig:
        _load_env()
        return cls(
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
            mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "ticket-classifier"),
            mlflow_model_name=os.getenv("MLFLOW_MODEL_NAME", "ticket-classifier"),
            mlflow_s3_endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", ""),
            mlflow_bucket_name=os.getenv("MLFLOW_BUCKET_NAME", "mlflow-artifacts"),
            mlflow_tracking_username=os.getenv("MLFLOW_TRACKING_USERNAME", ""),
            mlflow_tracking_password=os.getenv("MLFLOW_TRACKING_PASSWORD", ""),
            mlflow_public_url=os.getenv("MLFLOW_PUBLIC_URL", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
            gemini_enabled=os.getenv("GEMINI_ENABLED", "true").lower() in ("true", "1"),
            gemini_cascade_threshold=float(os.getenv("GEMINI_CASCADE_THRESHOLD", "0.75")),
            gemini_timeout=float(os.getenv("GEMINI_TIMEOUT", "10.0")),
            threshold_auto=float(os.getenv("ML_THRESHOLD_AUTO", "0.75")),
            host=os.getenv("SERVICE_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVICE_PORT", "8001")),
            debug=os.getenv("DEBUG", "false").lower() in ("true", "1"),
        )


@dataclass
class DataServiceConfig:
    """Config for Data & Analytics API service."""
    # Google Sheets
    google_service_account_json: Path = field(default_factory=lambda: Path("service_account.json"))
    google_spreadsheet_name: str = ""
    google_worksheet_name: str = "Logs"

    # AWS S3 / MinIO
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket_name: str = ""
    s3_region: str = "ap-southeast-1"
    s3_endpoint_url: str = ""
    s3_public_url: str = ""
    s3_media_prefix: str = "tech-media"

    # Service
    host: str = "0.0.0.0"
    port: int = 8002
    timezone: str = "Asia/Jakarta"
    debug: bool = False

    @classmethod
    def from_env(cls) -> DataServiceConfig:
        _load_env()
        return cls(
            google_service_account_json=Path(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")),
            google_spreadsheet_name=os.getenv("GOOGLE_SPREADSHEET_NAME", ""),
            google_worksheet_name=os.getenv("GOOGLE_WORKSHEET_NAME", "Logs"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            s3_bucket_name=os.getenv("AWS_S3_BUCKET", ""),
            s3_region=os.getenv("AWS_S3_REGION", "us-east-1"),
            s3_endpoint_url=os.getenv("AWS_S3_ENDPOINT", ""),
            s3_public_url=os.getenv("AWS_S3_PUBLIC_BASE_URL", ""),
            s3_media_prefix=os.getenv("S3_MEDIA_PREFIX") or os.getenv("AWS_S3_MEDIA_PREFIX", "tech-media"),
            host=os.getenv("SERVICE_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVICE_PORT", "8002")),
            timezone=os.getenv("TIMEZONE", "Asia/Jakarta"),
            debug=os.getenv("DEBUG", "false").lower() in ("true", "1"),
        )


@dataclass
class CollectorBotConfig:
    """Config for Collector Bot service."""
    # Telegram
    telegram_collecting_bot_token: str = ""
    telegram_admin_chat_id: int = 0
    admin_user_ids: list[int] = field(default_factory=list)

    # Reporting bot for notifications
    telegram_reporting_bot_token: str = ""

    # Service URLs (internal Docker network)
    prediction_api_url: str = "http://prediction-api:8001"
    data_api_url: str = "http://data-api:8002"

    # General
    timezone: str = "Asia/Jakarta"
    debug: bool = False

    @classmethod
    def from_env(cls) -> CollectorBotConfig:
        _load_env()
        admin_ids_str = os.getenv("ADMIN_USER_IDS", "") or os.getenv("ALLOWED_OPS_USER_IDS", "")
        admin_ids = []
        if admin_ids_str:
            try:
                admin_ids = [int(x.strip()) for x in admin_ids_str.split(",") if x.strip()]
            except ValueError:
                pass

        return cls(
            telegram_collecting_bot_token=os.getenv("TELEGRAM_BOT_TOKEN_COLLECTING", ""),
            telegram_admin_chat_id=int(os.getenv("TELEGRAM_ADMIN_CHAT_ID", "") or os.getenv("TARGET_GROUP_REPORTING", "0") or "0"),
            admin_user_ids=admin_ids,
            telegram_reporting_bot_token=os.getenv("TELEGRAM_BOT_TOKEN_REPORTING", ""),
            prediction_api_url=os.getenv("PREDICTION_API_URL", "http://prediction-api:8001"),
            data_api_url=os.getenv("DATA_API_URL", "http://data-api:8002"),
            timezone=os.getenv("TIMEZONE", "Asia/Jakarta"),
            debug=os.getenv("DEBUG", "false").lower() in ("true", "1"),
        )


@dataclass
class AdminBotConfig:
    """Config for Admin/Reporting Bot service."""
    # Telegram
    telegram_reporting_bot_token: str = ""
    admin_user_ids: list[int] = field(default_factory=list)
    target_group_reporting: int = 0   # TARGET_GROUP_REPORTING chat id for hourly stats

    # Service URLs
    prediction_api_url: str = "http://prediction-api:8001"
    data_api_url: str = "http://data-api:8002"
    training_api_url: str = "http://training-api:8005"

    # General
    timezone: str = "Asia/Jakarta"
    debug: bool = False

    @classmethod
    def from_env(cls) -> AdminBotConfig:
        _load_env()
        admin_ids_str = os.getenv("ADMIN_USER_IDS", "") or os.getenv("ALLOWED_OPS_USER_IDS", "")
        admin_ids = []
        if admin_ids_str:
            try:
                admin_ids = [int(x.strip()) for x in admin_ids_str.split(",") if x.strip()]
            except ValueError:
                pass

        return cls(
            telegram_reporting_bot_token=os.getenv("TELEGRAM_BOT_TOKEN_REPORTING", ""),
            admin_user_ids=admin_ids,
            target_group_reporting=int(os.getenv("TARGET_GROUP_REPORTING", "0") or "0"),
            prediction_api_url=os.getenv("PREDICTION_API_URL", "http://prediction-api:8001"),
            data_api_url=os.getenv("DATA_API_URL", "http://data-api:8002"),
            training_api_url=os.getenv("TRAINING_API_URL", "http://training-api:8005"),
            timezone=os.getenv("TIMEZONE", "Asia/Jakarta"),
            debug=os.getenv("DEBUG", "false").lower() in ("true", "1"),
        )


@dataclass
class TrainingServiceConfig:
    """Config for Training Pipeline service."""
    # Google Sheets (for reading training data)
    google_service_account_json: Path = field(default_factory=lambda: Path("service_account.json"))
    google_spreadsheet_name: str = ""

    # MLflow
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "ticket-classifier"
    mlflow_model_name: str = "ticket-classifier"
    mlflow_s3_endpoint_url: str = ""
    mlflow_bucket_name: str = "mlflow-artifacts"
    mlflow_tracking_username: str = ""
    mlflow_tracking_password: str = ""

    # Gemini (for generating embeddings during training)
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-2.0-flash"

    # Data API
    data_api_url: str = "http://data-api:8002"

    # Service
    host: str = "0.0.0.0"
    port: int = 8005
    debug: bool = False

    @classmethod
    def from_env(cls) -> TrainingServiceConfig:
        _load_env()
        return cls(
            google_service_account_json=Path(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")),
            google_spreadsheet_name=os.getenv("GOOGLE_SPREADSHEET_NAME", ""),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
            mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "ticket-classifier"),
            mlflow_model_name=os.getenv("MLFLOW_MODEL_NAME", "ticket-classifier"),
            mlflow_s3_endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", ""),
            mlflow_bucket_name=os.getenv("MLFLOW_BUCKET_NAME", "mlflow-artifacts"),
            mlflow_tracking_username=os.getenv("MLFLOW_TRACKING_USERNAME", ""),
            mlflow_tracking_password=os.getenv("MLFLOW_TRACKING_PASSWORD", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
            data_api_url=os.getenv("DATA_API_URL", "http://data-api:8002"),
            host=os.getenv("SERVICE_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVICE_PORT", "8005")),
            debug=os.getenv("DEBUG", "false").lower() in ("true", "1"),
        )


def setup_logging(debug: bool = False, service_name: str = "service") -> None:
    """Setup structured JSON logging for a microservice.

    Outputs JSON lines (requires python-json-logger) with fields:
      ts, level, logger, service, trace_id, message
    Falls back to plain-text if the package is not installed.
    """
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler()
    handler.addFilter(_TraceIdFilter())

    try:
        from pythonjsonlogger import jsonlogger  # python-json-logger 2.x / 3.x

        fmt = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(trace_id)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
            static_fields={"service": service_name},
        )
        handler.setFormatter(fmt)
    except ImportError:
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s | %(levelname)-8s | {service_name} | %(name)s | [%(trace_id)s] | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for lib in ("httpx", "httpcore", "telegram", "gspread", "uvicorn.access"):
        logging.getLogger(lib).setLevel(logging.WARNING)
