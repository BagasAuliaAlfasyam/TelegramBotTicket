"""Configuration loading utilities for the Telegram bot project."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Automatically load environment variables from a .env file if present.
load_dotenv()


@dataclass(frozen=True)
class Config:
    """Strongly-typed configuration container."""

    telegram_bot_token_collecting: str
    telegram_bot_token_reporting: str
    google_service_account_json: Path
    google_spreadsheet_name: str
    google_worksheet_name: str
    target_group_collecting: Optional[int]
    target_group_reporting: Optional[int]
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_s3_bucket: str
    aws_s3_endpoint: str
    aws_s3_public_base_url: str
    aws_s3_region: Optional[str]
    aws_s3_media_prefix: str
    aws_s3_signature_version: str
    aws_s3_addressing_style: str


def _require_env(name: str) -> str:
    """Read a required environment variable or raise a helpful error."""
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Environment variable '{name}' must be set in .env or the environment."
        )
    return value


def _optional_int(name: str) -> Optional[int]:
    """Convert an optional environment variable into an int, if provided."""
    raw_value = os.getenv(name)
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ValueError(
            f"Environment variable '{name}' must be an integer when provided."
        ) from exc


def load_config() -> Config:
    """Load and validate configuration values from the environment."""
    collecting_token = _require_env("TELEGRAM_BOT_TOKEN_COLLECTING")
    reporting_token = _require_env("TELEGRAM_BOT_TOKEN_REPORTING")
    service_account_path = Path(_require_env("GOOGLE_SERVICE_ACCOUNT_JSON")).expanduser()
    if not service_account_path.is_file():
        raise FileNotFoundError(
            f"Google service account JSON file not found: {service_account_path}"
        )

    spreadsheet_name = _require_env("GOOGLE_SPREADSHEET_NAME")
    worksheet_name = _require_env("GOOGLE_WORKSHEET_NAME")

    target_collecting = _optional_int("TARGET_GROUP_COLLECTING")
    target_reporting = _optional_int("TARGET_GROUP_REPORTING")

    aws_access_key = _require_env("AWS_ACCESS_KEY_ID")
    aws_secret_key = _require_env("AWS_SECRET_ACCESS_KEY")
    aws_bucket = _require_env("AWS_S3_BUCKET")
    aws_endpoint = _ensure_url(_require_env("AWS_S3_ENDPOINT"))
    aws_public_base = os.getenv("AWS_S3_PUBLIC_BASE_URL")
    if aws_public_base:
        aws_public_base = _ensure_url(aws_public_base)
    else:
        aws_public_base = _build_public_base_url(aws_endpoint, aws_bucket)

    aws_region = os.getenv("AWS_S3_REGION")
    media_prefix = os.getenv("AWS_S3_MEDIA_PREFIX", "tech-media/")
    media_prefix = media_prefix.strip().strip("/")
    if media_prefix:
        media_prefix = f"{media_prefix}/"

    signature_version = os.getenv("AWS_S3_SIGNATURE_VERSION", "s3").lower()
    if signature_version not in {"s3", "s3v4"}:
        raise ValueError("AWS_S3_SIGNATURE_VERSION must be either 's3' or 's3v4'.")

    addressing_style = os.getenv("AWS_S3_ADDRESSING_STYLE", "virtual").lower()
    if addressing_style not in {"virtual", "path"}:
        raise ValueError("AWS_S3_ADDRESSING_STYLE must be 'virtual' or 'path'.")

    return Config(
        telegram_bot_token_collecting=collecting_token,
        telegram_bot_token_reporting=reporting_token,
        google_service_account_json=service_account_path,
        google_spreadsheet_name=spreadsheet_name,
        google_worksheet_name=worksheet_name,
        target_group_collecting=target_collecting,
        target_group_reporting=target_reporting,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_s3_bucket=aws_bucket,
        aws_s3_endpoint=aws_endpoint,
        aws_s3_public_base_url=aws_public_base.rstrip("/"),
        aws_s3_region=aws_region,
        aws_s3_media_prefix=media_prefix,
        aws_s3_signature_version=signature_version,
        aws_s3_addressing_style=addressing_style,
    )


def _ensure_url(value: str) -> str:
    """Ensure the provided endpoint/public URL has a scheme."""
    value = value.strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"https://{value}"


def _build_public_base_url(endpoint: str, bucket: str) -> str:
    stripped = endpoint.replace("https://", "").replace("http://", "")
    stripped = stripped.strip("/")
    return f"https://{bucket}.{stripped}"
