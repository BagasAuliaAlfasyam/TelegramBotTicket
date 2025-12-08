"""Utility helpers for uploading media bytes to S3 compatible storage."""
from __future__ import annotations

import logging

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from config import Config

_LOGGER = logging.getLogger(__name__)


class S3Uploader:
    """Thin wrapper around boto3 to upload media objects and return public URLs."""

    def __init__(self, config: Config) -> None:
        self._bucket = config.aws_s3_bucket
        self._public_base_url = config.aws_s3_public_base_url.rstrip("/")
        self._media_prefix = config.aws_s3_media_prefix

        s3_config = {"addressing_style": config.aws_s3_addressing_style}
        if config.aws_s3_signature_version == "s3v4":
            # Prefer unsigned payloads to avoid SHA mismatch errors when supported.
            s3_config["payload_signing_enabled"] = False

        boto_config = BotoConfig(
            signature_version=config.aws_s3_signature_version,
            s3=s3_config,
        )

        self._client = boto3.client(
            "s3",
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            endpoint_url=config.aws_s3_endpoint,
            region_name=config.aws_s3_region,
            config=boto_config,
        )

    def upload_bytes(self, *, key: str, data: bytes, content_type: str) -> str:
        """Upload raw bytes under the provided key.

        Returns the publicly accessible URL if the upload succeeds.
        """

        object_key = self._build_object_key(key)
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=object_key,
                Body=data,
                ContentType=content_type,
                ACL="public-read",
            )
        except (ClientError, BotoCoreError) as exc:
            _LOGGER.exception("Failed to upload media to S3: %s", exc)
            raise

        return f"{self._public_base_url}/{object_key}"

    def _build_object_key(self, key: str) -> str:
        trimmed_key = key.lstrip("/")
        if self._media_prefix:
            return f"{self._media_prefix}{trimmed_key}"
        return trimmed_key
