"""
S3 Storage Service
===================
Utility helpers for uploading media bytes to S3 compatible storage.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

if TYPE_CHECKING:
    from src.core.config import Config

_LOGGER = logging.getLogger(__name__)


class S3Uploader:
    """Thin wrapper around boto3 to upload media objects and return public URLs."""

    def __init__(
        self, 
        config: "Config",
        bucket: str = None,
        public_base_url: str = None,
        media_prefix: str = "",
        endpoint_url: str = None,
        addressing_style: str = "virtual",
        signature_version: str = "s3v4",
    ) -> None:
        self._bucket = bucket or config.s3_bucket_name
        self._public_base_url = (public_base_url or f"https://{self._bucket}.s3.amazonaws.com").rstrip("/")
        self._media_prefix = media_prefix

        s3_config = {"addressing_style": addressing_style}
        if signature_version == "s3v4":
            # Prefer unsigned payloads to avoid SHA mismatch errors when supported.
            s3_config["payload_signing_enabled"] = False

        boto_config = BotoConfig(
            signature_version=signature_version,
            s3=s3_config,
        )

        self._client = boto3.client(
            "s3",
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            endpoint_url=endpoint_url,
            region_name=config.s3_region,
            config=boto_config,
        )

    def upload_bytes(self, *, key: str, data: bytes, content_type: str) -> str:
        """
        Upload raw bytes under the provided key.

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
        """Build full object key with prefix."""
        trimmed_key = key.lstrip("/")
        if self._media_prefix:
            return f"{self._media_prefix}{trimmed_key}"
        return trimmed_key
