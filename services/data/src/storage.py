"""
S3 Storage Client (microservice version)
==========================================
Upload media to S3/MinIO.
"""
from __future__ import annotations

import logging
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from services.shared.config import DataServiceConfig

_LOGGER = logging.getLogger(__name__)


class S3Uploader:
    def __init__(self, config: DataServiceConfig):
        self._bucket = config.s3_bucket_name
        _endpoint = config.s3_endpoint_url or None
        
        if config.s3_public_url:
            self._public_base_url = config.s3_public_url.rstrip("/")
        elif _endpoint:
            self._public_base_url = f"{_endpoint.rstrip('/')}/{self._bucket}"
        else:
            self._public_base_url = f"https://{self._bucket}.s3.amazonaws.com"
        
        self._media_prefix = config.s3_media_prefix

        boto_config = BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path", "payload_signing_enabled": False},
        )

        self._client = boto3.client(
            "s3",
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            endpoint_url=_endpoint,
            region_name=config.s3_region,
            config=boto_config,
        )

    def upload_bytes(self, *, key: str, data: bytes, content_type: str) -> str:
        object_key = f"{self._media_prefix}/{key.lstrip('/')}" if self._media_prefix else key.lstrip("/")
        self._client.put_object(
            Bucket=self._bucket, Key=object_key, Body=data,
            ContentType=content_type, ACL="public-read",
        )
        return f"{self._public_base_url}/{object_key}"
