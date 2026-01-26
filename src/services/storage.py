"""
S3 Storage Service
===================

Utility helper untuk upload media ke S3-compatible storage.

Mendukung:
    - AWS S3 standar
    - MinIO (self-hosted)
    - Storage S3-compatible lainnya

File yang diupload akan mendapat URL publik yang bisa diakses
langsung tanpa autentikasi.

Author: Bagas Aulia Alfasyam
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
    """
    Wrapper tipis untuk boto3 untuk upload media dan generate public URL.
    
    Class ini menangani konfigurasi boto3 client termasuk:
        - Path-style addressing untuk kompatibilitas MinIO
        - Signature v4 untuk keamanan
        - ACL public-read untuk akses publik
    
    Attributes:
        _bucket: Nama bucket S3
        _public_base_url: Base URL untuk akses publik
        _media_prefix: Prefix path untuk file yang diupload
        _client: boto3 S3 client
    """

    def __init__(
        self, 
        config: "Config",
        bucket: str = None,
        public_base_url: str = None,
        media_prefix: str = "",
        endpoint_url: str = None,
        addressing_style: str = "path",  # "path" for MinIO compatibility
        signature_version: str = "s3v4",
    ) -> None:
        self._bucket = bucket or config.s3_bucket_name
        
        # Determine endpoint and public URL
        _endpoint = endpoint_url or config.s3_endpoint_url or None
        
        # For MinIO/custom S3, use path-style addressing
        if _endpoint:
            addressing_style = "path"
        
        # Public URL for accessing files
        if public_base_url:
            self._public_base_url = public_base_url.rstrip("/")
        elif config.s3_public_url:
            self._public_base_url = config.s3_public_url.rstrip("/")
        elif _endpoint:
            self._public_base_url = f"{_endpoint.rstrip('/')}/{self._bucket}"
        else:
            self._public_base_url = f"https://{self._bucket}.s3.amazonaws.com"
        
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
            endpoint_url=_endpoint,
            region_name=config.s3_region,
            config=boto_config,
        )

    def upload_bytes(self, *, key: str, data: bytes, content_type: str) -> str:
        """
        Upload raw bytes ke S3 dengan key yang diberikan.
        
        File akan diupload dengan ACL public-read sehingga bisa
        diakses publik tanpa autentikasi.
        
        Args:
            key: Object key (nama file) di S3
            data: Bytes data yang akan diupload
            content_type: MIME type file (contoh: "image/jpeg")
        
        Returns:
            str: URL publik untuk mengakses file
        
        Raises:
            ClientError: Jika upload gagal (permission, network, dll)
            BotoCoreError: Jika ada error boto3
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
        """
        Build full object key dengan prefix.
        
        Args:
            key: Object key tanpa prefix
            
        Returns:
            str: Object key lengkap dengan prefix
        """
        trimmed_key = key.lstrip("/")
        if self._media_prefix:
            return f"{self._media_prefix}{trimmed_key}"
        return trimmed_key
