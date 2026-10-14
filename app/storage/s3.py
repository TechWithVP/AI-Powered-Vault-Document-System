"""S3-compatible storage backend (AWS S3 in production, MinIO locally)."""

from __future__ import annotations

import threading
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import S3Settings
from app.core.exceptions import NotFoundError, StorageError
from app.core.logging import get_logger
from app.storage.base import StorageBackend, StoredObject

logger = get_logger(__name__)

#: Error codes S3 returns when an object or bucket is absent.
MISSING_CODES = frozenset({"404", "NoSuchKey", "NoSuchBucket", "NotFound"})

#: Codes returned when a bucket create loses a race with another worker.
ALREADY_OWNED_CODES = frozenset({"BucketAlreadyOwnedByYou", "BucketAlreadyExists"})


class S3StorageBackend(StorageBackend):
    """Stores documents in an S3-compatible bucket.

    The boto3 client is created lazily behind a lock so the backend can be
    shared safely across the API thread pool and Celery worker threads.
    """

    def __init__(self, settings: S3Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._lock = threading.Lock()

    @property
    def client(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        settings = self._settings
        return boto3.client(
            "s3",
            region_name=settings.region,
            endpoint_url=settings.endpoint_url,
            aws_access_key_id=settings.access_key_id,
            aws_secret_access_key=settings.secret_access_key,
            use_ssl=settings.use_ssl,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=5,
                read_timeout=30,
            ),
        )

    def ensure_bucket(self) -> None:
        """Create the configured bucket when it does not already exist."""
        bucket = self._settings.bucket
        try:
            self.client.head_bucket(Bucket=bucket)
            return
        except ClientError as exc:
            if self._error_code(exc) not in MISSING_CODES:
                raise StorageError(f"Cannot access bucket {bucket}: {exc}") from exc
        except BotoCoreError as exc:
            raise StorageError(f"Cannot reach object storage: {exc}") from exc

        try:
            self.client.create_bucket(Bucket=bucket)
            logger.info("Created storage bucket", extra={"bucket": bucket})
        except ClientError as exc:
            if self._error_code(exc) not in ALREADY_OWNED_CODES:
                raise StorageError(f"Cannot create bucket {bucket}: {exc}") from exc

    def save(self, key: str, data: bytes, *, content_type: str) -> StoredObject:
        try:
            self.client.put_object(
                Bucket=self._settings.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"Failed to store {key}: {exc}") from exc

        logger.info(
            "Stored document",
            extra={"storage_key": key, "size_bytes": len(data)},
        )
        return StoredObject(key=key, size_bytes=len(data), content_type=content_type)

    def load(self, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self._settings.bucket, Key=key)
            return response["Body"].read()
        except ClientError as exc:
            if self._error_code(exc) in MISSING_CODES:
                raise NotFoundError.for_resource("Stored file", key) from exc
            raise StorageError(f"Failed to read {key}: {exc}") from exc
        except BotoCoreError as exc:
            raise StorageError(f"Failed to read {key}: {exc}") from exc

    def delete(self, key: str) -> None:
        try:
            self.client.delete_object(Bucket=self._settings.bucket, Key=key)
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"Failed to delete {key}: {exc}") from exc

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self._settings.bucket, Key=key)
            return True
        except ClientError as exc:
            if self._error_code(exc) in MISSING_CODES:
                return False
            raise StorageError(f"Failed to stat {key}: {exc}") from exc
        except BotoCoreError as exc:
            raise StorageError(f"Failed to stat {key}: {exc}") from exc

    def presigned_url(self, key: str, *, expires_in: int | None = None) -> str:
        ttl = expires_in or self._settings.presigned_url_ttl_seconds
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._settings.bucket, "Key": key},
                ExpiresIn=ttl,
            )
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"Failed to sign URL for {key}: {exc}") from exc

    def health_check(self) -> None:
        try:
            self.client.head_bucket(Bucket=self._settings.bucket)
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"Object storage unhealthy: {exc}") from exc

    @staticmethod
    def _error_code(exc: ClientError) -> str:
        return str(exc.response.get("Error", {}).get("Code", ""))
