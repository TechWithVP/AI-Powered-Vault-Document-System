"""Object storage package.

The concrete backend is selected from configuration, so switching between the
local filesystem (development) and S3 (EC2 / AWS) never requires a code change.
"""

from functools import lru_cache

from app.core.config import StorageBackendType, get_settings
from app.core.logging import get_logger
from app.storage.base import StorageBackend, StoredObject
from app.storage.local import LocalFileSystemStorageBackend
from app.storage.s3 import S3StorageBackend

logger = get_logger(__name__)

__all__ = [
    "LocalFileSystemStorageBackend",
    "S3StorageBackend",
    "StorageBackend",
    "StoredObject",
    "build_storage_backend",
    "get_storage_backend",
]


def build_storage_backend() -> StorageBackend:
    """Construct the backend named by configuration."""
    settings = get_settings()
    if settings.storage.backend is StorageBackendType.S3:
        backend = S3StorageBackend(settings.s3)
        backend.ensure_bucket()
        logger.info(
            "Using S3 storage backend",
            extra={"bucket": settings.s3.bucket, "endpoint": settings.s3.endpoint_url},
        )
        return backend

    backend = LocalFileSystemStorageBackend(settings.storage.local_path)
    logger.info(
        "Using local filesystem storage backend", extra={"root": str(backend.root)}
    )
    return backend


@lru_cache(maxsize=1)
def get_storage_backend() -> StorageBackend:
    """Return the process-wide storage backend."""
    return build_storage_backend()
