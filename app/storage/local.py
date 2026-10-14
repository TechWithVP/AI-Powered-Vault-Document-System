"""Filesystem storage backend.

Used for local development and tests so the stack needs no object-storage
container. The S3 backend is the production path; both satisfy the same
StorageBackend interface, so the ingestion pipeline is identical either way.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.core.exceptions import NotFoundError, StorageError
from app.core.logging import get_logger
from app.storage.base import StorageBackend, StoredObject

logger = get_logger(__name__)


class LocalFileSystemStorageBackend(StorageBackend):
    """Persists documents under a root directory on the local filesystem."""

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, key: str) -> Path:
        """Map a storage key to a path, rejecting traversal outside the root."""
        candidate = (self._root / key).resolve()
        if not candidate.is_relative_to(self._root):
            raise StorageError(f"Refusing to access path outside storage root: {key}")
        return candidate

    def save(self, key: str, data: bytes, *, content_type: str) -> StoredObject:
        path = self._resolve(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temporary file then rename so a crash mid-write cannot
            # leave a partially written document behind.
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        except OSError as exc:
            raise StorageError(f"Failed to store {key}: {exc}") from exc

        logger.info(
            "Stored document",
            extra={"storage_key": key, "size_bytes": len(data)},
        )
        return StoredObject(key=key, size_bytes=len(data), content_type=content_type)

    def load(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise NotFoundError.for_resource("Stored file", key) from exc
        except OSError as exc:
            raise StorageError(f"Failed to read {key}: {exc}") from exc

    def delete(self, key: str) -> None:
        try:
            self._resolve(key).unlink(missing_ok=True)
        except OSError as exc:
            raise StorageError(f"Failed to delete {key}: {exc}") from exc

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def presigned_url(self, key: str, *, expires_in: int | None = None) -> str:
        """Return a file URI.

        Time-limited URLs are an object-store feature; in local mode the API
        streams downloads itself rather than delegating to storage.
        """
        return self._resolve(key).as_uri()

    def health_check(self) -> None:
        if not self._root.is_dir():
            raise StorageError(f"Storage root does not exist: {self._root}")
        usage = shutil.disk_usage(self._root)
        if usage.free <= 0:
            raise StorageError("No free disk space in storage root")
