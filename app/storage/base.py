"""Storage abstraction for original document files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Metadata describing a successfully stored file."""

    key: str
    size_bytes: int
    content_type: str


class StorageBackend(ABC):
    """Interface every storage implementation must satisfy.

    Keeping persistence behind this port means the ingestion pipeline is
    unaware of whether bytes live in MinIO, AWS S3 or anything added later.
    """

    @abstractmethod
    def save(self, key: str, data: bytes, *, content_type: str) -> StoredObject:
        """Persist the data under the given key and return its metadata."""

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Return the bytes stored under the given key."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove the key. Deleting a missing key is not an error."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether the key is present."""

    @abstractmethod
    def presigned_url(self, key: str, *, expires_in: int | None = None) -> str:
        """Return a time-limited download URL for the key."""

    @abstractmethod
    def health_check(self) -> None:
        """Raise a StorageError if the backend is unreachable."""
