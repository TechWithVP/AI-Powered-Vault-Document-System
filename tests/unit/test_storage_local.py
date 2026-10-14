"""Filesystem storage backend behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import NotFoundError, StorageError
from app.storage.local import LocalFileSystemStorageBackend


@pytest.fixture
def backend(tmp_path: Path) -> LocalFileSystemStorageBackend:
    return LocalFileSystemStorageBackend(tmp_path / "documents")


class TestRoundTrip:
    def test_saved_bytes_are_returned_unchanged(
        self, backend: LocalFileSystemStorageBackend
    ) -> None:
        payload = b"%PDF-1.7 binary \x00\xff content"
        stored = backend.save("docs/a.pdf", payload, content_type="application/pdf")

        assert stored.size_bytes == len(payload)
        assert backend.load("docs/a.pdf") == payload

    def test_nested_keys_create_directories(
        self, backend: LocalFileSystemStorageBackend
    ) -> None:
        backend.save("a/b/c/d.txt", b"x", content_type="text/plain")
        assert backend.exists("a/b/c/d.txt")

    def test_save_overwrites_an_existing_key(
        self, backend: LocalFileSystemStorageBackend
    ) -> None:
        backend.save("k.txt", b"first", content_type="text/plain")
        backend.save("k.txt", b"second", content_type="text/plain")
        assert backend.load("k.txt") == b"second"

    def test_no_temporary_files_remain_after_save(
        self, backend: LocalFileSystemStorageBackend
    ) -> None:
        backend.save("k.txt", b"data", content_type="text/plain")
        assert list(backend.root.rglob("*.tmp")) == []


class TestMissingKeys:
    def test_load_raises_not_found(
        self, backend: LocalFileSystemStorageBackend
    ) -> None:
        with pytest.raises(NotFoundError):
            backend.load("absent.txt")

    def test_exists_is_false(self, backend: LocalFileSystemStorageBackend) -> None:
        assert backend.exists("absent.txt") is False

    def test_delete_is_idempotent(
        self, backend: LocalFileSystemStorageBackend
    ) -> None:
        backend.delete("absent.txt")  # must not raise


class TestPathTraversal:
    @pytest.mark.parametrize(
        "key", ["../escape.txt", "a/../../escape.txt", "docs/../../../etc/passwd"]
    )
    def test_keys_escaping_the_root_are_refused(
        self, backend: LocalFileSystemStorageBackend, key: str
    ) -> None:
        with pytest.raises(StorageError, match="outside storage root"):
            backend.save(key, b"malicious", content_type="text/plain")


class TestHealthCheck:
    def test_passes_for_an_existing_root(
        self, backend: LocalFileSystemStorageBackend
    ) -> None:
        backend.health_check()

    def test_fails_when_the_root_disappears(
        self, backend: LocalFileSystemStorageBackend
    ) -> None:
        backend.root.rmdir()
        with pytest.raises(StorageError):
            backend.health_check()
