"""Upload validation rules."""

from __future__ import annotations

import io

import pytest
from fastapi import UploadFile

from app.core.config import IngestionSettings, Settings
from app.core.constants import FileType
from app.core.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.services.document_service import StorageKeyBuilder, UploadValidator


class _StubSettings(Settings):
    """Settings double exposing only the ingestion section."""

    def __init__(self, max_upload_mb: int = 1) -> None:  # noqa: D107 - test double
        self.ingestion = IngestionSettings(max_upload_mb=max_upload_mb)


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content))


@pytest.fixture
def validator() -> UploadValidator:
    return UploadValidator(_StubSettings(max_upload_mb=1))


class TestAcceptedUploads:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("policy.txt", FileType.TXT),
            ("notes.md", FileType.MD),
            ("REPORT.PDF", FileType.PDF),
        ],
    )
    async def test_supported_types_are_accepted(
        self, validator: UploadValidator, filename: str, expected: FileType
    ) -> None:
        result = await validator.validate(make_upload(filename, b"real content here"))
        assert result.file_type is expected
        assert result.filename == filename

    async def test_content_hash_is_stable_for_identical_bytes(
        self, validator: UploadValidator
    ) -> None:
        first = await validator.validate(make_upload("a.txt", b"same bytes"))
        second = await validator.validate(make_upload("b.txt", b"same bytes"))
        assert first.content_hash == second.content_hash

    async def test_content_hash_differs_for_different_bytes(
        self, validator: UploadValidator
    ) -> None:
        first = await validator.validate(make_upload("a.txt", b"one"))
        second = await validator.validate(make_upload("a.txt", b"two"))
        assert first.content_hash != second.content_hash

    async def test_size_reflects_the_payload(
        self, validator: UploadValidator
    ) -> None:
        result = await validator.validate(make_upload("a.txt", b"x" * 500))
        assert result.size_bytes == 500


class TestRejectedUploads:
    async def test_unsupported_extension(self, validator: UploadValidator) -> None:
        with pytest.raises(UnsupportedFileTypeError) as error:
            await validator.validate(make_upload("malware.exe", b"content"))
        assert ".pdf" in error.value.message

    async def test_missing_filename(self, validator: UploadValidator) -> None:
        with pytest.raises(UnsupportedFileTypeError, match="missing a filename"):
            await validator.validate(make_upload("", b"content"))

    async def test_empty_payload(self, validator: UploadValidator) -> None:
        with pytest.raises(EmptyFileError):
            await validator.validate(make_upload("a.txt", b""))

    async def test_whitespace_only_payload(
        self, validator: UploadValidator
    ) -> None:
        with pytest.raises(EmptyFileError):
            await validator.validate(make_upload("a.txt", b"  \n\t  "))

    async def test_payload_over_the_limit(self, validator: UploadValidator) -> None:
        oversized = b"x" * (2 * 1024 * 1024)
        with pytest.raises(FileTooLargeError) as error:
            await validator.validate(make_upload("big.txt", oversized))
        assert error.value.details["limit_bytes"] == 1024 * 1024

    async def test_payload_at_the_limit_is_accepted(self) -> None:
        validator = UploadValidator(_StubSettings(max_upload_mb=1))
        exactly = b"x" * (1024 * 1024)
        result = await validator.validate(make_upload("edge.txt", exactly))
        assert result.size_bytes == 1024 * 1024


class TestStorageKeys:
    def test_key_is_namespaced_and_typed(self) -> None:
        import uuid

        document_id = uuid.uuid4()
        key = StorageKeyBuilder.build("user-key", document_id, FileType.PDF)

        assert key.startswith("documents/")
        assert key.endswith(f"{document_id}.pdf")

    def test_user_identifier_is_not_stored_in_clear_text(self) -> None:
        import uuid

        key = StorageKeyBuilder.build("secret-api-key", uuid.uuid4(), FileType.TXT)
        assert "secret-api-key" not in key

    def test_different_users_get_different_namespaces(self) -> None:
        import uuid

        document_id = uuid.uuid4()
        first = StorageKeyBuilder.build("user-a", document_id, FileType.TXT)
        second = StorageKeyBuilder.build("user-b", document_id, FileType.TXT)
        assert first != second
