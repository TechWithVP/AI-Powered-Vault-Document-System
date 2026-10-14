"""DocumentService against a real database.

Covers the upload path end to end: validation, storage, deduplication and the
ownership rules that gate chat sessions.
"""

from __future__ import annotations

import io
import uuid

import pytest
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.constants import DocumentStatus
from app.core.exceptions import (
    DocumentNotReadyError,
    EmptyFileError,
    NotFoundError,
    UnsupportedFileTypeError,
)
from app.core.security import Principal
from app.services.document_service import DocumentService
from tests.integration.conftest import make_document
from tests.integration.test_ingestion import InMemoryStorage

pytestmark = pytest.mark.integration


def upload_file(name: str, content: bytes) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(content))


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def service(db: AsyncSession, storage: InMemoryStorage) -> DocumentService:
    return DocumentService(db, settings=get_settings(), storage=storage)


@pytest.fixture
def principal(user_id: str) -> Principal:
    return Principal(user_id=user_id)


class TestUpload:
    async def test_an_accepted_upload_is_queued_and_stored(
        self,
        service: DocumentService,
        principal: Principal,
        storage: InMemoryStorage,
    ) -> None:
        document = await service.upload(
            upload_file("policy.txt", b"Policy content here."), principal
        )

        assert document.status is DocumentStatus.QUEUED
        assert document.user_id == principal.user_id
        assert storage.exists(document.storage_key)

    async def test_the_stored_bytes_match_the_upload(
        self,
        service: DocumentService,
        principal: Principal,
        storage: InMemoryStorage,
    ) -> None:
        payload = b"Exact bytes to round-trip."
        document = await service.upload(upload_file("a.txt", payload), principal)

        assert storage.load(document.storage_key) == payload

    async def test_the_storage_key_does_not_leak_the_api_key(
        self, service: DocumentService, principal: Principal
    ) -> None:
        document = await service.upload(upload_file("a.txt", b"content"), principal)
        assert principal.user_id not in document.storage_key

    async def test_an_unsupported_type_is_rejected_before_storing(
        self,
        service: DocumentService,
        principal: Principal,
        storage: InMemoryStorage,
    ) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            await service.upload(upload_file("virus.exe", b"content"), principal)

        assert storage.files == {}

    async def test_an_empty_file_is_rejected(
        self, service: DocumentService, principal: Principal
    ) -> None:
        with pytest.raises(EmptyFileError):
            await service.upload(upload_file("empty.txt", b"   "), principal)


class TestDeduplication:
    async def test_identical_content_returns_the_same_document(
        self, service: DocumentService, principal: Principal
    ) -> None:
        payload = b"Exactly the same content."
        first = await service.upload(upload_file("a.txt", payload), principal)
        second = await service.upload(upload_file("renamed.txt", payload), principal)

        assert first.id == second.id

    async def test_different_content_creates_separate_documents(
        self, service: DocumentService, principal: Principal
    ) -> None:
        first = await service.upload(upload_file("a.txt", b"one"), principal)
        second = await service.upload(upload_file("b.txt", b"two"), principal)

        assert first.id != second.id


class TestRetrieval:
    async def test_a_document_is_returned_to_its_owner(
        self, service: DocumentService, principal: Principal
    ) -> None:
        created = await service.upload(upload_file("a.txt", b"content"), principal)

        found = await service.get(created.id, principal)
        assert found.id == created.id

    async def test_another_users_document_is_not_found(
        self, service: DocumentService, principal: Principal
    ) -> None:
        created = await service.upload(upload_file("a.txt", b"content"), principal)

        with pytest.raises(NotFoundError):
            await service.get(created.id, Principal(user_id="someone-else"))

    async def test_an_unknown_id_is_not_found(
        self, service: DocumentService, principal: Principal
    ) -> None:
        with pytest.raises(NotFoundError):
            await service.get(uuid.uuid4(), principal)


class TestChatScoping:
    async def test_ready_documents_are_resolved(
        self, db: AsyncSession, service: DocumentService, principal: Principal
    ) -> None:
        document = make_document(principal.user_id, status=DocumentStatus.READY)
        db.add(document)
        await db.flush()

        resolved = await service.get_ready_documents([document.id], principal)
        assert [item.id for item in resolved] == [document.id]

    async def test_a_document_still_processing_is_rejected(
        self, db: AsyncSession, service: DocumentService, principal: Principal
    ) -> None:
        document = make_document(principal.user_id, status=DocumentStatus.PROCESSING)
        db.add(document)
        await db.flush()

        with pytest.raises(DocumentNotReadyError):
            await service.get_ready_documents([document.id], principal)

    async def test_an_unknown_document_is_reported_by_id(
        self, service: DocumentService, principal: Principal
    ) -> None:
        missing = uuid.uuid4()

        with pytest.raises(NotFoundError) as error:
            await service.get_ready_documents([missing], principal)

        assert str(missing) in error.value.details["document_ids"]

    async def test_vault_wide_scope_returns_only_ready_documents(
        self, db: AsyncSession, service: DocumentService, principal: Principal
    ) -> None:
        db.add(make_document(principal.user_id, status=DocumentStatus.READY))
        db.add(
            make_document(
                principal.user_id, filename="pending.pdf", status=DocumentStatus.QUEUED
            )
        )
        await db.flush()

        ready = await service.list_all_ready(principal)
        assert [document.status for document in ready] == [DocumentStatus.READY]


class TestDeletion:
    async def test_deleting_removes_the_row_and_the_file(
        self,
        service: DocumentService,
        principal: Principal,
        storage: InMemoryStorage,
    ) -> None:
        document = await service.upload(upload_file("a.txt", b"content"), principal)
        key = document.storage_key

        await service.delete(document.id, principal)

        assert not storage.exists(key)
        with pytest.raises(NotFoundError):
            await service.get(document.id, principal)

    async def test_another_user_cannot_delete_it(
        self, service: DocumentService, principal: Principal
    ) -> None:
        document = await service.upload(upload_file("a.txt", b"content"), principal)

        with pytest.raises(NotFoundError):
            await service.delete(document.id, Principal(user_id="attacker"))
