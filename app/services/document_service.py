"""API-side document operations: upload, retrieval and deletion."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.constants import DocumentStatus, FileType
from app.core.exceptions import (
    DocumentNotReadyError,
    EmptyFileError,
    FileTooLargeError,
    NotFoundError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.core.security import Principal
from app.models.document import Document
from app.repositories.document import DocumentFilter, DocumentRepository
from app.schemas.common import PageParams
from app.storage.base import StorageBackend

logger = get_logger(__name__)

#: Bytes read per iteration while enforcing the upload size limit.
READ_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    """An upload that passed validation and is ready to be stored."""

    filename: str
    file_type: FileType
    data: bytes
    content_hash: str

    @property
    def size_bytes(self) -> int:
        return len(self.data)


class UploadValidator:
    """Validates an incoming file before anything is persisted.

    Reading is bounded so an oversized upload is rejected without buffering the
    entire payload in memory.
    """

    def __init__(self, settings: Settings) -> None:
        self._max_bytes = settings.ingestion.max_upload_bytes

    async def validate(self, upload: UploadFile) -> ValidatedUpload:
        filename = (upload.filename or "").strip()
        if not filename:
            raise UnsupportedFileTypeError(
                "The upload is missing a filename, so its type cannot be determined."
            )

        file_type = FileType.from_filename(filename)
        if file_type is None:
            raise UnsupportedFileTypeError.for_filename(
                filename, FileType.supported_extensions()
            )

        data = await self._read_bounded(upload, filename)
        if not data.strip():
            raise EmptyFileError(f"The file {filename} contains no data.")

        return ValidatedUpload(
            filename=filename,
            file_type=file_type,
            data=data,
            content_hash=hashlib.sha256(data).hexdigest(),
        )

    async def _read_bounded(self, upload: UploadFile, filename: str) -> bytes:
        buffer = bytearray()
        while piece := await upload.read(READ_CHUNK_BYTES):
            buffer.extend(piece)
            if len(buffer) > self._max_bytes:
                raise FileTooLargeError.for_size(len(buffer), self._max_bytes)
        logger.debug(
            "Upload read", extra={"filename": filename, "size_bytes": len(buffer)}
        )
        return bytes(buffer)


class StorageKeyBuilder:
    """Builds collision-free, tenant-scoped storage keys."""

    @staticmethod
    def build(user_id: str, document_id: uuid.UUID, file_type: FileType) -> str:
        # The user id is a credential, so only a short digest of it is used.
        tenant = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return f"documents/{tenant}/{document_id}.{file_type.value}"


class DocumentService:
    """Coordinates validation, storage, persistence and queueing."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        settings: Settings,
        storage: StorageBackend,
    ) -> None:
        self._session = session
        self._settings = settings
        self._storage = storage
        self._documents = DocumentRepository(session)
        self._validator = UploadValidator(settings)

    async def upload(self, upload: UploadFile, principal: Principal) -> Document:
        """Validate, store and register a document ready for ingestion.

        The file is written to storage before the row is flushed, so a committed
        document always has retrievable bytes behind it.
        """
        validated = await self._validator.validate(upload)

        existing = await self._documents.find_by_content_hash(
            validated.content_hash, principal.user_id
        )
        if existing is not None:
            logger.info(
                "Duplicate upload resolved to existing document",
                extra={
                    "document_id": str(existing.id),
                    "filename": validated.filename,
                },
            )
            return existing

        document_id = uuid.uuid4()
        storage_key = StorageKeyBuilder.build(
            principal.user_id, document_id, validated.file_type
        )
        self._storage.save(
            storage_key,
            validated.data,
            content_type=validated.file_type.media_type,
        )

        document = Document(
            id=document_id,
            user_id=principal.user_id,
            filename=validated.filename,
            file_type=validated.file_type,
            file_size_bytes=validated.size_bytes,
            content_hash=validated.content_hash,
            storage_key=storage_key,
            status=DocumentStatus.QUEUED,
        )
        self._documents.add(document)
        await self._documents.flush()

        logger.info(
            "Document registered",
            extra={
                "document_id": str(document.id),
                "filename": document.filename,
                "size_bytes": document.file_size_bytes,
            },
        )
        return document

    async def get(self, document_id: uuid.UUID, principal: Principal) -> Document:
        document = await self._documents.get_for_user(document_id, principal.user_id)
        if document is None:
            raise NotFoundError.for_resource("Document", document_id)
        return document

    async def list(
        self, criteria: DocumentFilter, params: PageParams
    ) -> tuple[list[Document], int]:
        return await self._documents.list_page(
            criteria, limit=params.limit, offset=params.offset
        )

    async def get_ready_documents(
        self, document_ids: list[uuid.UUID], principal: Principal
    ) -> list[Document]:
        """Resolve documents for a chat session, rejecting any not yet ready."""
        documents = await self._documents.get_many_for_user(
            document_ids, principal.user_id
        )

        found = {document.id for document in documents}
        missing = [str(i) for i in document_ids if i not in found]
        if missing:
            raise NotFoundError(
                "These documents do not exist: " + ", ".join(missing),
                details={"document_ids": missing},
            )

        not_ready = [str(d.id) for d in documents if not d.is_ready]
        if not_ready:
            raise DocumentNotReadyError.for_documents(not_ready)

        return documents

    async def list_all_ready(self, principal: Principal) -> list[Document]:
        """Every ready document for the user, used for vault-wide chat."""
        criteria = DocumentFilter(
            user_id=principal.user_id, status=DocumentStatus.READY
        )
        return await self._documents.list(self._documents.build_query(criteria))

    async def delete(self, document_id: uuid.UUID, principal: Principal) -> None:
        """Remove a document, its derived rows and the stored file."""
        document = await self.get(document_id, principal)
        storage_key = document.storage_key

        await self._documents.delete(document)
        await self._session.flush()

        # Storage is cleaned up after the row is gone: an orphaned object is
        # harmless, whereas a row pointing at deleted bytes is not.
        self._storage.delete(storage_key)
        logger.info("Document deleted", extra={"document_id": str(document_id)})
