"""Document management endpoints."""

from __future__ import annotations

import uuid
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile

from app.api.deps import CurrentPrincipal, DocumentServiceDep, Pagination
from app.core.constants import DocumentCategory, DocumentStatus
from app.core.exceptions import NotFoundError
from app.core.logging import CorrelationId, get_logger
from app.repositories.document import DocumentFilter
from app.schemas.common import Page
from app.schemas.document import (
    DocumentDetail,
    DocumentSummary,
    InsightResponse,
    UploadAccepted,
)
from app.workers.tasks import process_document, reprocess_document

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    status_code=HTTPStatus.ACCEPTED,
    response_model=UploadAccepted,
    summary="Upload a document",
    description=(
        "Accepts a PDF, DOCX, TXT or Markdown file and queues it for "
        "processing. Returns immediately with a document id; poll "
        "GET /documents/{id} until the status is ready. Re-uploading identical "
        "content returns the existing document instead of duplicating it."
    ),
)
async def upload_document(
    principal: CurrentPrincipal,
    service: DocumentServiceDep,
    file: Annotated[UploadFile, File(description="The document to upload.")],
) -> UploadAccepted:
    document = await service.upload(file, principal)

    if document.status is DocumentStatus.QUEUED:
        # Queued after the response is committed by the session dependency, so
        # the worker can never observe a row that was later rolled back.
        process_document.apply_async(
            args=[str(document.id), CorrelationId.get()],
            countdown=1,
        )

    return UploadAccepted.from_model(document)


@router.get(
    "",
    response_model=Page[DocumentSummary],
    summary="List documents",
    description=(
        "Returns the caller's documents, newest first. Supports filtering by "
        "status, AI-assigned category or tag, and a text search across "
        "filenames and summaries."
    ),
)
async def list_documents(
    principal: CurrentPrincipal,
    service: DocumentServiceDep,
    params: Pagination,
    status: Annotated[DocumentStatus | None, Query()] = None,
    category: Annotated[DocumentCategory | None, Query()] = None,
    tag: Annotated[str | None, Query(max_length=64)] = None,
    search: Annotated[str | None, Query(max_length=200)] = None,
) -> Page[DocumentSummary]:
    criteria = DocumentFilter(
        user_id=principal.user_id,
        status=status,
        category=category,
        tag=tag,
        search=search,
    )
    documents, total = await service.list(criteria, params)
    return Page.build(
        [DocumentSummary.from_model(document) for document in documents],
        total=total,
        params=params,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    summary="Get a document",
    description=(
        "Returns document metadata, processing status and - once analysis has "
        "finished - the AI-generated insights."
    ),
)
async def get_document(
    document_id: uuid.UUID,
    principal: CurrentPrincipal,
    service: DocumentServiceDep,
) -> DocumentDetail:
    document = await service.get(document_id, principal)
    return DocumentDetail.from_model(document)


@router.get(
    "/{document_id}/insights",
    response_model=InsightResponse,
    summary="Get AI insights for a document",
    description=(
        "Returns the summary, key points, tags, entities, category and "
        "sentiment produced during processing."
    ),
)
async def get_document_insights(
    document_id: uuid.UUID,
    principal: CurrentPrincipal,
    service: DocumentServiceDep,
) -> InsightResponse:
    document = await service.get(document_id, principal)
    if document.insight is None:
        raise NotFoundError(
            "Insights are not available for this document yet.",
            details={
                "document_id": str(document_id),
                "status": document.status.value,
            },
        )
    return InsightResponse.from_model(document.insight)


@router.post(
    "/{document_id}/reprocess",
    status_code=HTTPStatus.ACCEPTED,
    response_model=UploadAccepted,
    summary="Reprocess a document",
    description="Requeues a document, typically after a transient failure.",
)
async def reprocess(
    document_id: uuid.UUID,
    principal: CurrentPrincipal,
    service: DocumentServiceDep,
) -> UploadAccepted:
    document = await service.get(document_id, principal)
    reprocess_document.delay(str(document.id))
    return UploadAccepted(
        id=document.id,
        filename=document.filename,
        status=DocumentStatus.QUEUED,
        message="Document requeued for processing.",
    )


@router.delete(
    "/{document_id}",
    status_code=HTTPStatus.NO_CONTENT,
    # Explicit None: FastAPI would otherwise infer NoneType as a response
    # model, which is rejected for a 204.
    response_model=None,
    summary="Delete a document",
    description=(
        "Permanently removes the document, its chunks, insights and the stored "
        "file. Chat sessions referencing it are updated automatically."
    ),
)
async def delete_document(
    document_id: uuid.UUID,
    principal: CurrentPrincipal,
    service: DocumentServiceDep,
) -> None:
    await service.delete(document_id, principal)
