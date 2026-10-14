"""Document API schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import (
    DocumentCategory,
    DocumentStatus,
    FileType,
    Sentiment,
)
from app.models.document import Document, DocumentInsight


class DocumentSummary(BaseModel):
    """Document metadata and processing state."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: FileType
    file_size_bytes: int
    status: DocumentStatus
    page_count: int | None = None
    chunk_count: int | None = None
    character_count: int | None = None
    failure_code: str | None = Field(
        default=None, description="Stable code when status is failed."
    )
    failure_reason: str | None = Field(
        default=None, description="Human-readable explanation of a failure."
    )
    created_at: datetime
    processed_at: datetime | None = None

    @classmethod
    def from_model(cls, document: Document) -> "DocumentSummary":
        return cls.model_validate(document)


class UploadAccepted(BaseModel):
    """Returned by the upload endpoint once a document is queued."""

    id: uuid.UUID
    filename: str
    status: DocumentStatus
    message: str = Field(
        default=(
            "Document accepted and queued for processing. Poll this document "
            "until its status is ready."
        )
    )

    @classmethod
    def from_model(cls, document: Document) -> "UploadAccepted":
        return cls(
            id=document.id, filename=document.filename, status=document.status
        )


class EntityRead(BaseModel):
    """A named entity extracted from the document."""

    name: str
    type: str


class InsightResponse(BaseModel):
    """AI-generated analysis of a document."""

    model_config = ConfigDict(from_attributes=True)

    document_id: uuid.UUID
    summary: str
    key_points: list[str]
    tags: list[str]
    entities: list[dict[str, Any]]
    category: DocumentCategory
    sentiment: Sentiment
    language: str
    model: str
    was_truncated: bool
    token_usage: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_model(cls, insight: DocumentInsight) -> "InsightResponse":
        return cls.model_validate(insight)


class DocumentDetail(DocumentSummary):
    """Document metadata plus its insights, when analysis has completed."""

    insight: InsightResponse | None = None

    @classmethod
    def from_model(cls, document: Document) -> "DocumentDetail":
        detail = cls.model_validate(document)
        if document.insight is not None:
            detail.insight = InsightResponse.from_model(document.insight)
        return detail
