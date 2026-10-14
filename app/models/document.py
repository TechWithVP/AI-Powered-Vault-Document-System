"""Document and AI-insight ORM models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import DocumentCategory, DocumentStatus, FileType, Sentiment
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import StrEnumType

if TYPE_CHECKING:
    from app.models.chunk import DocumentChunk
    from app.models.metrics import ProcessingMetric


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An uploaded document and its processing state."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_user_status", "user_id", "status"),
        Index("ix_documents_user_created", "user_id", "created_at"),
    )

    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # -- source file -------------------------------------------------------- #
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[FileType] = mapped_column(
        StrEnumType(FileType, 16), nullable=False
    )
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)

    # -- processing state --------------------------------------------------- #
    status: Mapped[DocumentStatus] = mapped_column(
        StrEnumType(DocumentStatus, 16),
        nullable=False,
        default=DocumentStatus.QUEUED,
        server_default=DocumentStatus.QUEUED.value,
        index=True,
    )
    failure_reason: Mapped[str | None] = mapped_column(Text)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    character_count: Mapped[int | None] = mapped_column(Integer)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # -- relationships ------------------------------------------------------ #
    insight: Mapped["DocumentInsight | None"] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    metrics: Mapped[list["ProcessingMetric"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # -- domain behaviour --------------------------------------------------- #
    @property
    def is_ready(self) -> bool:
        return self.status is DocumentStatus.READY

    def mark_processing(self) -> None:
        self.status = DocumentStatus.PROCESSING
        self.failure_reason = None
        self.failure_code = None

    def mark_ready(self) -> None:
        self.status = DocumentStatus.READY
        self.failure_reason = None
        self.failure_code = None
        self.processed_at = datetime.now(tz=UTC)

    def mark_failed(self, *, code: str, reason: str) -> None:
        self.status = DocumentStatus.FAILED
        self.failure_code = code
        self.failure_reason = reason
        self.processed_at = datetime.now(tz=UTC)


class DocumentInsight(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """AI-generated analysis of a document (one row per document)."""

    __tablename__ = "document_insights"

    document_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    summary: Mapped[str] = mapped_column(Text, nullable=False)
    key_points: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    tags: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    entities: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    category: Mapped[DocumentCategory] = mapped_column(
        StrEnumType(DocumentCategory, 32),
        nullable=False,
        default=DocumentCategory.OTHER,
    )
    sentiment: Mapped[Sentiment] = mapped_column(
        StrEnumType(Sentiment, 16), nullable=False, default=Sentiment.NEUTRAL
    )
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="en")

    # -- provenance --------------------------------------------------------- #
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    token_usage: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    was_truncated: Mapped[bool] = mapped_column(
        nullable=False, default=False, server_default="false"
    )

    document: Mapped[Document] = relationship(back_populates="insight")
