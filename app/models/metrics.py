"""Observability models: per-stage pipeline metrics and AI cost tracking."""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import AIOperation, ProcessingStage, StageStatus
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import StrEnumType
from app.models.document import Document


class ProcessingMetric(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One row per pipeline stage execution.

    Stage-level rows (rather than a single per-document timing) make it possible
    to answer "which stage is slow?" and "where do documents fail?" directly
    from the metrics endpoints.
    """

    __tablename__ = "processing_metrics"
    __table_args__ = (
        Index("ix_metrics_stage_status", "stage", "status"),
        Index("ix_metrics_document_stage", "document_id", "stage"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    stage: Mapped[ProcessingStage] = mapped_column(
        StrEnumType(ProcessingStage, 16), nullable=False
    )
    status: Mapped[StageStatus] = mapped_column(
        StrEnumType(StageStatus, 16), nullable=False
    )
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    items_processed: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="metrics")

    @property
    def succeeded(self) -> bool:
        return self.status is StageStatus.SUCCEEDED


class ApiUsage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A single billable AI provider call, recorded for cost reporting."""

    __tablename__ = "api_usage"
    __table_args__ = (
        Index("ix_usage_operation_created", "operation", "created_at"),
    )

    operation: Mapped[AIOperation] = mapped_column(
        StrEnumType(AIOperation, 16), nullable=False
    )
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    endpoint: Mapped[str | None] = mapped_column(String(128))
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL")
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="SET NULL")
    )

    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False, default=Decimal("0")
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer)


class RetrievalMetric(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One row per search, recording which arm the answer came from.

    Hybrid retrieval is two systems whose relative value is an empirical
    question, not a settled one, and it changes with the corpus. Fusion is the
    only point where both signals exist on a comparable scale, so the share is
    computed there and stored here; without it the only way to know whether
    the lexical arm is still earning its place is to re-run the eval harness
    by hand against a snapshot of the corpus.
    """

    __tablename__ = "retrieval_metrics"
    __table_args__ = (
        Index("ix_retrieval_created", "created_at"),
        Index("ix_retrieval_session", "session_id"),
    )

    session_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE")
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE")
    )
    #: DOCUMENT, CONVERSATION or AGGREGATE - counting questions retrieve wider,
    #: so their shares are not comparable with ordinary ones.
    question_kind: Mapped[str] = mapped_column(String(16), nullable=False)

    #: Summed RRF contribution of each arm across the returned chunks.
    semantic_contribution: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    lexical_contribution: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    #: Derived, but stored: recomputing a ratio over millions of rows at read
    #: time is the kind of thing that quietly makes a dashboard unusable.
    semantic_share: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    lexical_share: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    #: How many returned chunks each arm found, and how many both agreed on.
    results_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    semantic_only: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lexical_only: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    found_by_both: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    best_similarity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    best_lexical_rank: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    is_relevant: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reranked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
