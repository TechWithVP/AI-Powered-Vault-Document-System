"""Observability models: per-stage pipeline metrics and AI cost tracking."""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text
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
