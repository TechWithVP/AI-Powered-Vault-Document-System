"""Metrics API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.services.metrics_service import (
    AiUsageStatistics,
    DocumentStatistics,
    ProcessingStatistics,
)


class TagCount(BaseModel):
    tag: str
    count: int


class DailyCount(BaseModel):
    date: str
    count: int


class DocumentTotals(BaseModel):
    """Corpus-wide totals."""

    documents: int
    pages: int
    chunks: int
    storage_bytes: int
    avg_chunks_per_document: float

    @property
    def storage_mb(self) -> float:
        return round(self.storage_bytes / (1024 * 1024), 2)


class DocumentStatisticsResponse(BaseModel):
    """What is in the vault and how it is composed."""

    totals: DocumentTotals
    success_rate: float = Field(
        description="Share of finished documents that processed successfully."
    )
    by_status: dict[str, int]
    by_file_type: dict[str, int]
    by_category: dict[str, int]
    top_tags: list[TagCount]
    uploads_by_day: list[DailyCount]

    @classmethod
    def from_domain(cls, stats: DocumentStatistics) -> "DocumentStatisticsResponse":
        return cls(
            totals=DocumentTotals(**stats.totals),
            success_rate=stats.success_rate,
            by_status=stats.by_status,
            by_file_type=stats.by_file_type,
            by_category=stats.by_category,
            top_tags=[TagCount(**row) for row in stats.top_tags],
            uploads_by_day=[DailyCount(**row) for row in stats.uploads_by_day],
        )


class StageMetrics(BaseModel):
    """Timing and reliability for one pipeline stage."""

    stage: str
    runs: int
    succeeded: int
    failed: int
    avg_ms: float
    p95_ms: float
    max_ms: int


class ProcessingStatisticsResponse(BaseModel):
    """How the ingestion pipeline is performing."""

    overall_success_rate: float
    total_runs: int
    total_failures: int
    slowest_stage: str | None
    stages: list[StageMetrics]

    @classmethod
    def from_domain(
        cls, stats: ProcessingStatistics
    ) -> "ProcessingStatisticsResponse":
        return cls(
            overall_success_rate=stats.overall_success_rate,
            total_runs=stats.total_runs,
            total_failures=stats.total_failures,
            slowest_stage=stats.slowest_stage,
            stages=[StageMetrics(**stage) for stage in stats.stages],
        )


class UsageTotals(BaseModel):
    calls: int
    total_tokens: int
    cost_usd: float


class OperationUsage(BaseModel):
    """Spend attributed to one operation and model."""

    operation: str
    model: str
    calls: int
    total_tokens: int
    cost_usd: float


class AiUsageResponse(BaseModel):
    """AI token consumption and cost."""

    total: UsageTotals
    by_operation: list[OperationUsage]

    @classmethod
    def from_domain(cls, stats: AiUsageStatistics) -> "AiUsageResponse":
        return cls(
            total=UsageTotals(**stats.total),
            by_operation=[OperationUsage(**row) for row in stats.by_operation],
        )
