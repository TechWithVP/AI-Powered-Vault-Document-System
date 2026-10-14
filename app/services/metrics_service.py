"""Monitoring data for the metrics endpoints.

Each method answers one operational question - what is in the vault, how is
processing performing, what is the AI spend - by aggregating in the database
rather than loading rows into Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import DocumentStatus, ProcessingStage
from app.core.logging import get_logger
from app.core.security import Principal
from app.repositories.document import DocumentStatsRepository
from app.repositories.metrics import ApiUsageRepository, ProcessingMetricRepository

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DocumentStatistics:
    """Composition and size of the user's document corpus."""

    totals: dict[str, float]
    by_status: dict[str, int]
    by_file_type: dict[str, int]
    by_category: dict[str, int]
    top_tags: list[dict[str, int]]
    uploads_by_day: list[dict[str, Any]]

    @property
    def success_rate(self) -> float:
        """Share of terminal documents that completed successfully."""
        ready = self.by_status.get(DocumentStatus.READY.value, 0)
        failed = self.by_status.get(DocumentStatus.FAILED.value, 0)
        terminal = ready + failed
        return round(ready / terminal, 4) if terminal else 1.0


@dataclass(frozen=True, slots=True)
class ProcessingStatistics:
    """Pipeline throughput and reliability, broken down per stage."""

    stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_runs(self) -> int:
        return sum(int(stage["runs"]) for stage in self.stages)

    @property
    def total_failures(self) -> int:
        return sum(int(stage["failed"]) for stage in self.stages)

    @property
    def overall_success_rate(self) -> float:
        if not self.total_runs:
            return 1.0
        return round(1 - self.total_failures / self.total_runs, 4)

    @property
    def slowest_stage(self) -> str | None:
        if not self.stages:
            return None
        return max(self.stages, key=lambda stage: stage["avg_ms"])["stage"]


@dataclass(frozen=True, slots=True)
class AiUsageStatistics:
    """Token consumption and cost, overall and per operation."""

    total: dict[str, Any]
    by_operation: list[dict[str, Any]]


class MetricsService:
    """Aggregates monitoring data for the metrics API."""

    def __init__(self, session: AsyncSession) -> None:
        self._documents = DocumentStatsRepository(session)
        self._processing = ProcessingMetricRepository(session)
        self._usage = ApiUsageRepository(session)

    async def document_statistics(
        self, principal: Principal
    ) -> DocumentStatistics:
        user_id = principal.user_id
        return DocumentStatistics(
            totals=await self._documents.totals(user_id),
            by_status=await self._documents.counts_by_status(user_id),
            by_file_type=await self._documents.counts_by_file_type(user_id),
            by_category=await self._documents.counts_by_category(user_id),
            top_tags=await self._documents.top_tags(user_id),
            uploads_by_day=await self._documents.uploads_by_day(user_id),
        )

    async def processing_statistics(self) -> ProcessingStatistics:
        stages = await self._processing.stage_statistics()
        # Report every stage, including ones that have never run, so a stage
        # that silently stopped executing is visible as a zero rather than a
        # missing key.
        seen = {stage["stage"] for stage in stages}
        for stage in ProcessingStage:
            if stage.value not in seen:
                stages.append(
                    {
                        "stage": stage.value,
                        "runs": 0,
                        "succeeded": 0,
                        "failed": 0,
                        "avg_ms": 0.0,
                        "p95_ms": 0.0,
                        "max_ms": 0,
                    }
                )

        order = [stage.value for stage in ProcessingStage]
        stages.sort(key=lambda item: order.index(item["stage"]))
        return ProcessingStatistics(stages=stages)

    async def ai_usage_statistics(self) -> AiUsageStatistics:
        return AiUsageStatistics(
            total=await self._usage.grand_total(),
            by_operation=await self._usage.totals_by_operation(),
        )
