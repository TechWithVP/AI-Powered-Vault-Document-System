"""Metrics aggregation logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.constants import ProcessingStage
from app.services.metrics_service import (
    AiUsageStatistics,
    DocumentStatistics,
    MetricsService,
    ProcessingStatistics,
)


def make_document_stats(by_status: dict[str, int]) -> DocumentStatistics:
    return DocumentStatistics(
        totals={
            "documents": sum(by_status.values()),
            "pages": 0,
            "chunks": 0,
            "storage_bytes": 0,
            "avg_chunks_per_document": 0.0,
        },
        by_status=by_status,
        by_file_type={},
        by_category={},
        top_tags=[],
        uploads_by_day=[],
    )


class TestSuccessRate:
    def test_all_ready_is_a_perfect_rate(self) -> None:
        assert make_document_stats({"ready": 4}).success_rate == 1.0

    def test_failures_reduce_the_rate(self) -> None:
        stats = make_document_stats({"ready": 3, "failed": 1})
        assert stats.success_rate == 0.75

    def test_in_flight_documents_are_excluded(self) -> None:
        """Queued and processing documents have no outcome yet."""
        stats = make_document_stats({"ready": 1, "failed": 1, "processing": 10})
        assert stats.success_rate == 0.5

    def test_an_empty_vault_is_not_a_failure(self) -> None:
        assert make_document_stats({}).success_rate == 1.0


class TestProcessingStatistics:
    @staticmethod
    def _stage(name: str, runs: int, failed: int, avg: float) -> dict:
        return {
            "stage": name,
            "runs": runs,
            "succeeded": runs - failed,
            "failed": failed,
            "avg_ms": avg,
            "p95_ms": avg,
            "max_ms": int(avg),
        }

    def test_totals_are_summed_across_stages(self) -> None:
        stats = ProcessingStatistics(
            stages=[self._stage("extract", 10, 1, 5), self._stage("embed", 10, 2, 50)]
        )
        assert stats.total_runs == 20
        assert stats.total_failures == 3

    def test_overall_success_rate_reflects_failures(self) -> None:
        stats = ProcessingStatistics(
            stages=[self._stage("extract", 8, 2, 5)]
        )
        assert stats.overall_success_rate == 0.75

    def test_no_runs_is_not_a_failure(self) -> None:
        assert ProcessingStatistics(stages=[]).overall_success_rate == 1.0

    def test_slowest_stage_is_identified(self) -> None:
        stats = ProcessingStatistics(
            stages=[
                self._stage("extract", 1, 0, 5),
                self._stage("embed", 1, 0, 500),
                self._stage("chunk", 1, 0, 10),
            ]
        )
        assert stats.slowest_stage == "embed"

    def test_slowest_stage_is_none_without_data(self) -> None:
        assert ProcessingStatistics(stages=[]).slowest_stage is None


class TestServiceAggregation:
    @pytest.fixture
    def service(self) -> MetricsService:
        instance = MetricsService(MagicMock())
        instance._documents = MagicMock()
        instance._processing = MagicMock()
        instance._usage = MagicMock()
        return instance

    async def test_every_stage_is_reported_even_when_unused(
        self, service: MetricsService
    ) -> None:
        """A stage that stopped running must show as zero, not vanish."""
        service._processing.stage_statistics = AsyncMock(
            return_value=[
                {
                    "stage": "extract",
                    "runs": 3,
                    "succeeded": 3,
                    "failed": 0,
                    "avg_ms": 5.0,
                    "p95_ms": 7.0,
                    "max_ms": 8,
                }
            ]
        )

        stats = await service.processing_statistics()

        reported = {stage["stage"] for stage in stats.stages}
        assert reported == {stage.value for stage in ProcessingStage}

    async def test_stages_are_returned_in_pipeline_order(
        self, service: MetricsService
    ) -> None:
        service._processing.stage_statistics = AsyncMock(return_value=[])

        stats = await service.processing_statistics()

        assert [stage["stage"] for stage in stats.stages] == [
            stage.value for stage in ProcessingStage
        ]

    async def test_ai_usage_combines_totals_and_breakdown(
        self, service: MetricsService
    ) -> None:
        service._usage.grand_total = AsyncMock(
            return_value={"calls": 5, "total_tokens": 100, "cost_usd": 0.01}
        )
        service._usage.totals_by_operation = AsyncMock(
            return_value=[
                {
                    "operation": "chat",
                    "model": "gpt-4o-mini",
                    "calls": 5,
                    "total_tokens": 100,
                    "cost_usd": 0.01,
                }
            ]
        )

        stats: AiUsageStatistics = await service.ai_usage_statistics()

        assert stats.total["calls"] == 5
        assert stats.by_operation[0]["operation"] == "chat"
