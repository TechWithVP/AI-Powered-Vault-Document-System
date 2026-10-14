"""Persistence for pipeline metrics, AI usage and retrieval attribution."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import case, func, select

from app.core.constants import AIOperation, ProcessingStage, StageStatus
from app.models.metrics import ApiUsage, ProcessingMetric, RetrievalMetric
from app.repositories.base import AsyncRepository, SyncRepository

if TYPE_CHECKING:  # pragma: no cover - import cycle: retrieval imports chunks
    from app.ai.rag.retriever import RetrievalOutcome


class SyncProcessingMetricRepository(SyncRepository[ProcessingMetric]):
    """Records stage outcomes from inside the worker."""

    model = ProcessingMetric

    def record(
        self,
        *,
        document_id: uuid.UUID,
        stage: ProcessingStage,
        status: StageStatus,
        duration_ms: int,
        items_processed: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> ProcessingMetric:
        metric = ProcessingMetric(
            document_id=document_id,
            stage=stage,
            status=status,
            duration_ms=duration_ms,
            items_processed=items_processed,
            error_code=error_code,
            error_message=(error_message or None) and error_message[:2000],
        )
        return self.add(metric)


class SyncApiUsageRepository(SyncRepository[ApiUsage]):
    """Records billable AI calls from inside the worker."""

    model = ApiUsage

    def record(
        self,
        *,
        operation: AIOperation,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: Decimal,
        document_id: uuid.UUID | None = None,
        session_id: uuid.UUID | None = None,
        latency_ms: int | None = None,
        endpoint: str | None = None,
    ) -> ApiUsage:
        usage = ApiUsage(
            operation=operation,
            model=model,
            endpoint=endpoint,
            document_id=document_id,
            session_id=session_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        return self.add(usage)


class ProcessingMetricRepository(AsyncRepository[ProcessingMetric]):
    """Async aggregation for the processing-metrics endpoint."""

    model = ProcessingMetric

    async def stage_statistics(self) -> list[dict[str, object]]:
        """Per-stage duration percentiles and success/failure counts."""
        duration = ProcessingMetric.duration_ms
        statement = (
            select(
                ProcessingMetric.stage,
                func.count().label("runs"),
                func.sum(
                    case((ProcessingMetric.status == StageStatus.SUCCEEDED, 1), else_=0)
                ).label("succeeded"),
                func.avg(duration).label("avg_ms"),
                func.percentile_cont(0.95)
                .within_group(duration.asc())
                .label("p95_ms"),
                func.max(duration).label("max_ms"),
            )
            .where(ProcessingMetric.status != StageStatus.STARTED)
            .group_by(ProcessingMetric.stage)
        )
        rows = await self.session.execute(statement)
        return [
            {
                "stage": str(row.stage),
                "runs": int(row.runs or 0),
                "succeeded": int(row.succeeded or 0),
                "failed": int(row.runs or 0) - int(row.succeeded or 0),
                "avg_ms": round(float(row.avg_ms or 0), 2),
                "p95_ms": round(float(row.p95_ms or 0), 2),
                "max_ms": int(row.max_ms or 0),
            }
            for row in rows.all()
        ]


class ApiUsageRepository(AsyncRepository[ApiUsage]):
    """Async recording and aggregation for the AI cost endpoint."""

    model = ApiUsage

    def record(
        self,
        *,
        operation: AIOperation,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: Decimal,
        session_id: uuid.UUID | None = None,
        document_id: uuid.UUID | None = None,
        latency_ms: int | None = None,
    ) -> ApiUsage:
        """Stage a usage row; the request's unit of work commits it."""
        usage = ApiUsage(
            operation=operation,
            model=model,
            document_id=document_id,
            session_id=session_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        return self.add(usage)

    async def totals_by_operation(self) -> list[dict[str, object]]:
        statement = (
            select(
                ApiUsage.operation,
                ApiUsage.model,
                func.count().label("calls"),
                func.sum(ApiUsage.total_tokens).label("tokens"),
                func.sum(ApiUsage.cost_usd).label("cost"),
            )
            .group_by(ApiUsage.operation, ApiUsage.model)
            .order_by(func.sum(ApiUsage.cost_usd).desc())
        )
        rows = await self.session.execute(statement)
        return [
            {
                "operation": str(row.operation),
                "model": row.model,
                "calls": int(row.calls or 0),
                "total_tokens": int(row.tokens or 0),
                "cost_usd": float(row.cost or Decimal("0")),
            }
            for row in rows.all()
        ]

    async def grand_total(self) -> dict[str, object]:
        statement = select(
            func.count().label("calls"),
            func.coalesce(func.sum(ApiUsage.total_tokens), 0).label("tokens"),
            func.coalesce(func.sum(ApiUsage.cost_usd), 0).label("cost"),
        )
        row = (await self.session.execute(statement)).one()
        return {
            "calls": int(row.calls or 0),
            "total_tokens": int(row.tokens or 0),
            "cost_usd": float(row.cost or 0),
        }


class RetrievalMetricRepository(AsyncRepository[RetrievalMetric]):
    """Records the arm split of each search and aggregates it for reporting."""

    model = RetrievalMetric

    def record(
        self,
        outcome: "RetrievalOutcome",
        *,
        question_kind: str,
        session_id: uuid.UUID | None = None,
        message_id: uuid.UUID | None = None,
        latency_ms: int | None = None,
    ) -> RetrievalMetric:
        """Stage a row from a retrieval outcome; the unit of work commits it.

        Takes the outcome rather than a dozen scalars so a new field on the
        outcome cannot be silently left out of the metric.
        """
        split = outcome.contribution
        return self.add(
            RetrievalMetric(
                session_id=session_id,
                message_id=message_id,
                question_kind=question_kind,
                semantic_contribution=split.semantic,
                lexical_contribution=split.lexical,
                semantic_share=split.semantic_share,
                lexical_share=split.lexical_share,
                results_count=len(outcome.results),
                semantic_only=split.semantic_only,
                lexical_only=split.lexical_only,
                found_by_both=split.both,
                best_similarity=outcome.best_similarity,
                best_lexical_rank=outcome.best_lexical_rank,
                is_relevant=outcome.is_relevant,
                reranked=outcome.reranked,
                latency_ms=latency_ms,
            )
        )

    async def arm_summary(
        self, session_id: uuid.UUID | None = None
    ) -> dict[str, object]:
        """Split between the two arms, corpus-wide or for one conversation.

        Averages the per-search shares rather than dividing summed
        contributions: every search should count once regardless of how many
        chunks it happened to return.
        """
        statement = select(
            func.count(RetrievalMetric.id).label("searches"),
            func.avg(RetrievalMetric.semantic_share).label("semantic_share"),
            func.avg(RetrievalMetric.lexical_share).label("lexical_share"),
            func.sum(RetrievalMetric.semantic_only).label("semantic_only"),
            func.sum(RetrievalMetric.lexical_only).label("lexical_only"),
            func.sum(RetrievalMetric.found_by_both).label("found_by_both"),
            func.avg(RetrievalMetric.best_similarity).label("avg_best_similarity"),
            func.avg(RetrievalMetric.best_lexical_rank).label("avg_best_lexical_rank"),
        )
        if session_id is not None:
            statement = statement.where(RetrievalMetric.session_id == session_id)
        row = (await self.session.execute(statement)).one()
        searches = int(row.searches or 0)
        return {
            "searches": searches,
            "semantic_share": round(float(row.semantic_share or 0.0), 4),
            "lexical_share": round(float(row.lexical_share or 0.0), 4),
            "chunks_from_semantic_only": int(row.semantic_only or 0),
            "chunks_from_lexical_only": int(row.lexical_only or 0),
            "chunks_from_both": int(row.found_by_both or 0),
            "avg_best_similarity": round(float(row.avg_best_similarity or 0.0), 4),
            "avg_best_lexical_rank": round(float(row.avg_best_lexical_rank or 0.0), 4),
        }

    async def by_question_kind(
        self, session_id: uuid.UUID | None = None
    ) -> list[dict[str, object]]:
        """The same split per question kind.

        Counting questions retrieve a wider window, so mixing their shares
        with ordinary ones would blur exactly the comparison this is for.
        """
        statement = (
            select(
                RetrievalMetric.question_kind,
                func.count(RetrievalMetric.id).label("searches"),
                func.avg(RetrievalMetric.semantic_share).label("semantic_share"),
                func.avg(RetrievalMetric.lexical_share).label("lexical_share"),
                func.avg(
                    case((RetrievalMetric.is_relevant, 1.0), else_=0.0)
                ).label("relevant_rate"),
            )
            .group_by(RetrievalMetric.question_kind)
            .order_by(func.count(RetrievalMetric.id).desc())
        )
        if session_id is not None:
            statement = statement.where(RetrievalMetric.session_id == session_id)
        rows = await self.session.execute(statement)
        return [
            {
                "question_kind": row.question_kind,
                "searches": int(row.searches),
                "semantic_share": round(float(row.semantic_share or 0.0), 4),
                "lexical_share": round(float(row.lexical_share or 0.0), 4),
                "relevant_rate": round(float(row.relevant_rate or 0.0), 4),
            }
            for row in rows
        ]
