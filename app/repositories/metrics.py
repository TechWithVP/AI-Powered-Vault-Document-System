"""Persistence for pipeline metrics and AI usage records."""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import case, func, select

from app.core.constants import AIOperation, ProcessingStage, StageStatus
from app.models.metrics import ApiUsage, ProcessingMetric
from app.repositories.base import AsyncRepository, SyncRepository


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
