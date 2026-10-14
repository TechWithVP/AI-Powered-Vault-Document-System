"""Monitoring endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentPrincipal, MetricsServiceDep
from app.schemas.metrics import (
    AiUsageResponse,
    DocumentStatisticsResponse,
    ProcessingStatisticsResponse,
    RetrievalMetricsResponse,
)

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get(
    "/documents",
    response_model=DocumentStatisticsResponse,
    summary="Document statistics",
    description=(
        "Composition of your vault: totals, breakdowns by status, file type "
        "and AI-assigned category, the most common tags, and upload volume "
        "over the last two weeks."
    ),
)
async def document_statistics(
    principal: CurrentPrincipal,
    service: MetricsServiceDep,
) -> DocumentStatisticsResponse:
    return DocumentStatisticsResponse.from_domain(
        await service.document_statistics(principal)
    )


@router.get(
    "/processing",
    response_model=ProcessingStatisticsResponse,
    summary="Processing metrics",
    description=(
        "Pipeline health per stage: run counts, success and failure totals, "
        "and average, p95 and worst-case durations. Stages that have never "
        "run are reported as zeros so a stalled stage is visible."
    ),
)
async def processing_statistics(
    principal: CurrentPrincipal,
    service: MetricsServiceDep,
) -> ProcessingStatisticsResponse:
    return ProcessingStatisticsResponse.from_domain(
        await service.processing_statistics()
    )


@router.get(
    "/ai-usage",
    response_model=AiUsageResponse,
    summary="AI usage and cost",
    description=(
        "Token consumption and USD cost, in total and broken down by "
        "operation and model. Every provider call is recorded as it happens."
    ),
)
async def ai_usage(
    principal: CurrentPrincipal,
    service: MetricsServiceDep,
) -> AiUsageResponse:
    return AiUsageResponse.from_domain(await service.ai_usage_statistics())


@router.get(
    "/retrieval",
    response_model=RetrievalMetricsResponse,
    summary="Retrieval arm attribution",
    description=(
        "How much of each search came from vector similarity versus "
        "exact-term matching. Recorded per search at fusion, which is the "
        "only point where the two signals share a scale. A lexical share "
        "that drifts to zero means the hybrid has quietly become "
        "single-arm retrieval."
    ),
)
async def retrieval_metrics(
    principal: CurrentPrincipal,
    service: MetricsServiceDep,
) -> RetrievalMetricsResponse:
    return RetrievalMetricsResponse.from_domain(
        await service.retrieval_statistics()
    )
