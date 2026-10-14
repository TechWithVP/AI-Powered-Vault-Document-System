"""Health and readiness endpoints."""

from __future__ import annotations

from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, Response

from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse, LivenessResponse
from app.services.health import HealthService

router = APIRouter(tags=["Health"])


def get_health_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthService:
    return HealthService(settings)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Full dependency health check",
    description=(
        "Probes PostgreSQL, Redis, object storage and the AI provider "
        "configuration. Returns 503 when a critical dependency is unreachable."
    ),
)
async def health(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[HealthService, Depends(get_health_service)],
) -> HealthResponse:
    report = await service.check()
    if not report.healthy:
        response.status_code = HTTPStatus.SERVICE_UNAVAILABLE
    return HealthResponse.from_domain(
        report,
        version=settings.app.version,
        environment=settings.app.env.value,
    )


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Cheap check that the process is serving requests.",
)
async def liveness() -> LivenessResponse:
    return LivenessResponse()
