"""Health endpoint schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.services.health import ComponentHealth, ComponentStatus, HealthReport


class ComponentHealthResponse(BaseModel):
    """Health of a single dependency."""

    name: str
    status: ComponentStatus
    latency_ms: float
    detail: str | None = None

    @classmethod
    def from_domain(cls, component: ComponentHealth) -> "ComponentHealthResponse":
        return cls(
            name=component.name,
            status=component.status,
            latency_ms=component.latency_ms,
            detail=component.detail,
        )


class HealthResponse(BaseModel):
    """Overall service health plus a per-dependency breakdown."""

    status: str = Field(description="ok when every critical dependency is reachable.")
    version: str
    environment: str
    components: list[ComponentHealthResponse]

    @classmethod
    def from_domain(
        cls, report: HealthReport, *, version: str, environment: str
    ) -> "HealthResponse":
        return cls(
            status="ok" if report.healthy else "degraded",
            version=version,
            environment=environment,
            components=[
                ComponentHealthResponse.from_domain(component)
                for component in report.components
            ],
        )


class LivenessResponse(BaseModel):
    """Process-liveness payload; does not touch dependencies."""

    status: str = "alive"
