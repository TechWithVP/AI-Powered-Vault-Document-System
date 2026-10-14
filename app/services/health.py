"""Health checking.

Each dependency is probed by a small, independently testable class. Probes run
concurrently and never raise: a failure is captured as a degraded component so
the endpoint always answers with a complete picture of the system.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.redis import get_async_redis
from app.db.session import db_manager
from app.storage import get_storage_backend

logger = get_logger(__name__)


class ComponentStatus(StrEnum):
    UP = "up"
    DOWN = "down"
    NOT_CONFIGURED = "not_configured"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """Outcome of probing a single dependency."""

    name: str
    status: ComponentStatus
    latency_ms: float
    detail: str | None = None

    @property
    def is_critical_failure(self) -> bool:
        return self.status is ComponentStatus.DOWN


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Aggregated health of every dependency."""

    healthy: bool
    components: list[ComponentHealth]


class HealthProbe(ABC):
    """A single dependency check."""

    name: str

    @abstractmethod
    async def run(self) -> None:
        """Complete without raising when the dependency is usable."""

    async def execute(self) -> ComponentHealth:
        """Run the probe, converting any failure into a report entry."""
        started = time.perf_counter()
        try:
            await self.run()
            status, detail = ComponentStatus.UP, None
        except Exception as exc:
            logger.warning(
                "Health probe failed",
                extra={"component": self.name, "error": str(exc)},
            )
            status, detail = ComponentStatus.DOWN, str(exc)[:300]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ComponentHealth(
            name=self.name,
            status=status,
            latency_ms=round(elapsed_ms, 2),
            detail=detail,
        )


class DatabaseProbe(HealthProbe):
    name = "database"

    async def run(self) -> None:
        await db_manager.verify_connection()


class RedisProbe(HealthProbe):
    name = "redis"

    async def run(self) -> None:
        await get_async_redis().ping()


class StorageProbe(HealthProbe):
    name = "object_storage"

    async def run(self) -> None:
        # boto3 is blocking; keep the event loop free.
        await asyncio.to_thread(get_storage_backend().health_check)


class LLMConfigProbe(HealthProbe):
    """Verifies the AI provider is configured without spending a token."""

    name = "llm_provider"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self) -> None:
        if not self._settings.openai.is_configured:
            raise RuntimeError("OPENAI_API_KEY is not set")

    async def execute(self) -> ComponentHealth:
        report = await super().execute()
        if report.status is ComponentStatus.DOWN:
            # Missing configuration is reported distinctly from an outage.
            return ComponentHealth(
                name=report.name,
                status=ComponentStatus.NOT_CONFIGURED,
                latency_ms=report.latency_ms,
                detail=report.detail,
            )
        return report


class HealthService:
    """Runs every probe concurrently and aggregates the result."""

    def __init__(self, settings: Settings, probes: list[HealthProbe] | None = None):
        self._probes = probes or [
            DatabaseProbe(),
            RedisProbe(),
            StorageProbe(),
            LLMConfigProbe(settings),
        ]

    async def check(self) -> HealthReport:
        components = await asyncio.gather(
            *(probe.execute() for probe in self._probes)
        )
        ordered = list(components)
        healthy = not any(component.is_critical_failure for component in ordered)
        return HealthReport(healthy=healthy, components=ordered)
