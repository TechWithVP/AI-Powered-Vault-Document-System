"""Health endpoints and the error envelope."""

from __future__ import annotations

from http import HTTPStatus
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.services.health import (
    ComponentHealth,
    ComponentStatus,
    HealthProbe,
    HealthReport,
    HealthService,
)


class TestLiveness:
    def test_returns_alive_without_touching_dependencies(
        self, client: TestClient
    ) -> None:
        response = client.get("/health/live")
        assert response.status_code == HTTPStatus.OK
        assert response.json() == {"status": "alive"}

    def test_requires_no_api_key(self, client: TestClient) -> None:
        assert client.get("/health/live").status_code == HTTPStatus.OK


class TestHealth:
    def test_reports_ok_when_every_probe_passes(self, client: TestClient) -> None:
        report = HealthReport(
            healthy=True,
            components=[ComponentHealth("database", ComponentStatus.UP, 1.0)],
        )
        with patch.object(HealthService, "check", AsyncMock(return_value=report)):
            response = client.get("/health")

        assert response.status_code == HTTPStatus.OK
        assert response.json()["status"] == "ok"

    def test_returns_503_when_a_dependency_is_down(self, client: TestClient) -> None:
        report = HealthReport(
            healthy=False,
            components=[
                ComponentHealth("database", ComponentStatus.DOWN, 1.0, "refused")
            ],
        )
        with patch.object(HealthService, "check", AsyncMock(return_value=report)):
            response = client.get("/health")

        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert response.json()["status"] == "degraded"
        assert response.json()["components"][0]["detail"] == "refused"


class TestErrorEnvelope:
    def test_unknown_route_uses_the_standard_envelope(
        self, client: TestClient
    ) -> None:
        body = client.get("/does-not-exist").json()
        assert body["error"]["code"] == "not_found"
        assert "message" in body["error"]

    def test_correlation_id_is_echoed_back(self, client: TestClient) -> None:
        response = client.get("/health/live", headers={"X-Request-ID": "trace-123"})
        assert response.headers["X-Request-ID"] == "trace-123"

    def test_correlation_id_is_generated_when_absent(
        self, client: TestClient
    ) -> None:
        assert client.get("/health").headers.get("X-Request-ID")


class _FailingProbe(HealthProbe):
    name = "explodes"

    async def run(self) -> None:
        raise RuntimeError("connection refused")


class _PassingProbe(HealthProbe):
    name = "fine"

    async def run(self) -> None:
        return None


class TestHealthService:
    @pytest.mark.asyncio
    async def test_a_failing_probe_degrades_the_report_without_raising(self) -> None:
        service = HealthService(settings=None, probes=[_PassingProbe(), _FailingProbe()])
        report = await service.check()

        assert report.healthy is False
        statuses = {c.name: c.status for c in report.components}
        assert statuses == {
            "fine": ComponentStatus.UP,
            "explodes": ComponentStatus.DOWN,
        }

    @pytest.mark.asyncio
    async def test_failure_detail_is_captured(self) -> None:
        service = HealthService(settings=None, probes=[_FailingProbe()])
        report = await service.check()
        assert "connection refused" in (report.components[0].detail or "")

    @pytest.mark.asyncio
    async def test_all_passing_probes_are_healthy(self) -> None:
        service = HealthService(settings=None, probes=[_PassingProbe()])
        assert (await service.check()).healthy is True
