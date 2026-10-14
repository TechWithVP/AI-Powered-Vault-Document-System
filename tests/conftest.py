"""Shared pytest fixtures.

Environment defaults are set before application modules are imported so tests
never reach a real provider or a developer bucket.
"""

from __future__ import annotations

import os

# Forced rather than defaulted: docker compose exports the developer's .env into
# the container, so setdefault would leave real values in place and make the
# suite depend on local configuration.
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["APP_API_KEYS"] = "test-api-key"
os.environ["APP_LOG_JSON"] = "false"
os.environ["STORAGE_BACKEND"] = "local"
# The limiter shares one Redis counter, so leaving it on would make later
# tests fail with 429s. Its behaviour is covered directly in
# tests/unit/test_rate_limit.py.
os.environ["RATE_LIMIT_ENABLED"] = "false"

from collections.abc import AsyncIterator, Iterator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.main import create_app  # noqa: E402

TEST_API_KEY = "test-api-key"


@asynccontextmanager
async def _null_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Replaces the real lifespan so unit tests need no live dependencies."""
    yield


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers carrying a valid API key."""
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def app() -> FastAPI:
    """Application instance with external start-up hooks disabled."""
    application = create_app()
    application.router.lifespan_context = _null_lifespan
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
