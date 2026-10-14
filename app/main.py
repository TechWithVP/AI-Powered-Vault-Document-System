"""FastAPI application factory and process lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.errors import register_exception_handlers
from app.api.middleware import AccessLogMiddleware, CorrelationIdMiddleware
from app.api.rate_limit import (
    RateLimitMiddleware,
    RateLimitPolicy,
    RedisRateLimiter,
)
from app.api.v1 import health
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.core.redis import close_async_redis, get_async_redis
from app.core.security import get_principal
from app.db.session import db_manager
from app.schemas.common import ErrorResponse

logger = get_logger(__name__)

API_V1_PREFIX = "/api/v1"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

#: Documented on every authenticated route so the generated OpenAPI spec shows
#: the shared error envelope rather than FastAPI's default detail object.
COMMON_RESPONSES: dict[int | str, dict] = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    404: {"model": ErrorResponse, "description": "Resource not found"},
    422: {"model": ErrorResponse, "description": "Invalid request payload"},
    500: {"model": ErrorResponse, "description": "Unexpected server error"},
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start-up and shutdown hooks for shared resources."""
    settings: Settings = get_settings()
    logger.info(
        "Starting application",
        extra={
            "version": settings.app.version,
            "environment": settings.app.env.value,
            "chat_model": settings.openai.chat_model,
            "embedding_model": settings.openai.embedding_model,
        },
    )
    yield
    await db_manager.dispose()
    await close_async_redis()
    logger.info("Application shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(
        level=settings.app.log_level, json_output=settings.app.log_json
    )

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description=(
            "Upload documents, get AI-generated insights, and chat with your "
            "documents using retrieval-augmented generation with source "
            "citations."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    _register_middleware(app, settings)
    register_exception_handlers(app)
    _register_routers(app)
    _mount_static(app)
    return app


def _register_middleware(app: FastAPI, settings: Settings) -> None:
    # Added innermost first: rate limiting runs after the correlation id is
    # bound and inside access logging, so a rejected request is still logged.
    if settings.rate_limit.enabled:
        policy = RateLimitPolicy(
            requests=settings.rate_limit.requests,
            window_seconds=settings.rate_limit.window_seconds,
        )
        app.add_middleware(
            RateLimitMiddleware,
            limiter=RedisRateLimiter(get_async_redis(), policy),
            policy=policy,
        )

    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.app.is_production else [],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-ms"],
    )


def _register_routers(app: FastAPI) -> None:
    # Unauthenticated operational probes.
    app.include_router(health.router)
    # Authenticated resource API.
    app.include_router(
        api_router,
        prefix=API_V1_PREFIX,
        dependencies=[Depends(get_principal)],
        responses=COMMON_RESPONSES,
    )


def _mount_static(app: FastAPI) -> None:
    """Serve the minimal chat UI when the static bundle is present."""
    if STATIC_DIR.is_dir() and any(STATIC_DIR.iterdir()):
        app.mount(
            "/ui",
            StaticFiles(directory=STATIC_DIR, html=True),
            name="ui",
        )


app = create_app()
