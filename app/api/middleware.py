"""HTTP middleware: request correlation and access logging."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import CorrelationId, get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

NextCall = Callable[[Request], Awaitable[Response]]


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Binds a correlation id to the request context and echoes it back.

    An inbound X-Request-ID is honoured so a caller can trace a request across
    service boundaries; otherwise a fresh id is generated.
    """

    async def dispatch(self, request: Request, call_next: NextCall) -> Response:
        token = CorrelationId.set(request.headers.get(REQUEST_ID_HEADER))
        correlation_id = CorrelationId.get() or ""
        request.state.correlation_id = correlation_id
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = correlation_id
            return response
        finally:
            CorrelationId.reset(token)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Emits one structured log line per request with method, path and timing."""

    #: Paths excluded to keep probe traffic out of the logs.
    EXCLUDED_PATHS = frozenset({"/health/live", "/metrics-probe", "/favicon.ico"})

    async def dispatch(self, request: Request, call_next: NextCall) -> Response:
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The exception handlers render the response; log the failure here
            # so timing is captured even for unhandled errors.
            logger.exception(
                "Request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["X-Response-Time-ms"] = str(duration_ms)
        return response
