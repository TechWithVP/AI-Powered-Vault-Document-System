"""Request rate limiting.

A fixed-window counter in Redis, keyed per API key. Redis rather than process
memory so the limit holds across both API workers and survives a restart; if
Redis is unavailable the limiter fails open, because losing rate limiting is
preferable to losing the service.

A fixed window is used rather than a sliding one for its single round trip per
request. The known trade-off is that a client straddling a window boundary can
briefly send up to twice the limit; that is acceptable for abuse protection at
this scale, where the goal is bounding sustained load rather than policing
burst shape precisely.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.exceptions import RateLimitedError
from app.core.logging import get_logger
from app.core.security import API_KEY_HEADER_NAME

logger = get_logger(__name__)

#: Paths exempt from limiting: probes and docs must stay reachable.
EXEMPT_PREFIXES = ("/health", "/docs", "/redoc", "/openapi.json", "/ui")


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """How many requests a client may make per window."""

    requests: int = 120
    window_seconds: int = 60

    @property
    def description(self) -> str:
        return f"{self.requests} requests per {self.window_seconds}s"


@dataclass(frozen=True, slots=True)
class RateLimitVerdict:
    """Outcome of checking one request against the policy."""

    allowed: bool
    remaining: int
    reset_after: int


class RedisRateLimiter:
    """Fixed-window counter backed by Redis."""

    def __init__(self, redis: Redis, policy: RateLimitPolicy) -> None:
        self._redis = redis
        self._policy = policy

    def _key(self, identity: str, now: float) -> str:
        window = int(now // self._policy.window_seconds)
        return f"ratelimit:{identity}:{window}"

    async def check(self, identity: str) -> RateLimitVerdict:
        """Count this request and report whether it is within the limit."""
        now = time.time()
        key = self._key(identity, now)
        window = self._policy.window_seconds
        reset_after = window - int(now % window)

        try:
            pipeline = self._redis.pipeline()
            pipeline.incr(key)
            # Expire slightly past the window so the key always clears itself.
            pipeline.expire(key, window + 1)
            count, _ = await pipeline.execute()
        except RedisError as exc:
            logger.warning(
                "Rate limiter unavailable; allowing request",
                extra={"error": str(exc)},
            )
            return RateLimitVerdict(True, self._policy.requests, reset_after)

        remaining = max(self._policy.requests - int(count), 0)
        return RateLimitVerdict(
            allowed=int(count) <= self._policy.requests,
            remaining=remaining,
            reset_after=reset_after,
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Applies the rate limit and advertises the result in headers."""

    def __init__(self, app, limiter: RedisRateLimiter, policy: RateLimitPolicy):
        super().__init__(app)
        self._limiter = limiter
        self._policy = policy

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path.startswith(EXEMPT_PREFIXES):
            return await call_next(request)

        identity = self._identity(request)
        verdict = await self._limiter.check(identity)

        if not verdict.allowed:
            error = RateLimitedError(
                f"Rate limit exceeded ({self._policy.description}). "
                f"Retry in {verdict.reset_after}s.",
                details={"retry_after_seconds": verdict.reset_after},
            )
            logger.warning(
                "Request rate limited", extra={"path": request.url.path}
            )
            return JSONResponse(
                status_code=error.status_code,
                content=error.to_payload(),
                headers=self._headers(verdict) | {"Retry-After": str(verdict.reset_after)},
            )

        response = await call_next(request)
        response.headers.update(self._headers(verdict))
        return response

    def _headers(self, verdict: RateLimitVerdict) -> dict[str, str]:
        return {
            "X-RateLimit-Limit": str(self._policy.requests),
            "X-RateLimit-Remaining": str(verdict.remaining),
            "X-RateLimit-Reset": str(verdict.reset_after),
        }

    @staticmethod
    def _identity(request: Request) -> str:
        """Limit per API key, falling back to client address when absent."""
        if api_key := request.headers.get(API_KEY_HEADER_NAME):
            return f"key:{api_key[:32]}"
        client = request.client.host if request.client else "unknown"
        return f"ip:{client}"
