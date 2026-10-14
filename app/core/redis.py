"""Redis client providers.

Redis backs both the Celery broker and the response/embedding cache. Async and
sync clients are exposed separately so the API and the worker each use the
right flavour, and both are cached per process.
"""

from __future__ import annotations

from functools import lru_cache

import redis
import redis.asyncio as aioredis

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_async_redis() -> aioredis.Redis:
    """Async client used by the FastAPI request path."""
    return aioredis.from_url(
        get_settings().redis.cache_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=5,
        health_check_interval=30,
    )


@lru_cache(maxsize=1)
def get_sync_redis() -> redis.Redis:
    """Sync client used by Celery tasks."""
    return redis.from_url(
        get_settings().redis.cache_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=5,
        health_check_interval=30,
    )


async def close_async_redis() -> None:
    """Close pooled async connections on shutdown."""
    client = get_async_redis()
    await client.aclose()
    get_async_redis.cache_clear()
