"""Embedding cache behaviour."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.exceptions import RedisError

from app.ai.cache import NullEmbeddingCache, RedisEmbeddingCache

VECTOR = [0.1, 0.2, 0.3]
MODEL = "text-embedding-3-small"


def fake_redis(stored: str | None = None, *, failing: bool = False) -> MagicMock:
    redis = MagicMock()
    error = RedisError("unavailable")
    redis.get = AsyncMock(side_effect=error if failing else None, return_value=stored)
    redis.setex = AsyncMock(side_effect=error if failing else None)
    return redis


class TestKeying:
    def test_identical_text_and_model_share_a_key(self) -> None:
        first = RedisEmbeddingCache.build_key("hello", model=MODEL)
        second = RedisEmbeddingCache.build_key("hello", model=MODEL)
        assert first == second

    def test_surrounding_whitespace_is_ignored(self) -> None:
        assert RedisEmbeddingCache.build_key("  hello  ", model=MODEL) == (
            RedisEmbeddingCache.build_key("hello", model=MODEL)
        )

    def test_different_text_gets_a_different_key(self) -> None:
        assert RedisEmbeddingCache.build_key("a", model=MODEL) != (
            RedisEmbeddingCache.build_key("b", model=MODEL)
        )

    def test_the_model_is_part_of_the_key(self) -> None:
        """A model change must not serve vectors of the wrong width."""
        assert RedisEmbeddingCache.build_key("hello", model="small") != (
            RedisEmbeddingCache.build_key("hello", model="large")
        )

    def test_keys_are_bounded_regardless_of_input_length(self) -> None:
        key = RedisEmbeddingCache.build_key("x" * 100_000, model=MODEL)
        assert len(key) < 120


class TestReadWrite:
    async def test_a_stored_vector_is_returned(self) -> None:
        cache = RedisEmbeddingCache(fake_redis(json.dumps(VECTOR)))
        assert await cache.get("hello", model=MODEL) == VECTOR
        assert cache.hits == 1

    async def test_a_missing_entry_is_a_miss(self) -> None:
        cache = RedisEmbeddingCache(fake_redis(None))
        assert await cache.get("hello", model=MODEL) is None
        assert cache.misses == 1

    async def test_writes_use_the_configured_ttl(self) -> None:
        redis = fake_redis()
        cache = RedisEmbeddingCache(redis, ttl_seconds=60)

        await cache.set("hello", VECTOR, model=MODEL)

        key, ttl, payload = redis.setex.await_args.args
        assert ttl == 60
        assert json.loads(payload) == VECTOR


class TestResilience:
    async def test_a_read_failure_is_treated_as_a_miss(self) -> None:
        cache = RedisEmbeddingCache(fake_redis(failing=True))
        assert await cache.get("hello", model=MODEL) is None

    async def test_a_write_failure_does_not_raise(self) -> None:
        cache = RedisEmbeddingCache(fake_redis(failing=True))
        await cache.set("hello", VECTOR, model=MODEL)  # must not raise

    async def test_corrupt_entries_are_treated_as_misses(self) -> None:
        cache = RedisEmbeddingCache(fake_redis("not json"))
        assert await cache.get("hello", model=MODEL) is None


class TestNullCache:
    async def test_it_never_returns_a_vector(self) -> None:
        cache = NullEmbeddingCache()
        await cache.set("hello", VECTOR, model=MODEL)
        assert await cache.get("hello", model=MODEL) is None
