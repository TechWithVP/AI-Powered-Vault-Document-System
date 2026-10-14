"""Caching for AI results.

Query embeddings are deterministic for a given model and text, so repeating a
question should not cost another provider call. The cache is keyed on both the
model and the text, so changing the embedding model cannot serve stale vectors
of the wrong dimensionality.

Every operation degrades gracefully: a Redis failure is logged and treated as a
miss, because losing a cache must never lose a request.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Query embeddings never change for a given model, so they are held for a day.
EMBEDDING_TTL_SECONDS = 24 * 60 * 60

KEY_PREFIX = "embedding"


class EmbeddingCache(ABC):
    """Stores and retrieves query embeddings."""

    @abstractmethod
    async def get(self, text: str, *, model: str) -> list[float] | None:
        """Return a cached vector, or None on a miss."""

    @abstractmethod
    async def set(self, text: str, vector: list[float], *, model: str) -> None:
        """Store a vector."""


class NullEmbeddingCache(EmbeddingCache):
    """No-op cache used when caching is disabled or in tests."""

    async def get(self, text: str, *, model: str) -> list[float] | None:
        return None

    async def set(self, text: str, vector: list[float], *, model: str) -> None:
        return None


class RedisEmbeddingCache(EmbeddingCache):
    """Redis-backed embedding cache."""

    def __init__(self, redis: Redis, ttl_seconds: int = EMBEDDING_TTL_SECONDS):
        self._redis = redis
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    @staticmethod
    def build_key(text: str, *, model: str) -> str:
        """Key on the model and a digest of the text.

        Hashing keeps keys bounded regardless of question length, and including
        the model prevents a model change from serving vectors of the wrong
        width.
        """
        digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        return f"{KEY_PREFIX}:{model}:{digest}"

    async def get(self, text: str, *, model: str) -> list[float] | None:
        try:
            raw = await self._redis.get(self.build_key(text, model=model))
        except RedisError as exc:
            logger.warning("Embedding cache read failed", extra={"error": str(exc)})
            return None

        if raw is None:
            self.misses += 1
            return None

        try:
            vector = json.loads(raw)
        except (TypeError, ValueError):
            # A corrupt entry is treated as a miss rather than an error.
            self.misses += 1
            return None

        self.hits += 1
        logger.debug("Embedding cache hit", extra={"model": model})
        return vector

    async def set(self, text: str, vector: list[float], *, model: str) -> None:
        try:
            await self._redis.setex(
                self.build_key(text, model=model), self._ttl, json.dumps(vector)
            )
        except RedisError as exc:
            logger.warning("Embedding cache write failed", extra={"error": str(exc)})
