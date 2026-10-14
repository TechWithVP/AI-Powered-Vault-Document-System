"""Rate limiting behaviour."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.exceptions import RedisError

from app.api.rate_limit import RateLimitPolicy, RedisRateLimiter


def fake_redis(count: int, *, failing: bool = False) -> MagicMock:
    """Redis double whose pipeline returns the given counter value."""
    pipeline = MagicMock()
    pipeline.incr = MagicMock()
    pipeline.expire = MagicMock()
    pipeline.execute = AsyncMock(
        side_effect=RedisError("down") if failing else None,
        return_value=[count, True],
    )
    redis = MagicMock()
    redis.pipeline = MagicMock(return_value=pipeline)
    return redis


@pytest.fixture
def policy() -> RateLimitPolicy:
    return RateLimitPolicy(requests=5, window_seconds=60)


class TestWithinLimit:
    async def test_first_request_is_allowed(self, policy: RateLimitPolicy) -> None:
        verdict = await RedisRateLimiter(fake_redis(1), policy).check("key:abc")

        assert verdict.allowed is True
        assert verdict.remaining == 4

    async def test_the_final_permitted_request_is_allowed(
        self, policy: RateLimitPolicy
    ) -> None:
        verdict = await RedisRateLimiter(fake_redis(5), policy).check("key:abc")

        assert verdict.allowed is True
        assert verdict.remaining == 0

    async def test_reset_is_within_the_window(
        self, policy: RateLimitPolicy
    ) -> None:
        verdict = await RedisRateLimiter(fake_redis(1), policy).check("key:abc")
        assert 0 < verdict.reset_after <= policy.window_seconds


class TestOverLimit:
    async def test_the_request_after_the_limit_is_rejected(
        self, policy: RateLimitPolicy
    ) -> None:
        verdict = await RedisRateLimiter(fake_redis(6), policy).check("key:abc")

        assert verdict.allowed is False
        assert verdict.remaining == 0

    async def test_remaining_never_goes_negative(
        self, policy: RateLimitPolicy
    ) -> None:
        verdict = await RedisRateLimiter(fake_redis(500), policy).check("key:abc")
        assert verdict.remaining == 0


class TestResilience:
    async def test_the_limiter_fails_open_when_redis_is_down(
        self, policy: RateLimitPolicy
    ) -> None:
        """Losing rate limiting is preferable to losing the service."""
        verdict = await RedisRateLimiter(fake_redis(1, failing=True), policy).check(
            "key:abc"
        )

        assert verdict.allowed is True
        assert verdict.remaining == policy.requests


class TestKeying:
    async def test_the_window_is_part_of_the_key(
        self, policy: RateLimitPolicy
    ) -> None:
        limiter = RedisRateLimiter(fake_redis(1), policy)
        first = limiter._key("key:abc", 0.0)
        later = limiter._key("key:abc", float(policy.window_seconds))
        assert first != later

    async def test_identities_are_counted_separately(
        self, policy: RateLimitPolicy
    ) -> None:
        limiter = RedisRateLimiter(fake_redis(1), policy)
        assert limiter._key("key:a", 0.0) != limiter._key("key:b", 0.0)


class TestPolicy:
    def test_description_is_human_readable(self) -> None:
        assert RateLimitPolicy(10, 30).description == "10 requests per 30s"
