"""Reranking: ordering, and the guards that stop it making things worse."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.ai.base import ChatMessage, StructuredResult, TokenUsage
from app.ai.rag.rerank import (
    MIN_SCORE_COVERAGE,
    LLMReranker,
    NullReranker,
    build_reranker,
)
from app.core.config import RagSettings
from app.models.chunk import DocumentChunk
from app.repositories.chunk import ScoredChunk

DOCUMENT_ID = uuid.uuid4()


def candidate(index: int) -> ScoredChunk:
    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=DOCUMENT_ID,
        chunk_index=index,
        content=f"passage {index}",
        token_count=5,
    )
    return ScoredChunk(chunk=chunk, similarity=0.5)


def candidates(count: int) -> list[ScoredChunk]:
    return [candidate(index) for index in range(count)]


class StubLLM:
    """Returns a canned structured response, or raises."""

    def __init__(self, scores: list[dict[str, int]] | None = None, error: Exception | None = None):
        self._scores = scores or []
        self._error = error
        self.calls: list[Sequence[ChatMessage]] = []

    @property
    def model(self) -> str:
        return "stub"

    def complete_structured(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
    ) -> StructuredResult:
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        return StructuredResult(
            data={"scores": self._scores},
            usage=TokenUsage(prompt_tokens=100, cost_usd=Decimal("0.0001")),
            model="stub",
        )


def indexes(results: list[ScoredChunk]) -> list[int]:
    return [item.chunk.chunk_index for item in results]


@pytest.mark.asyncio
class TestNullReranker:
    async def test_the_input_order_is_kept(self) -> None:
        items = candidates(4)

        outcome = await NullReranker().rerank("q", items, limit=3)

        assert indexes(outcome.results) == [0, 1, 2]
        assert outcome.applied is False


@pytest.mark.asyncio
class TestOrdering:
    async def test_candidates_are_ordered_by_score(self) -> None:
        llm = StubLLM([
            {"position": 1, "score": 2},
            {"position": 2, "score": 9},
            {"position": 3, "score": 5},
        ])

        outcome = await LLMReranker(llm).rerank("q", candidates(3), limit=3)

        assert indexes(outcome.results) == [1, 2, 0]
        assert outcome.applied is True

    async def test_equal_scores_fall_back_to_retrieval_order(self) -> None:
        llm = StubLLM([
            {"position": 1, "score": 7},
            {"position": 2, "score": 7},
            {"position": 3, "score": 7},
        ])

        outcome = await LLMReranker(llm).rerank("q", candidates(3), limit=3)

        assert indexes(outcome.results) == [0, 1, 2]

    async def test_a_zero_score_drops_the_candidate(self) -> None:
        llm = StubLLM([
            {"position": 1, "score": 0},
            {"position": 2, "score": 8},
            {"position": 3, "score": 4},
        ])

        outcome = await LLMReranker(llm).rerank("q", candidates(3), limit=3)

        assert indexes(outcome.results) == [1, 2]

    async def test_results_are_capped_at_the_limit(self) -> None:
        llm = StubLLM([{"position": n, "score": 10 - n} for n in range(1, 6)])

        outcome = await LLMReranker(llm).rerank("q", candidates(5), limit=2)

        assert len(outcome.results) == 2

    async def test_usage_is_reported_so_the_call_can_be_costed(self) -> None:
        llm = StubLLM([{"position": 1, "score": 5}, {"position": 2, "score": 5}])

        outcome = await LLMReranker(llm).rerank("q", candidates(2), limit=2)

        assert outcome.usage.cost_usd == Decimal("0.0001")


@pytest.mark.asyncio
class TestGuards:
    """A reranker is an optimisation; it must never degrade the result."""

    async def test_a_provider_failure_keeps_the_fused_order(self) -> None:
        llm = StubLLM(error=RuntimeError("provider exploded"))

        outcome = await LLMReranker(llm).rerank("q", candidates(4), limit=3)

        assert indexes(outcome.results) == [0, 1, 2]
        assert outcome.applied is False

    async def test_partial_scores_are_rejected_rather_than_half_applied(self) -> None:
        """Measured behaviour: asked to score 40 passages the model returned 5.

        Placing the other 35 by an opinion it never gave produces a different
        ranking, not a weaker one, so the retrieval order stands.
        """
        llm = StubLLM([{"position": 1, "score": 1}, {"position": 2, "score": 10}])

        outcome = await LLMReranker(llm).rerank("q", candidates(10), limit=4)

        assert indexes(outcome.results) == [0, 1, 2, 3]
        assert outcome.applied is False

    async def test_coverage_at_the_threshold_is_applied(self) -> None:
        count = 10
        scored = int(count * MIN_SCORE_COVERAGE)
        llm = StubLLM(
            [{"position": n, "score": n} for n in range(1, scored + 1)]
        )

        outcome = await LLMReranker(llm).rerank("q", candidates(count), limit=3)

        assert outcome.applied is True
        # Highest score first, then the unjudged tail in retrieval order.
        assert indexes(outcome.results)[0] == scored - 1

    async def test_unjudged_candidates_rank_behind_judged_ones(self) -> None:
        """An absent score must not promote a passage above a judged one."""
        count = 10
        llm = StubLLM(
            [{"position": n, "score": 2} for n in range(1, count)]
            # position 10 deliberately omitted, coverage 9/10 = threshold
        )

        outcome = await LLMReranker(llm).rerank("q", candidates(count), limit=count)

        assert outcome.applied is True
        assert indexes(outcome.results)[-1] == count - 1

    async def test_scores_outside_the_candidate_range_are_ignored(self) -> None:
        llm = StubLLM([
            {"position": 1, "score": 5},
            {"position": 2, "score": 5},
            {"position": 99, "score": 10},
        ])

        outcome = await LLMReranker(llm).rerank("q", candidates(2), limit=2)

        assert indexes(outcome.results) == [0, 1]

    async def test_a_single_candidate_skips_the_call_entirely(self) -> None:
        llm = StubLLM([{"position": 1, "score": 9}])

        outcome = await LLMReranker(llm).rerank("q", candidates(1), limit=6)

        assert llm.calls == []
        assert indexes(outcome.results) == [0]

    async def test_long_candidates_are_truncated_with_a_marker(self) -> None:
        reranker = LLMReranker(StubLLM(), max_candidate_chars=20)
        long = candidate(0)
        long.chunk.content = "word " * 50

        rendered = reranker._render(long)

        assert len(rendered) <= 23
        assert rendered.endswith("...")


class TestFactory:
    def test_reranking_is_off_by_default(self) -> None:
        """Measured: it does not earn its latency on the labelled corpus."""
        assert isinstance(build_reranker(RagSettings(), MagicMock()), NullReranker)

    def test_enabling_it_selects_the_llm_reranker(self) -> None:
        settings = RagSettings(rerank_enabled=True)

        assert isinstance(build_reranker(settings, MagicMock()), LLMReranker)
