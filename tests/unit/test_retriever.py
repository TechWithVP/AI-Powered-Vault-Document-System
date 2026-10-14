"""Retrieval and the relevance threshold."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ai.base import EmbeddingProvider, EmbeddingResult, TokenUsage
from app.ai.rag.retriever import ChunkRetriever, RetrievalOutcome
from app.core.config import RagSettings
from app.models.chunk import DocumentChunk
from app.repositories.chunk import ScoredChunk

DOCUMENT_ID = uuid.uuid4()


class StubEmbedder(EmbeddingProvider):
    """Records every text embedded and reports a fixed token usage."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    @property
    def model(self) -> str:
        return "stub"

    @property
    def dimensions(self) -> int:
        return 4

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        self.queries.extend(texts)
        return EmbeddingResult(
            vectors=[[0.1] * 4 for _ in texts],
            usage=TokenUsage(prompt_tokens=7, cost_usd=Decimal("0.0000002")),
            model="stub",
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text]).vectors[0]


def make_chunks(semantic=None, lexical=None) -> MagicMock:
    """Repository double covering both halves of hybrid retrieval."""
    repo = MagicMock()
    repo.search = AsyncMock(return_value=semantic if semantic is not None else [])
    repo.search_lexical = AsyncMock(return_value=lexical if lexical is not None else [])
    return repo


def scored(
    similarity: float, *, lexical_rank: float = 0.0, index: int = 0
) -> ScoredChunk:
    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=DOCUMENT_ID,
        chunk_index=index,
        content="text",
        token_count=5,
    )
    return ScoredChunk(
        chunk=chunk, similarity=similarity, lexical_rank=lexical_rank
    )


@pytest.fixture
def settings() -> RagSettings:
    return RagSettings(top_k=4, min_similarity=0.25)


class TestRelevanceDecision:
    def test_scores_above_the_threshold_are_relevant(self) -> None:
        outcome = RetrievalOutcome([scored(0.6)], 0.6, 0.25)
        assert outcome.is_relevant

    def test_scores_below_the_threshold_are_not(self) -> None:
        outcome = RetrievalOutcome([scored(0.1)], 0.1, 0.25)
        assert not outcome.is_relevant

    def test_a_score_exactly_at_the_threshold_is_relevant(self) -> None:
        assert RetrievalOutcome([scored(0.25)], 0.25, 0.25).is_relevant

    def test_no_results_is_never_relevant(self) -> None:
        assert not RetrievalOutcome([], 0.0, 0.25).is_relevant


class TestRetrieval:
    async def test_query_is_embedded_and_searched(
        self, settings: RagSettings
    ) -> None:
        chunks = make_chunks(semantic=[scored(0.7), scored(0.5)])
        embedder = StubEmbedder()

        outcome = await ChunkRetriever(chunks, embedder, settings).retrieve(
            "what is the limit?", document_ids=[DOCUMENT_ID]
        )

        assert embedder.queries == ["what is the limit?"]
        assert outcome.best_similarity == pytest.approx(0.7)
        assert outcome.is_relevant

    async def test_query_embedding_usage_is_reported_for_costing(
        self, settings: RagSettings
    ) -> None:
        chunks = make_chunks(semantic=[scored(0.7)])

        outcome = await ChunkRetriever(chunks, StubEmbedder(), settings).retrieve(
            "q", document_ids=[DOCUMENT_ID]
        )

        assert outcome.usage.prompt_tokens == 7
        assert outcome.usage.cost_usd > 0

    async def test_search_is_scoped_to_the_given_documents(
        self, settings: RagSettings
    ) -> None:
        chunks = make_chunks()
        other = uuid.uuid4()

        await ChunkRetriever(chunks, StubEmbedder(), settings).retrieve(
            "q", document_ids=[DOCUMENT_ID, other]
        )

        assert chunks.search.await_args.kwargs["document_ids"] == [DOCUMENT_ID, other]

    async def test_results_are_capped_at_top_k(self, settings: RagSettings) -> None:
        chunks = make_chunks(
            semantic=[scored(0.9 - index / 100) for index in range(20)]
        )

        outcome = await ChunkRetriever(chunks, StubEmbedder(), settings).retrieve(
            "q", document_ids=[DOCUMENT_ID]
        )

        assert len(outcome.results) == settings.top_k

    async def test_top_k_can_be_overridden(self, settings: RagSettings) -> None:
        chunks = make_chunks(
            semantic=[scored(0.9 - index / 100) for index in range(20)]
        )

        outcome = await ChunkRetriever(chunks, StubEmbedder(), settings).retrieve(
            "q", document_ids=[DOCUMENT_ID], top_k=10
        )

        assert len(outcome.results) == 10

    async def test_candidates_are_fetched_deeper_than_top_k(
        self, settings: RagSettings
    ) -> None:
        """Fusion rewards agreement, which needs lists deep enough to overlap.

        Fetching only top_k from each arm leaves the two lists disjoint, and
        the hybrid silently degrades into semantic search with a lexical tail.
        """
        chunks = make_chunks()

        await ChunkRetriever(chunks, StubEmbedder(), settings).retrieve(
            "geographical limits", document_ids=[DOCUMENT_ID]
        )

        assert chunks.search.await_args.kwargs["limit"] == settings.candidate_pool
        assert (
            chunks.search_lexical.await_args.kwargs["limit"]
            == settings.candidate_pool
        )
        assert settings.candidate_pool > settings.top_k

    async def test_a_lexical_only_hit_can_reach_the_results(
        self, settings: RagSettings
    ) -> None:
        """Regression: exact-term hits used to be fused out of the top_k.

        With both arms truncated to top_k and lexical weighted below semantic,
        a lexical rank-1 chunk scored below a semantic rank-k chunk, so a
        passage only exact-term search could find never reached the model.
        """
        wanted = scored(0.0, lexical_rank=0.5, index=99)
        chunks = make_chunks(
            semantic=[scored(0.9 - i / 100, index=i) for i in range(settings.top_k)],
            lexical=[wanted],
        )

        outcome = await ChunkRetriever(chunks, StubEmbedder(), settings).retrieve(
            "geographical limits", document_ids=[DOCUMENT_ID]
        )

        assert 99 in {item.chunk.chunk_index for item in outcome.results}

    async def test_an_empty_scope_skips_the_database_entirely(
        self, settings: RagSettings
    ) -> None:
        chunks = make_chunks()
        embedder = StubEmbedder()

        outcome = await ChunkRetriever(chunks, embedder, settings).retrieve(
            "q", document_ids=[]
        )

        assert not outcome.is_relevant
        chunks.search.assert_not_awaited()
        assert embedder.queries == []
