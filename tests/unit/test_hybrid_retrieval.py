"""Hybrid retrieval: fusion, the lexical fallback, and the relevance gate."""

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
    @property
    def model(self) -> str:
        return "stub"

    @property
    def dimensions(self) -> int:
        return 4

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[[0.1] * 4 for _ in texts],
            usage=TokenUsage(prompt_tokens=5, cost_usd=Decimal("0.000001")),
            model="stub",
        )

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 4


def chunk(label: str) -> DocumentChunk:
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=DOCUMENT_ID,
        chunk_index=0,
        content=label,
        token_count=5,
    )


def make_repo(semantic=None, lexical=None, lexical_calls=None) -> MagicMock:
    repo = MagicMock()
    repo.search = AsyncMock(return_value=semantic or [])
    if lexical_calls is not None:
        repo.search_lexical = AsyncMock(side_effect=lexical_calls)
    else:
        repo.search_lexical = AsyncMock(return_value=lexical or [])
    return repo


@pytest.fixture
def settings() -> RagSettings:
    return RagSettings(top_k=5, min_similarity=0.25, min_lexical_rank=0.02)


class TestRelevanceGate:
    def test_a_strong_vector_match_is_relevant(self, settings: RagSettings) -> None:
        outcome = RetrievalOutcome(
            results=[ScoredChunk(chunk("a"), similarity=0.6)],
            best_similarity=0.6,
            threshold=0.25,
        )
        assert outcome.is_relevant

    def test_a_weak_vector_match_alone_is_not(self, settings: RagSettings) -> None:
        outcome = RetrievalOutcome(
            results=[ScoredChunk(chunk("a"), similarity=0.1)],
            best_similarity=0.1,
            threshold=0.25,
        )
        assert not outcome.is_relevant

    def test_an_exact_term_match_rescues_a_weak_vector_score(self) -> None:
        """The measured failure: a fact in a table scores 0.199 semantically."""
        outcome = RetrievalOutcome(
            results=[ScoredChunk(chunk("a"), similarity=0.199, lexical_rank=0.11)],
            best_similarity=0.199,
            threshold=0.25,
            best_lexical_rank=0.11,
            lexical_threshold=0.02,
            lexical_eligible=True,
        )
        assert outcome.lexical_hit
        assert outcome.is_relevant

    def test_a_single_generic_word_cannot_rescue_on_lexical_evidence(self) -> None:
        outcome = RetrievalOutcome(
            results=[ScoredChunk(chunk("a"), similarity=0.1, lexical_rank=0.9)],
            best_similarity=0.1,
            threshold=0.25,
            best_lexical_rank=0.9,
            lexical_threshold=0.02,
            lexical_eligible=False,
        )
        assert not outcome.is_relevant

    def test_nothing_retrieved_is_never_relevant(self) -> None:
        outcome = RetrievalOutcome(results=[], best_similarity=0.0, threshold=0.25)
        assert not outcome.is_relevant


class TestFusion:
    async def test_both_searches_contribute_results(
        self, settings: RagSettings
    ) -> None:
        semantic_only = ScoredChunk(chunk("semantic"), similarity=0.5)
        lexical_only = ScoredChunk(chunk("lexical"), lexical_rank=0.3)
        repo = make_repo(semantic=[semantic_only], lexical=[lexical_only])

        outcome = await ChunkRetriever(repo, StubEmbedder(), settings).retrieve(
            "geographical limits of coverage", document_ids=[DOCUMENT_ID]
        )

        contents = {item.chunk.content for item in outcome.results}
        assert contents == {"semantic", "lexical"}

    async def test_a_chunk_found_by_both_keeps_both_scores(
        self, settings: RagSettings
    ) -> None:
        shared = chunk("shared")
        repo = make_repo(
            semantic=[ScoredChunk(shared, similarity=0.42)],
            lexical=[ScoredChunk(shared, lexical_rank=0.31)],
        )

        outcome = await ChunkRetriever(repo, StubEmbedder(), settings).retrieve(
            "geographical limits of coverage", document_ids=[DOCUMENT_ID]
        )

        assert len(outcome.results) == 1
        assert outcome.results[0].similarity == pytest.approx(0.42)
        assert outcome.results[0].lexical_rank == pytest.approx(0.31)

    async def test_results_are_capped_at_the_limit(
        self, settings: RagSettings
    ) -> None:
        repo = make_repo(
            semantic=[ScoredChunk(chunk(f"s{i}"), similarity=0.4) for i in range(5)],
            lexical=[ScoredChunk(chunk(f"l{i}"), lexical_rank=0.2) for i in range(5)],
        )

        outcome = await ChunkRetriever(repo, StubEmbedder(), settings).retrieve(
            "a reasonably specific question", document_ids=[DOCUMENT_ID], top_k=3
        )

        assert len(outcome.results) == 3


class TestLexicalFallback:
    async def test_all_terms_are_tried_first(self, settings: RagSettings) -> None:
        repo = make_repo(lexical=[ScoredChunk(chunk("hit"), lexical_rank=0.2)])

        await ChunkRetriever(repo, StubEmbedder(), settings).retrieve(
            "what is the main geographical limits we have",
            document_ids=[DOCUMENT_ID],
        )

        assert repo.search_lexical.await_count == 1

    async def test_a_miss_retries_with_the_distinctive_terms(
        self, settings: RagSettings
    ) -> None:
        """One filler word the document lacks must not sink the whole query."""
        repo = make_repo(
            lexical_calls=[[], [ScoredChunk(chunk("hit"), lexical_rank=0.11)]]
        )

        outcome = await ChunkRetriever(repo, StubEmbedder(), settings).retrieve(
            "what is the main geographical limits we have",
            document_ids=[DOCUMENT_ID],
        )

        assert repo.search_lexical.await_count == 2
        retried = repo.search_lexical.await_args_list[1].args[0]
        assert "geographical" in retried
        assert outcome.best_lexical_rank == pytest.approx(0.11)

    async def test_a_short_query_is_not_retried(self, settings: RagSettings) -> None:
        repo = make_repo(lexical=[])

        await ChunkRetriever(repo, StubEmbedder(), settings).retrieve(
            "geographical limits", document_ids=[DOCUMENT_ID]
        )

        assert repo.search_lexical.await_count == 1

    def test_distinctive_terms_are_the_longest_ones(self) -> None:
        terms = ChunkRetriever._significant_terms(
            "what is the main geographical limits we have"
        )
        assert ChunkRetriever._distinctive_terms(terms, 2) == [
            "geographical",
            "limits",
        ]

    def test_short_words_are_not_searched_for(self) -> None:
        assert "we" not in ChunkRetriever._significant_terms("what do we have")

    def test_terms_are_deduplicated(self) -> None:
        terms = ChunkRetriever._significant_terms("limits limits LIMITS clause")
        assert terms.count("limits") == 1
