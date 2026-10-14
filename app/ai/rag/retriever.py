"""Hybrid retrieval over document chunks.

Dense vectors and exact-term matching fail in opposite directions: embeddings
generalise well but dilute rare, specific terms inside heterogeneous text such
as policy schedules, while full-text search nails those terms but misses
paraphrase. Running both and fusing the results covers each other's blind spot,
and the relevance gate accepts a question if either signal is convincing.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from app.ai.base import EmbeddingProvider, TokenUsage
from app.ai.cache import EmbeddingCache, NullEmbeddingCache
from app.core.config import RagSettings
from app.core.logging import get_logger
from app.repositories.chunk import ChunkRepository, ScoredChunk

logger = get_logger(__name__)

#: Reciprocal-rank-fusion constant. The standard value; it damps the influence
#: of any single ranker's top position so one list cannot dominate the other.
RRF_K = 60

#: Weights applied to each ranker during fusion. These are equal by design.
#: Down-weighting the lexical arm is tempting - it has far lower standalone
#: recall - but it silently breaks fusion: a chunk only exact-term search
#: found enters at lex@1, and any weight below SEMANTIC_WEIGHT * (RRF_K + 1) /
#: (RRF_K + top_k) scores it under the weakest semantic hit, so it can never
#: reach the model. At the default top_k that bar sits at 0.924. Lexical's
#: lower precision is handled by the relevance gate, not by the weight.
SEMANTIC_WEIGHT = 1.0
LEXICAL_WEIGHT = 1.0

#: A single short word is too generic to justify answering on lexical evidence
#: alone, so the lexical gate needs at least this many substantial terms.
MIN_LEXICAL_TERMS = 2
MIN_TERM_LENGTH = 4

#: Terms shorter than this are rarely the distinctive part of a question.
MIN_SEARCH_TERM_LENGTH = 3

#: Matching every term is precise but brittle: one filler word the document
#: happens not to use ("the main geographical limits") sinks the whole query.
#: When that happens, retry with the few most distinctive terms - domain
#: vocabulary is reliably longer than conversational filler.
FALLBACK_TERM_COUNT = 3


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """Chunks found for a query, with the relevance decision already made."""

    results: list[ScoredChunk]
    best_similarity: float
    threshold: float
    best_lexical_rank: float = 0.0
    lexical_threshold: float = 0.0
    lexical_eligible: bool = False
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def semantic_hit(self) -> bool:
        return self.best_similarity >= self.threshold

    @property
    def lexical_hit(self) -> bool:
        """Whether the documents literally contain what was asked about."""
        return (
            self.lexical_eligible
            and self.best_lexical_rank >= self.lexical_threshold
        )

    @property
    def is_relevant(self) -> bool:
        """Whether anything retrieved justifies attempting an answer.

        Vector search always returns its nearest neighbours, even for a question
        the corpus cannot answer, so an explicit gate is what lets the system
        decline honestly instead of reasoning over noise.
        """
        return bool(self.results) and (self.semantic_hit or self.lexical_hit)


class ChunkRetriever:
    """Finds the passages most likely to answer a question."""

    def __init__(
        self,
        chunks: ChunkRepository,
        embedder: EmbeddingProvider,
        settings: RagSettings,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._chunks = chunks
        self._embedder = embedder
        self._settings = settings
        self._cache = cache or NullEmbeddingCache()

    async def retrieve(
        self,
        query: str,
        *,
        document_ids: list[uuid.UUID],
        top_k: int | None = None,
    ) -> RetrievalOutcome:
        if not document_ids:
            return self._empty()

        limit = top_k or self._settings.top_k
        pool = max(self._settings.candidate_pool, limit)
        vector, usage = await self._embed(query)
        if vector is None:
            return self._empty(usage=usage)

        # Each arm returns a deep candidate list rather than the final top_k:
        # fusion scores a chunk higher when both rankers found it, which can
        # only happen if the lists are deep enough to overlap.
        #
        # These run sequentially even though they are logically independent.
        # They share the request's AsyncSession, and SQLAlchemy does not permit
        # concurrent operations on one session - gathering them races to
        # provision the connection and raises InvalidRequestError. Two index
        # lookups cost single-digit milliseconds against an LLM call measured
        # in seconds, so a second connection is not worth the pool pressure.
        semantic = await self._chunks.search(
            vector, document_ids=document_ids, limit=pool
        )
        lexical = await self._search_lexical(
            query, document_ids=document_ids, limit=pool
        )

        fused = self._fuse(semantic, lexical, limit=limit)
        best_similarity = max((item.similarity for item in fused), default=0.0)
        best_lexical = max((item.lexical_rank for item in fused), default=0.0)

        outcome = RetrievalOutcome(
            results=fused,
            best_similarity=best_similarity,
            threshold=self._settings.min_similarity,
            best_lexical_rank=best_lexical,
            lexical_threshold=self._settings.min_lexical_rank,
            lexical_eligible=self._is_lexically_eligible(query),
            usage=usage,
        )

        logger.info(
            "Retrieval completed",
            extra={
                "documents": len(document_ids),
                "semantic_results": len(semantic),
                "lexical_results": len(lexical),
                "best_similarity": round(best_similarity, 4),
                "best_lexical_rank": round(best_lexical, 4),
                "relevant": outcome.is_relevant,
            },
        )
        return outcome

    async def _search_lexical(
        self, query: str, *, document_ids: list[uuid.UUID], limit: int
    ) -> list[ScoredChunk]:
        """Match on exact terms, relaxing to the distinctive ones if needed."""
        terms = self._significant_terms(query)
        if not terms:
            return []

        results = await self._chunks.search_lexical(
            " ".join(terms), document_ids=document_ids, limit=limit
        )
        if results or len(terms) <= FALLBACK_TERM_COUNT:
            return results

        distinctive = self._distinctive_terms(terms, FALLBACK_TERM_COUNT)
        logger.debug(
            "Retrying lexical search with distinctive terms",
            extra={"terms": distinctive},
        )
        return await self._chunks.search_lexical(
            " ".join(distinctive), document_ids=document_ids, limit=limit
        )

    @staticmethod
    def _significant_terms(query: str) -> list[str]:
        """Query words worth searching for, de-duplicated and in order."""
        seen: dict[str, None] = {}
        for raw in query.split():
            word = raw.strip(".,?!:;\"'()[]").lower()
            if len(word) >= MIN_SEARCH_TERM_LENGTH:
                seen.setdefault(word, None)
        return list(seen)

    @staticmethod
    def _distinctive_terms(terms: list[str], count: int) -> list[str]:
        """The longest terms, as a proxy for the most domain-specific ones."""
        return sorted(terms, key=len, reverse=True)[:count]

    def _empty(self, *, usage: TokenUsage | None = None) -> RetrievalOutcome:
        return RetrievalOutcome(
            results=[],
            best_similarity=0.0,
            threshold=self._settings.min_similarity,
            lexical_threshold=self._settings.min_lexical_rank,
            usage=usage or TokenUsage(),
        )

    @staticmethod
    def _is_lexically_eligible(query: str) -> bool:
        """Require enough substantial words before trusting lexical evidence."""
        terms = [
            word for word in query.split() if len(word.strip(".,?!:;")) >= MIN_TERM_LENGTH
        ]
        return len(terms) >= MIN_LEXICAL_TERMS

    @staticmethod
    def _fuse(
        semantic: list[ScoredChunk], lexical: list[ScoredChunk], *, limit: int
    ) -> list[ScoredChunk]:
        """Merge two ranked lists using weighted reciprocal rank fusion.

        RRF combines rankings without requiring the two score scales to be
        comparable, which matters because cosine similarity and ts_rank are not.
        """
        merged: dict[uuid.UUID, ScoredChunk] = {}
        scores: dict[uuid.UUID, float] = {}

        for weight, results in ((SEMANTIC_WEIGHT, semantic), (LEXICAL_WEIGHT, lexical)):
            for rank, item in enumerate(results, start=1):
                key = item.chunk.id
                existing = merged.get(key)
                merged[key] = existing.merged_with(item) if existing else item
                scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank)

        ordered = sorted(
            merged.values(), key=lambda item: scores[item.chunk.id], reverse=True
        )
        return ordered[:limit]

    async def _embed(self, query: str) -> tuple[list[float] | None, TokenUsage]:
        """Embed the query, serving a cached vector when one exists."""
        model = self._embedder.model
        cached = await self._cache.get(query, model=model)
        if cached:
            return cached, TokenUsage()

        # Embedding is a blocking HTTP call; run it off the event loop.
        # embed_documents rather than embed_query so token usage comes back
        # with the vector and the query cost can be attributed.
        embedding = await asyncio.to_thread(self._embedder.embed_documents, [query])
        if not embedding.vectors:
            return None, TokenUsage()

        vector = embedding.vectors[0]
        await self._cache.set(query, vector, model=model)
        return vector, embedding.usage
