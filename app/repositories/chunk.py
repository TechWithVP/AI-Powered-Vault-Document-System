"""Chunk persistence and vector similarity search."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, or_, select

from app.models.chunk import DocumentChunk
from app.repositories.base import AsyncRepository, SyncRepository


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieved chunk with the scores that found it.

    A chunk may be found by vector similarity, by exact-term matching, or by
    both; the scores are kept separate so the relevance decision can reason
    about each signal rather than a single blended number.

    ``similarity`` and ``lexical_rank`` are the raw scores each arm returned.
    ``rrf_semantic`` and ``rrf_lexical`` are what fusion made of them, and are
    on a single comparable scale - which is what makes it meaningful to ask how
    much of a result each arm is responsible for. They stay zero until fusion
    runs.
    """

    chunk: DocumentChunk
    similarity: float = 0.0
    lexical_rank: float = 0.0
    rrf_semantic: float = 0.0
    rrf_lexical: float = 0.0

    @property
    def is_relevant(self) -> bool:
        return self.similarity > 0.0 or self.lexical_rank > 0.0

    @property
    def rrf_total(self) -> float:
        return self.rrf_semantic + self.rrf_lexical

    @property
    def semantic_share(self) -> float:
        """Fraction of this chunk's fused score contributed by vector search.

        Zero when fusion has not run, which is indistinguishable from a purely
        lexical hit - callers that care should check ``rrf_total`` first.
        """
        total = self.rrf_total
        return self.rrf_semantic / total if total else 0.0

    @property
    def found_by(self) -> str:
        """Which arm or arms produced this chunk."""
        if self.rrf_semantic and self.rrf_lexical:
            return "both"
        if self.rrf_lexical:
            return "lexical"
        return "semantic"

    def merged_with(self, other: "ScoredChunk") -> "ScoredChunk":
        """Combine the two scores for a chunk found by both searches."""
        return ScoredChunk(
            chunk=self.chunk,
            similarity=max(self.similarity, other.similarity),
            lexical_rank=max(self.lexical_rank, other.lexical_rank),
            rrf_semantic=max(self.rrf_semantic, other.rrf_semantic),
            rrf_lexical=max(self.rrf_lexical, other.rrf_lexical),
        )

    def with_fusion(self, *, semantic: float, lexical: float) -> "ScoredChunk":
        """A copy carrying the RRF contribution of each arm."""
        return ScoredChunk(
            chunk=self.chunk,
            similarity=self.similarity,
            lexical_rank=self.lexical_rank,
            rrf_semantic=semantic,
            rrf_lexical=lexical,
        )


class ChunkRepository(AsyncRepository[DocumentChunk]):
    """Async retrieval over the pgvector index."""

    model = DocumentChunk

    async def search(
        self,
        embedding: Sequence[float],
        *,
        document_ids: Sequence[uuid.UUID],
        limit: int,
        model: str | None = None,
    ) -> list[ScoredChunk]:
        """Return the nearest chunks within the given documents.

        Uses pgvector cosine distance; similarity is reported as ``1 - distance``
        so callers can threshold on an intuitive 0..1 score.

        When ``model`` is given, only vectors produced by that model are
        searched. Embeddings from different models occupy unrelated spaces, so
        comparing across them yields confident nonsense rather than an error.
        Skipping them degrades to finding nothing, which the relevance gate
        reports honestly.
        """
        if not document_ids:
            return []

        distance = DocumentChunk.embedding.cosine_distance(list(embedding))
        filters = [
            DocumentChunk.document_id.in_(list(document_ids)),
            DocumentChunk.embedding.is_not(None),
        ]
        if model is not None:
            filters.append(DocumentChunk.embedding_model == model)

        statement = (
            select(DocumentChunk, distance.label("distance"))
            .where(*filters)
            .order_by(distance)
            .limit(limit)
        )

        rows = await self.session.execute(statement)
        return [
            ScoredChunk(chunk=chunk, similarity=1.0 - float(dist))
            for chunk, dist in rows.all()
        ]

    async def search_lexical(
        self,
        query: str,
        *,
        document_ids: Sequence[uuid.UUID],
        limit: int,
    ) -> list[ScoredChunk]:
        """Return chunks containing every significant term in the query.

        ``plainto_tsquery`` ANDs the query terms, which makes this deliberately
        high-precision: it fires when the document literally contains what was
        asked about, covering the rare-term case dense vectors dilute.
        """
        if not document_ids or not query.strip():
            return []

        tsquery = func.plainto_tsquery("english", query)
        rank = func.ts_rank(DocumentChunk.search_vector, tsquery)

        statement = (
            select(DocumentChunk, rank.label("rank"))
            .where(
                DocumentChunk.document_id.in_(list(document_ids)),
                DocumentChunk.search_vector.op("@@")(tsquery),
            )
            .order_by(rank.desc())
            .limit(limit)
        )

        rows = await self.session.execute(statement)
        return [
            ScoredChunk(chunk=chunk, lexical_rank=float(value))
            for chunk, value in rows.all()
        ]

    async def count_stale(self, model: str) -> int:
        """Embedded chunks whose vector came from some other model.

        These are invisible to retrieval, so the corpus silently shrinks
        rather than failing; the health report exists to make that visible.
        """
        statement = select(DocumentChunk).where(
            DocumentChunk.embedding.is_not(None),
            DocumentChunk.embedding_model.is_distinct_from(model),
        )
        return await self.count(statement)

    async def count_for_document(self, document_id: uuid.UUID) -> int:
        statement = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id
        )
        return await self.count(statement)


class SyncChunkRepository(SyncRepository[DocumentChunk]):
    """Sync chunk writes performed by the ingestion worker."""

    model = DocumentChunk

    def replace_for_document(
        self, document_id: uuid.UUID, chunks: Sequence[DocumentChunk]
    ) -> int:
        """Swap in a fresh chunk set, making re-processing idempotent."""
        self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        self.session.flush()
        self.add_all(chunks)
        self.session.flush()
        return len(chunks)

    def pending_embeddings(
        self, document_id: uuid.UUID, *, model: str | None = None
    ) -> list[DocumentChunk]:
        """Chunks still awaiting a usable embedding, in document order.

        A chunk embedded by a different model counts as pending: its vector
        cannot be compared against the current model's output, so it has to be
        redone rather than left to silently poison retrieval.
        """
        needs_embedding = DocumentChunk.embedding.is_(None)
        if model is not None:
            needs_embedding = or_(
                needs_embedding, DocumentChunk.embedding_model != model
            )

        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id, needs_embedding)
            .order_by(DocumentChunk.chunk_index)
        )
        return self.list(statement)


    def ordered_for_document(self, document_id: uuid.UUID) -> list[DocumentChunk]:
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return self.list(statement)
