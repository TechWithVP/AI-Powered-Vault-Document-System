"""Chunk persistence and vector similarity search."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, select

from app.models.chunk import DocumentChunk
from app.repositories.base import AsyncRepository, SyncRepository


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieved chunk with the scores that found it.

    A chunk may be found by vector similarity, by exact-term matching, or by
    both; the scores are kept separate so the relevance decision can reason
    about each signal rather than a single blended number.
    """

    chunk: DocumentChunk
    similarity: float = 0.0
    lexical_rank: float = 0.0

    @property
    def is_relevant(self) -> bool:
        return self.similarity > 0.0 or self.lexical_rank > 0.0

    def merged_with(self, other: "ScoredChunk") -> "ScoredChunk":
        """Combine the two scores for a chunk found by both searches."""
        return ScoredChunk(
            chunk=self.chunk,
            similarity=max(self.similarity, other.similarity),
            lexical_rank=max(self.lexical_rank, other.lexical_rank),
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
    ) -> list[ScoredChunk]:
        """Return the nearest chunks within the given documents.

        Uses pgvector cosine distance; similarity is reported as ``1 - distance``
        so callers can threshold on an intuitive 0..1 score.
        """
        if not document_ids:
            return []

        distance = DocumentChunk.embedding.cosine_distance(list(embedding))
        statement = (
            select(DocumentChunk, distance.label("distance"))
            .where(
                DocumentChunk.document_id.in_(list(document_ids)),
                DocumentChunk.embedding.is_not(None),
            )
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

    def pending_embeddings(self, document_id: uuid.UUID) -> list[DocumentChunk]:
        """Chunks still awaiting an embedding, in document order."""
        statement = (
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.embedding.is_(None),
            )
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
