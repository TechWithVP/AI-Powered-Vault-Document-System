"""pgvector similarity search against the real database.

Cosine distance, ordering and document scoping are enforced by Postgres, so
these can only be verified with a live pgvector instance.
"""

from __future__ import annotations

import math
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import DocumentChunk
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from tests.integration.conftest import make_document

pytestmark = pytest.mark.integration

WIDTH = 1536


def unit_vector(*leading: float) -> list[float]:
    """A normalised vector whose first components are the given values."""
    values = list(leading) + [0.0] * (WIDTH - len(leading))
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


async def add_chunk(
    db: AsyncSession, document_id, index: int, text: str, embedding: list[float]
) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=index,
        content=text,
        page_number=index + 1,
        token_count=10,
        embedding=embedding,
    )
    db.add(chunk)
    await db.flush()
    return chunk


class TestSimilaritySearch:
    async def test_the_closest_chunk_ranks_first(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        await add_chunk(db, document.id, 0, "about cats", unit_vector(1.0, 0.0))
        await add_chunk(db, document.id, 1, "about dogs", unit_vector(0.0, 1.0))

        results = await ChunkRepository(db).search(
            unit_vector(1.0, 0.0), document_ids=[document.id], limit=5
        )

        assert results[0].chunk.content == "about cats"

    async def test_an_identical_vector_scores_near_one(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        embedding = unit_vector(0.6, 0.8)
        await add_chunk(db, document.id, 0, "exact match", embedding)

        results = await ChunkRepository(db).search(
            embedding, document_ids=[document.id], limit=1
        )

        assert results[0].similarity == pytest.approx(1.0, abs=1e-4)

    async def test_an_orthogonal_vector_scores_near_zero(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        await add_chunk(db, document.id, 0, "unrelated", unit_vector(0.0, 1.0))

        results = await ChunkRepository(db).search(
            unit_vector(1.0, 0.0), document_ids=[document.id], limit=1
        )

        assert results[0].similarity == pytest.approx(0.0, abs=1e-4)

    async def test_results_are_ordered_by_descending_similarity(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        await add_chunk(db, document.id, 0, "near", unit_vector(0.9, 0.1))
        await add_chunk(db, document.id, 1, "mid", unit_vector(0.5, 0.5))
        await add_chunk(db, document.id, 2, "far", unit_vector(0.0, 1.0))

        results = await ChunkRepository(db).search(
            unit_vector(1.0, 0.0), document_ids=[document.id], limit=5
        )

        scores = [result.similarity for result in results]
        assert scores == sorted(scores, reverse=True)

    async def test_limit_caps_the_result_count(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        for index in range(5):
            await add_chunk(
                db, document.id, index, f"chunk {index}", unit_vector(1.0, index / 10)
            )

        results = await ChunkRepository(db).search(
            unit_vector(1.0, 0.0), document_ids=[document.id], limit=2
        )
        assert len(results) == 2

    async def test_page_numbers_survive_retrieval_for_citations(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        await add_chunk(db, document.id, 4, "cited passage", unit_vector(1.0))

        results = await ChunkRepository(db).search(
            unit_vector(1.0), document_ids=[document.id], limit=1
        )

        assert results[0].chunk.page_number == 5


class TestScoping:
    async def test_search_never_crosses_into_other_documents(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        mine = documents.add(make_document(user_id, filename="mine.pdf"))
        other = documents.add(make_document(user_id, filename="other.pdf"))
        await documents.flush()

        await add_chunk(db, mine.id, 0, "in scope", unit_vector(1.0, 0.0))
        await add_chunk(db, other.id, 0, "out of scope", unit_vector(1.0, 0.0))

        results = await ChunkRepository(db).search(
            unit_vector(1.0, 0.0), document_ids=[mine.id], limit=10
        )

        assert len(results) == 1
        assert results[0].chunk.content == "in scope"

    async def test_multi_document_scope_searches_all_of_them(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        first = documents.add(make_document(user_id, filename="a.pdf"))
        second = documents.add(make_document(user_id, filename="b.pdf"))
        await documents.flush()

        await add_chunk(db, first.id, 0, "from a", unit_vector(1.0, 0.0))
        await add_chunk(db, second.id, 0, "from b", unit_vector(0.9, 0.1))

        results = await ChunkRepository(db).search(
            unit_vector(1.0, 0.0), document_ids=[first.id, second.id], limit=10
        )

        assert len(results) == 2

    async def test_an_empty_scope_returns_nothing(self, db: AsyncSession) -> None:
        assert await ChunkRepository(db).search(
            unit_vector(1.0), document_ids=[], limit=5
        ) == []

    async def test_chunks_without_an_embedding_are_skipped(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        await add_chunk(db, document.id, 0, "embedded", unit_vector(1.0))
        db.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=1,
                content="not yet embedded",
                page_number=2,
                token_count=5,
                embedding=None,
            )
        )
        await db.flush()

        results = await ChunkRepository(db).search(
            unit_vector(1.0), document_ids=[document.id], limit=10
        )

        assert [result.chunk.content for result in results] == ["embedded"]


class TestLexicalSearch:
    """Full-text search over chunks, backed by the generated tsvector column."""

    async def test_an_exact_term_is_found(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        await add_chunk(
            db,
            document.id,
            0,
            "1.4 Geographical Limits India (no location barrier)",
            unit_vector(0.0, 1.0),
        )
        await add_chunk(db, document.id, 1, "Unrelated administrative text", unit_vector(1.0))

        results = await ChunkRepository(db).search_lexical(
            "Geographical Limits", document_ids=[document.id], limit=5
        )

        assert len(results) == 1
        assert "India" in results[0].chunk.content
        assert results[0].lexical_rank > 0

    async def test_every_term_must_be_present(
        self, db: AsyncSession, user_id: str
    ) -> None:
        """AND semantics keep lexical evidence high-precision."""
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        await add_chunk(db, document.id, 0, "Geographical limits apply", unit_vector(1.0))

        results = await ChunkRepository(db).search_lexical(
            "geographical limits submarine", document_ids=[document.id], limit=5
        )
        assert results == []

    async def test_matching_is_stem_aware(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        await add_chunk(
            db, document.id, 0, "The insured person is covered", unit_vector(1.0)
        )

        results = await ChunkRepository(db).search_lexical(
            "insurance covering", document_ids=[document.id], limit=5
        )
        assert len(results) == 1

    async def test_search_is_scoped_to_the_given_documents(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        mine = documents.add(make_document(user_id, filename="mine.pdf"))
        other = documents.add(make_document(user_id, filename="other.pdf"))
        await documents.flush()

        await add_chunk(db, mine.id, 0, "Geographical limits apply here", unit_vector(1.0))
        await add_chunk(db, other.id, 0, "Geographical limits apply there", unit_vector(1.0))

        results = await ChunkRepository(db).search_lexical(
            "geographical limits", document_ids=[mine.id], limit=5
        )

        assert len(results) == 1
        assert "here" in results[0].chunk.content

    async def test_an_empty_query_returns_nothing(self, db: AsyncSession) -> None:
        assert await ChunkRepository(db).search_lexical(
            "   ", document_ids=[uuid.uuid4()], limit=5
        ) == []


class TestEmbeddingModelScoping:
    """Vectors from a different model must not be compared against.

    Nothing about a stored vector reveals which model produced it, so without
    an explicit filter a model change turns every score into confident
    nonsense rather than an error.
    """

    async def test_search_ignores_vectors_from_another_model(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        current = await add_chunk(
            db, document.id, 0, "current model", unit_vector(1.0, 0.0)
        )
        stale = await add_chunk(
            db, document.id, 1, "older model", unit_vector(1.0, 0.0)
        )
        current.embedding_model = "text-embedding-3-small"
        stale.embedding_model = "text-embedding-ada-002"
        await db.flush()

        results = await ChunkRepository(db).search(
            unit_vector(1.0, 0.0),
            document_ids=[document.id],
            limit=5,
            model="text-embedding-3-small",
        )

        assert [item.chunk.content for item in results] == ["current model"]

    async def test_omitting_the_model_compares_every_vector(
        self, db: AsyncSession, user_id: str
    ) -> None:
        """The filter is opt-in, so existing callers keep their behaviour."""
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        chunk = await add_chunk(db, document.id, 0, "any model", unit_vector(1.0))
        chunk.embedding_model = "text-embedding-ada-002"
        await db.flush()

        results = await ChunkRepository(db).search(
            unit_vector(1.0), document_ids=[document.id], limit=5
        )

        assert len(results) == 1

    async def test_stale_vectors_are_counted(
        self, db: AsyncSession, user_id: str
    ) -> None:
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        fresh = await add_chunk(db, document.id, 0, "fresh", unit_vector(1.0))
        stale = await add_chunk(db, document.id, 1, "stale", unit_vector(0.0, 1.0))
        fresh.embedding_model = "text-embedding-3-small"
        stale.embedding_model = "text-embedding-ada-002"
        await db.flush()

        counted = await ChunkRepository(db).count_stale("text-embedding-3-small")

        assert counted == 1

    async def test_a_null_model_counts_as_stale(
        self, db: AsyncSession, user_id: str
    ) -> None:
        """Rows predating the column cannot be trusted either."""
        documents = DocumentRepository(db)
        document = documents.add(make_document(user_id))
        await documents.flush()

        await add_chunk(db, document.id, 0, "unlabelled", unit_vector(1.0))

        counted = await ChunkRepository(db).count_stale("text-embedding-3-small")

        assert counted == 1
