"""Repository behaviour against a real PostgreSQL database.

These exercise the SQL itself - filters, joins, JSONB containment and pgvector
similarity - which unit tests with a mocked session cannot verify.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import DocumentCategory, DocumentStatus, Sentiment
from app.models.chunk import DocumentChunk
from app.models.document import DocumentInsight
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentFilter, DocumentRepository
from tests.integration.conftest import make_document

pytestmark = pytest.mark.integration

EMBEDDING_WIDTH = 1536


def vector(seed: float) -> list[float]:
    """A deterministic vector for similarity tests."""
    return [seed] + [0.0] * (EMBEDDING_WIDTH - 1)


def insight(document_id, *, tags, category=DocumentCategory.LEGAL) -> DocumentInsight:
    return DocumentInsight(
        document_id=document_id,
        summary="A policy document.",
        key_points=[],
        tags=tags,
        entities=[],
        category=category,
        sentiment=Sentiment.NEUTRAL,
        language="en",
        model="test-model",
    )


class TestOwnershipScoping:
    async def test_a_document_is_found_by_its_owner(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        document = repo.add(make_document(user_id))
        await repo.flush()

        found = await repo.get_for_user(document.id, user_id)
        assert found is not None
        assert found.filename == "policy.pdf"

    async def test_another_users_document_is_invisible(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        document = repo.add(make_document(user_id))
        await repo.flush()

        assert await repo.get_for_user(document.id, "someone-else") is None

    async def test_listing_is_scoped_to_the_owner(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        for name in ("first.pdf", "second.pdf", "third.pdf"):
            repo.add(make_document(user_id, filename=name))
        repo.add(make_document("other-user", filename="theirs.pdf"))
        await repo.flush()

        page, total = await repo.list_page(
            DocumentFilter(user_id=user_id), limit=10, offset=0
        )

        assert total == 3
        assert "theirs.pdf" not in {document.filename for document in page}

    async def test_many_documents_are_fetched_scoped_to_the_owner(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        mine = [repo.add(make_document(user_id, filename=f"m{i}.pdf")) for i in range(2)]
        theirs = repo.add(make_document("other", filename="theirs.pdf"))
        await repo.flush()

        found = await repo.get_many_for_user(
            [document.id for document in mine] + [theirs.id], user_id
        )
        assert len(found) == 2


class TestFiltering:
    async def test_status_filter_is_applied(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        repo.add(make_document(user_id, status=DocumentStatus.READY))
        repo.add(make_document(user_id, filename="b.pdf", status=DocumentStatus.FAILED))
        await repo.flush()

        page, total = await repo.list_page(
            DocumentFilter(user_id=user_id, status=DocumentStatus.FAILED),
            limit=10,
            offset=0,
        )

        assert total == 1
        assert page[0].status is DocumentStatus.FAILED

    async def test_pagination_splits_the_result_set(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        for index in range(5):
            repo.add(make_document(user_id, filename=f"doc{index}.pdf"))
        await repo.flush()

        page, total = await repo.list_page(
            DocumentFilter(user_id=user_id), limit=2, offset=0
        )

        assert total == 5
        assert len(page) == 2

    async def test_search_matches_the_filename(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        repo.add(make_document(user_id, filename="retention-policy.pdf"))
        repo.add(make_document(user_id, filename="invoice.pdf"))
        await repo.flush()

        page, total = await repo.list_page(
            DocumentFilter(user_id=user_id, search="retention"), limit=10, offset=0
        )

        assert total == 1
        assert page[0].filename == "retention-policy.pdf"

    async def test_tag_filter_uses_jsonb_containment(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        tagged = repo.add(make_document(user_id, filename="tagged.pdf"))
        plain = repo.add(make_document(user_id, filename="plain.pdf"))
        await repo.flush()

        db.add(insight(tagged.id, tags=["data-retention", "legal"]))
        db.add(
            insight(
                plain.id, tags=["marketing"], category=DocumentCategory.MARKETING
            )
        )
        await db.flush()

        page, total = await repo.list_page(
            DocumentFilter(user_id=user_id, tag="data-retention"), limit=10, offset=0
        )

        assert total == 1
        assert page[0].filename == "tagged.pdf"

    async def test_category_filter_joins_insights(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        document = repo.add(make_document(user_id))
        await repo.flush()
        db.add(insight(document.id, tags=[]))
        await db.flush()

        _, total = await repo.list_page(
            DocumentFilter(user_id=user_id, category=DocumentCategory.LEGAL),
            limit=10,
            offset=0,
        )
        assert total == 1


class TestDeduplication:
    async def test_duplicate_content_is_detected_per_user(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        digest = uuid.uuid4().hex
        repo.add(make_document(user_id, content_hash=digest))
        await repo.flush()

        assert await repo.find_by_content_hash(digest, user_id) is not None
        assert await repo.find_by_content_hash(digest, "other-user") is None

    async def test_failed_documents_do_not_block_a_re_upload(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        digest = uuid.uuid4().hex
        repo.add(
            make_document(user_id, content_hash=digest, status=DocumentStatus.FAILED)
        )
        await repo.flush()

        assert await repo.find_by_content_hash(digest, user_id) is None


class TestCascades:
    async def test_deleting_a_document_removes_its_chunks(
        self, db: AsyncSession, user_id: str
    ) -> None:
        repo = DocumentRepository(db)
        document = repo.add(make_document(user_id))
        await repo.flush()

        db.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=0,
                content="Some content here.",
                page_number=1,
                token_count=4,
                embedding=vector(0.5),
            )
        )
        await db.flush()

        await repo.delete(document)
        await db.flush()

        assert await ChunkRepository(db).count_for_document(document.id) == 0
