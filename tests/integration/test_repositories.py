"""Repository behaviour against a real PostgreSQL database.

These exercise the SQL itself - filters, joins, JSONB containment and pgvector
similarity - which unit tests with a mocked session cannot verify.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import DocumentCategory, DocumentStatus, Sentiment
from app.models.chat import ChatSession
from app.models.chunk import DocumentChunk
from app.models.document import DocumentInsight
from app.ai.rag.retriever import RetrievalOutcome
from app.repositories.chunk import ChunkRepository, ScoredChunk
from app.repositories.document import DocumentFilter, DocumentRepository
from app.repositories.metrics import RetrievalMetricRepository
from tests.integration.conftest import make_chat_session, make_document

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


class TestRetrievalMetricRepository:
    """Per-search arm attribution, recorded and aggregated.

    Scoped to a session per test: the aggregates are deliberately global, so
    without a scope these would also count rows the running application wrote.
    """

    @staticmethod
    def outcome(
        *, semantic: float, lexical: float, relevant: bool = True
    ) -> RetrievalOutcome:
        chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            chunk_index=0,
            content="text",
            token_count=5,
        )
        scored = ScoredChunk(
            chunk=chunk,
            similarity=0.5 if semantic else 0.0,
            lexical_rank=0.4 if lexical else 0.0,
            rrf_semantic=semantic,
            rrf_lexical=lexical,
        )
        return RetrievalOutcome(
            results=[scored],
            best_similarity=0.5 if relevant else 0.01,
            threshold=0.25,
            best_lexical_rank=0.4 if lexical else 0.0,
            lexical_threshold=0.02,
            lexical_eligible=bool(lexical),
        )

    @pytest_asyncio.fixture
    async def chat_session(self, db: AsyncSession, user_id: str) -> ChatSession:
        session = make_chat_session(user_id)
        db.add(session)
        await db.flush()
        return session

    async def test_a_search_is_recorded_with_its_split(
        self, db: AsyncSession, chat_session: ChatSession
    ) -> None:
        repository = RetrievalMetricRepository(db)

        repository.record(
            self.outcome(semantic=0.75, lexical=0.25),
            question_kind="document",
            session_id=chat_session.id,
            latency_ms=120,
        )
        await repository.flush()

        summary = await repository.arm_summary(chat_session.id)
        assert summary["searches"] == 1
        assert summary["semantic_share"] == pytest.approx(0.75)
        assert summary["lexical_share"] == pytest.approx(0.25)
        assert summary["chunks_from_both"] == 1

    async def test_the_summary_averages_searches_not_chunks(
        self, db: AsyncSession, chat_session: ChatSession
    ) -> None:
        """Each search counts once however many chunks it returned."""
        repository = RetrievalMetricRepository(db)

        repository.record(
            self.outcome(semantic=1.0, lexical=0.0),
            question_kind="document",
            session_id=chat_session.id,
        )
        repository.record(
            self.outcome(semantic=0.0, lexical=1.0),
            question_kind="document",
            session_id=chat_session.id,
        )
        await repository.flush()

        summary = await repository.arm_summary(chat_session.id)
        assert summary["semantic_share"] == pytest.approx(0.5)
        assert summary["chunks_from_semantic_only"] == 1
        assert summary["chunks_from_lexical_only"] == 1

    async def test_kinds_are_reported_separately(
        self, db: AsyncSession, chat_session: ChatSession
    ) -> None:
        """Counting questions retrieve wider, so their shares are not
        comparable with ordinary ones."""
        repository = RetrievalMetricRepository(db)

        repository.record(
            self.outcome(semantic=1.0, lexical=0.0),
            question_kind="document",
            session_id=chat_session.id,
        )
        repository.record(
            self.outcome(semantic=0.5, lexical=0.5),
            question_kind="aggregate",
            session_id=chat_session.id,
        )
        await repository.flush()

        rows = {
            row["question_kind"]: row
            for row in await repository.by_question_kind(chat_session.id)
        }
        assert rows["document"]["semantic_share"] == pytest.approx(1.0)
        assert rows["aggregate"]["semantic_share"] == pytest.approx(0.5)

    async def test_refusals_are_recorded_too(
        self, db: AsyncSession, chat_session: ChatSession
    ) -> None:
        """A search that found nothing is still a search worth counting."""
        repository = RetrievalMetricRepository(db)

        repository.record(
            self.outcome(semantic=0.2, lexical=0.0, relevant=False),
            question_kind="document",
            session_id=chat_session.id,
        )
        await repository.flush()

        rows = await repository.by_question_kind(chat_session.id)
        assert rows[0]["searches"] == 1
        assert rows[0]["relevant_rate"] == pytest.approx(0.0)

    async def test_a_scope_with_no_searches_reports_zeroes(
        self, db: AsyncSession, chat_session: ChatSession
    ) -> None:
        summary = await RetrievalMetricRepository(db).arm_summary(chat_session.id)

        assert summary["searches"] == 0
        assert summary["semantic_share"] == 0.0
