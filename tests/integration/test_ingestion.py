"""The ingestion pipeline running against a real database.

AI providers are replaced with deterministic doubles, but extraction, chunking,
persistence, metrics and status transitions are all real.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.analyzer import DocumentAnalyzer
from app.ai.base import (
    ChatMessage,
    EmbeddingProvider,
    EmbeddingResult,
    LLMProvider,
    StructuredResult,
    TokenUsage,
)
from app.core.config import get_settings
from app.core.constants import DocumentStatus, FileType, ProcessingStage, StageStatus
from app.core.exceptions import StorageError, TextExtractionError
from app.db.session import db_manager
from app.models.chunk import DocumentChunk
from app.models.document import Document, DocumentInsight
from app.models.metrics import ApiUsage, ProcessingMetric
from app.processing.extractors import ExtractorRegistry
from app.services.ingestion_service import IngestionService
from app.storage.base import StorageBackend, StoredObject

pytestmark = pytest.mark.integration

WIDTH = 1536

SAMPLE_TEXT = (
    "TechCorp Expense Policy\n\n"
    "Employees may claim up to 500 USD per year for home office equipment. "
    "Receipts must be submitted within thirty days of purchase.\n\n"
    "Travel expenses require manager approval before booking. Economy class "
    "is standard for flights under six hours.\n\n"
    "Client entertainment is capped at 150 USD per person per event and must "
    "be recorded against a specific account.\n"
)


class InMemoryStorage(StorageBackend):
    """Storage double holding bytes in a dictionary."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = files or {}

    def save(self, key: str, data: bytes, *, content_type: str) -> StoredObject:
        self.files[key] = data
        return StoredObject(key=key, size_bytes=len(data), content_type=content_type)

    def load(self, key: str) -> bytes:
        if key not in self.files:
            raise StorageError(f"missing {key}")
        return self.files[key]

    def delete(self, key: str) -> None:
        self.files.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self.files

    def presigned_url(self, key: str, *, expires_in: int | None = None) -> str:
        return f"memory://{key}"

    def health_check(self) -> None:
        return None


class FakeEmbedder(EmbeddingProvider):
    """Returns deterministic vectors without calling a provider."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def model(self) -> str:
        return "fake-embedding"

    @property
    def dimensions(self) -> int:
        return WIDTH

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        self.call_count += 1
        return EmbeddingResult(
            vectors=[[(len(t) % 10) / 10] + [0.0] * (WIDTH - 1) for t in texts],
            usage=TokenUsage(prompt_tokens=len(texts) * 12, cost_usd=Decimal("0.0001")),
            model=self.model,
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text]).vectors[0]


class FakeLLM(LLMProvider):
    """Returns a fixed analysis payload."""

    @property
    def model(self) -> str:
        return "fake-chat"

    def complete(self, messages, *, max_tokens=None, temperature=None):
        raise NotImplementedError

    def complete_structured(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema,
        schema_name: str,
        max_tokens: int | None = None,
    ) -> StructuredResult:
        return StructuredResult(
            data={
                "summary": "An expense policy covering equipment and travel.",
                "key_points": ["Home office claims are capped at 500 USD."],
                "tags": ["Expenses", "POLICY"],
                "category": "financial",
                "sentiment": "neutral",
                "language": "en",
                "entities": [{"name": "TechCorp", "type": "organization"}],
            },
            usage=TokenUsage(
                prompt_tokens=200, completion_tokens=60, cost_usd=Decimal("0.0005")
            ),
            model=self.model,
        )

    def stream(self, messages, *, max_tokens=None, temperature=None, accumulator=None):
        raise NotImplementedError


@pytest.fixture
def sync_db():
    """A sync session whose work is rolled back after the test."""
    connection = db_manager.sync_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


def seed_document(
    session: Session,
    storage: InMemoryStorage,
    *,
    content: bytes = SAMPLE_TEXT.encode(),
    file_type: FileType = FileType.TXT,
) -> Document:
    """Persist a queued document whose bytes are already in storage."""
    document_id = uuid.uuid4()
    key = f"documents/test/{document_id}.{file_type.value}"
    storage.save(key, content, content_type=file_type.media_type)

    document = Document(
        id=document_id,
        user_id=f"test-{uuid.uuid4().hex[:8]}",
        filename=f"policy.{file_type.value}",
        file_type=file_type,
        file_size_bytes=len(content),
        storage_key=key,
        status=DocumentStatus.QUEUED,
    )
    session.add(document)
    session.flush()
    return document


def build_service(
    session: Session, storage: InMemoryStorage, *, embedder=None, llm=None
) -> IngestionService:
    return IngestionService(
        session,
        settings=get_settings(),
        storage=storage,
        extractors=ExtractorRegistry(),
        embedder=embedder or FakeEmbedder(),
        llm=llm or FakeLLM(),
    )


class TestSuccessfulIngestion:
    def test_the_document_becomes_ready(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        document = seed_document(sync_db, storage)

        result = build_service(sync_db, storage).process(document.id)

        sync_db.refresh(document)
        assert document.status is DocumentStatus.READY
        assert document.processed_at is not None
        assert result.chunks > 0

    def test_chunks_are_persisted_with_embeddings(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        document = seed_document(sync_db, storage)

        build_service(sync_db, storage).process(document.id)

        chunks = sync_db.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        ).all()
        assert chunks
        assert all(chunk.embedding is not None for chunk in chunks)
        assert all(len(chunk.embedding) == WIDTH for chunk in chunks)

    def test_insights_are_written_and_normalised(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        document = seed_document(sync_db, storage)

        build_service(sync_db, storage).process(document.id)

        insight = sync_db.scalar(
            select(DocumentInsight).where(DocumentInsight.document_id == document.id)
        )
        assert insight is not None
        assert insight.category.value == "financial"
        assert insight.tags == ["expenses", "policy"]

    def test_every_stage_is_recorded_as_succeeded(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        document = seed_document(sync_db, storage)

        build_service(sync_db, storage).process(document.id)

        metrics = sync_db.scalars(
            select(ProcessingMetric).where(ProcessingMetric.document_id == document.id)
        ).all()

        assert {metric.stage for metric in metrics} == set(ProcessingStage)
        assert all(metric.status is StageStatus.SUCCEEDED for metric in metrics)

    def test_ai_spend_is_attributed_to_the_document(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        document = seed_document(sync_db, storage)

        build_service(sync_db, storage).process(document.id)

        usage = sync_db.scalars(
            select(ApiUsage).where(ApiUsage.document_id == document.id)
        ).all()
        assert {record.operation.value for record in usage} == {"embedding", "analysis"}
        assert sum(record.cost_usd for record in usage) > 0

    def test_counters_are_updated_on_the_document(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        document = seed_document(sync_db, storage)

        build_service(sync_db, storage).process(document.id)

        sync_db.refresh(document)
        assert document.page_count
        assert document.chunk_count
        assert document.character_count


class TestIdempotency:
    def test_reprocessing_replaces_rather_than_duplicates_chunks(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        document = seed_document(sync_db, storage)
        service = build_service(sync_db, storage)

        first = service.process(document.id)
        second = service.process(document.id)

        chunks = sync_db.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        ).all()
        assert first.chunks == second.chunks == len(chunks)

    def test_reprocessing_leaves_exactly_one_insight(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        document = seed_document(sync_db, storage)
        service = build_service(sync_db, storage)

        service.process(document.id)
        service.process(document.id)

        insights = sync_db.scalars(
            select(DocumentInsight).where(DocumentInsight.document_id == document.id)
        ).all()
        assert len(insights) == 1


class TestFailureHandling:
    def test_unreadable_content_fails_with_a_precise_reason(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        document = seed_document(sync_db, storage, content=b"   \n  ")

        with pytest.raises(TextExtractionError):
            build_service(sync_db, storage).process(document.id)

        reloaded = sync_db.get(Document, document.id)
        assert reloaded.status is DocumentStatus.FAILED
        assert reloaded.failure_code == "no_extractable_text"
        assert reloaded.failure_reason

    def test_a_missing_file_fails_at_the_extract_stage(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        document = seed_document(sync_db, storage)
        storage.delete(document.storage_key)

        with pytest.raises(StorageError):
            build_service(sync_db, storage).process(document.id)

        failures = sync_db.scalars(
            select(ProcessingMetric).where(
                ProcessingMetric.document_id == document.id,
                ProcessingMetric.status == StageStatus.FAILED,
            )
        ).all()
        assert [metric.stage for metric in failures] == [ProcessingStage.EXTRACT]

    def test_a_failed_document_can_be_requeued(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        document = seed_document(sync_db, storage, content=b"  ")
        service = build_service(sync_db, storage)

        with pytest.raises(TextExtractionError):
            service.process(document.id)

        service.reset_for_retry(document.id)
        sync_db.flush()

        reloaded = sync_db.get(Document, document.id)
        assert reloaded.status is DocumentStatus.QUEUED
        assert reloaded.failure_code is None


class TestLargerDocuments:
    def test_a_long_document_produces_many_chunks(
        self, sync_db: Session, storage: InMemoryStorage
    ) -> None:
        long_text = (SAMPLE_TEXT + "\n\n") * 40
        document = seed_document(sync_db, storage, content=long_text.encode())
        embedder = FakeEmbedder()

        result = build_service(sync_db, storage, embedder=embedder).process(document.id)

        assert result.chunks > 1
        assert embedder.call_count >= 1
