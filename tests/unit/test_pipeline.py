"""Ingestion pipeline stages, runner and failure handling."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from app.ai.base import EmbeddingProvider, EmbeddingResult, TokenUsage
from app.core.config import IngestionSettings
from app.core.constants import FileType, ProcessingStage, StageStatus
from app.core.exceptions import ProcessingError, TextExtractionError
from app.models.document import Document
from app.processing.chunker import DocumentChunker
from app.processing.extractors import ExtractorRegistry
from app.processing.pipeline import (
    ChunkStage,
    EmbedStage,
    ExtractStage,
    PipelineContext,
    PipelineStage,
    StageOutcome,
    StageRunner,
    failure_details,
)

EMBEDDING_WIDTH = 8


class FakeEmbedder(EmbeddingProvider):
    """Deterministic embedder that records the batches it was given."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    @property
    def model(self) -> str:
        return "fake-embedding-model"

    @property
    def dimensions(self) -> int:
        return EMBEDDING_WIDTH

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        self.batches.append(list(texts))
        return EmbeddingResult(
            vectors=[[float(len(t) % 7)] * EMBEDDING_WIDTH for t in texts],
            usage=TokenUsage(prompt_tokens=len(texts) * 10, cost_usd=Decimal("0.001")),
            model=self.model,
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text]).vectors[0]


@pytest.fixture
def document() -> Document:
    return Document(
        id=uuid.uuid4(),
        user_id="user-1",
        filename="policy.txt",
        file_type=FileType.TXT,
        file_size_bytes=1000,
        storage_key="documents/abc/policy.txt",
    )


@pytest.fixture
def context(document: Document) -> PipelineContext:
    return PipelineContext(document=document)


class TestExtractStage:
    def test_populates_text_and_document_counters(
        self, context: PipelineContext
    ) -> None:
        storage = MagicMock()
        storage.load.return_value = b"Policy content for the vault. " * 20

        outcome = ExtractStage(storage, ExtractorRegistry()).execute(context)

        assert context.extracted is not None
        assert context.document.page_count == context.extracted.page_count
        assert context.document.character_count > 0
        assert outcome.items_processed == context.extracted.page_count
        storage.load.assert_called_once_with("documents/abc/policy.txt")

    def test_unreadable_content_raises_extraction_error(
        self, context: PipelineContext
    ) -> None:
        storage = MagicMock()
        storage.load.return_value = b"    "

        with pytest.raises(TextExtractionError):
            ExtractStage(storage, ExtractorRegistry()).execute(context)


class TestChunkStage:
    def test_persists_chunks_and_sets_the_count(
        self, context: PipelineContext
    ) -> None:
        storage = MagicMock()
        storage.load.return_value = b"A policy sentence with substance. " * 60
        ExtractStage(storage, ExtractorRegistry()).execute(context)

        chunks_repo = MagicMock()
        chunks_repo.replace_for_document.side_effect = lambda _id, rows: len(rows)
        chunker = DocumentChunker(IngestionSettings(chunk_tokens=100))

        outcome = ChunkStage(chunker, chunks_repo).execute(context)

        assert outcome.items_processed > 0
        assert context.document.chunk_count == outcome.items_processed
        chunks_repo.replace_for_document.assert_called_once()

    def test_a_document_without_chunks_is_a_processing_error(
        self, context: PipelineContext
    ) -> None:
        chunker = MagicMock()
        chunker.chunk.return_value = []
        context.extracted = MagicMock()

        with pytest.raises(ProcessingError, match="no usable text chunks"):
            ChunkStage(chunker, MagicMock()).execute(context)


class TestEmbedStage:
    @staticmethod
    def _stage(
        embedder: EmbeddingProvider, repo: MagicMock, batch: int = 2
    ) -> EmbedStage:
        return EmbedStage(
            embedder, repo, MagicMock(), IngestionSettings(embed_batch_size=batch)
        )

    @staticmethod
    def _rows(count: int) -> list[MagicMock]:
        return [MagicMock(content=f"chunk {i}", embedding=None) for i in range(count)]

    def test_assigns_a_vector_to_every_chunk(self, context: PipelineContext) -> None:
        rows = self._rows(5)
        repo = MagicMock()
        repo.pending_embeddings.return_value = rows

        outcome = self._stage(FakeEmbedder(), repo).execute(context)

        assert outcome.items_processed == 5
        assert all(row.embedding is not None for row in rows)
        assert all(len(row.embedding) == EMBEDDING_WIDTH for row in rows)

    def test_requests_are_batched_to_the_configured_size(
        self, context: PipelineContext
    ) -> None:
        repo = MagicMock()
        repo.pending_embeddings.return_value = self._rows(5)
        embedder = FakeEmbedder()

        self._stage(embedder, repo, batch=2).execute(context)

        assert [len(batch) for batch in embedder.batches] == [2, 2, 1]

    def test_usage_is_accumulated_across_batches(
        self, context: PipelineContext
    ) -> None:
        repo = MagicMock()
        repo.pending_embeddings.return_value = self._rows(4)

        self._stage(FakeEmbedder(), repo, batch=2).execute(context)

        assert context.usage.cost_usd == Decimal("0.002")
        assert context.usage.prompt_tokens == 40

    def test_already_embedded_documents_do_no_work(
        self, context: PipelineContext
    ) -> None:
        repo = MagicMock()
        repo.pending_embeddings.return_value = []
        embedder = FakeEmbedder()

        outcome = self._stage(embedder, repo).execute(context)

        assert outcome.items_processed == 0
        assert embedder.batches == []

    def test_vector_count_mismatch_is_rejected(
        self, context: PipelineContext
    ) -> None:
        repo = MagicMock()
        repo.pending_embeddings.return_value = self._rows(3)

        embedder = MagicMock()
        embedder.embed_documents.return_value = EmbeddingResult(
            vectors=[[0.1] * EMBEDDING_WIDTH], usage=TokenUsage(), model="fake"
        )

        with pytest.raises(ProcessingError, match="returned 1 vectors"):
            self._stage(embedder, repo, batch=3).execute(context)


class _BoomStage(PipelineStage):
    stage = ProcessingStage.EXTRACT

    def execute(self, context: PipelineContext) -> StageOutcome:
        raise TextExtractionError("the file is unreadable")


class _OkStage(PipelineStage):
    stage = ProcessingStage.CHUNK

    def execute(self, context: PipelineContext) -> StageOutcome:
        return StageOutcome(items_processed=3)


class TestStageRunner:
    def test_success_is_recorded_with_timing(self, context: PipelineContext) -> None:
        metrics = MagicMock()
        StageRunner(metrics).run([_OkStage()], context)

        kwargs = metrics.record.call_args.kwargs
        assert kwargs["status"] is StageStatus.SUCCEEDED
        assert kwargs["stage"] is ProcessingStage.CHUNK
        assert kwargs["items_processed"] == 3
        assert kwargs["duration_ms"] >= 0

    def test_failure_is_recorded_then_re_raised(
        self, context: PipelineContext
    ) -> None:
        metrics = MagicMock()

        with pytest.raises(TextExtractionError):
            StageRunner(metrics).run([_BoomStage()], context)

        kwargs = metrics.record.call_args.kwargs
        assert kwargs["status"] is StageStatus.FAILED
        assert kwargs["error_code"] == "text_extraction_failed"
        assert "unreadable" in kwargs["error_message"]

    def test_later_stages_do_not_run_after_a_failure(
        self, context: PipelineContext
    ) -> None:
        metrics = MagicMock()
        later = MagicMock(spec=PipelineStage)

        with pytest.raises(TextExtractionError):
            StageRunner(metrics).run([_BoomStage(), later], context)

        later.execute.assert_not_called()


class TestFailureDetails:
    def test_domain_errors_expose_their_code(self) -> None:
        code, message = failure_details(TextExtractionError("bad file"))
        assert code == "text_extraction_failed"
        assert message == "bad file"

    def test_unexpected_errors_fall_back_to_the_class_name(self) -> None:
        code, message = failure_details(ValueError("odd"))
        assert code == "ValueError"
        assert message == "odd"

    def test_blank_messages_get_a_default(self) -> None:
        _, message = failure_details(ValueError())
        assert message
