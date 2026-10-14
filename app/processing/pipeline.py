"""Asynchronous document ingestion pipeline.

The pipeline is a sequence of independent stages. Each stage reads and writes a
shared context, and the runner times every stage and records its outcome, so a
failure is attributable to a specific step rather than to "processing" as a
whole. Stages are individually idempotent, which makes a retry safe.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.ai.analyzer import DocumentAnalyzer
from app.ai.base import EmbeddingProvider, TokenUsage
from app.core.config import IngestionSettings
from app.core.constants import AIOperation, ProcessingStage, StageStatus
from app.core.exceptions import AppError, ProcessingError
from app.core.logging import get_logger
from app.models.chunk import DocumentChunk
from app.models.document import Document, DocumentInsight
from app.processing.chunker import DocumentChunker, TextChunk
from app.processing.extractors import ExtractedDocument, ExtractorRegistry
from app.repositories.chunk import SyncChunkRepository
from app.repositories.document import SyncDocumentRepository
from app.repositories.metrics import (
    SyncApiUsageRepository,
    SyncProcessingMetricRepository,
)
from app.storage.base import StorageBackend

logger = get_logger(__name__)


@dataclass
class PipelineContext:
    """Mutable state threaded through the stages of one ingestion run."""

    document: Document
    file_bytes: bytes | None = None
    extracted: ExtractedDocument | None = None
    text_chunks: list[TextChunk] = field(default_factory=list)
    persisted_chunks: int = 0
    embedded_chunks: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def document_id(self) -> uuid.UUID:
        return self.document.id

    def require_extracted(self) -> ExtractedDocument:
        if self.extracted is None:
            raise ProcessingError("Extraction has not run for this document.")
        return self.extracted


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """Result of one stage, used for metrics and logging."""

    items_processed: int = 0


class PipelineStage(ABC):
    """One step of the ingestion pipeline."""

    stage: ProcessingStage

    @abstractmethod
    def execute(self, context: PipelineContext) -> StageOutcome:
        """Advance the context, raising on failure."""


class ExtractStage(PipelineStage):
    """Loads the stored file and converts it to page-aware text."""

    stage = ProcessingStage.EXTRACT

    def __init__(self, storage: StorageBackend, registry: ExtractorRegistry) -> None:
        self._storage = storage
        self._registry = registry

    def execute(self, context: PipelineContext) -> StageOutcome:
        document = context.document
        context.file_bytes = self._storage.load(document.storage_key)
        context.extracted = self._registry.extract(
            context.file_bytes,
            file_type=document.file_type,
            filename=document.filename,
        )

        document.page_count = context.extracted.page_count
        document.character_count = context.extracted.character_count
        return StageOutcome(items_processed=context.extracted.page_count)


class ChunkStage(PipelineStage):
    """Splits extracted text into overlapping, page-attributed chunks."""

    stage = ProcessingStage.CHUNK

    def __init__(self, chunker: DocumentChunker, chunks: SyncChunkRepository) -> None:
        self._chunker = chunker
        self._chunks = chunks

    def execute(self, context: PipelineContext) -> StageOutcome:
        context.text_chunks = self._chunker.chunk(context.require_extracted())
        if not context.text_chunks:
            raise ProcessingError(
                "The document produced no usable text chunks. It may contain "
                "only images or formatting."
            )

        rows = [
            DocumentChunk(
                document_id=context.document.id,
                chunk_index=chunk.index,
                content=chunk.content,
                page_number=chunk.page_number,
                token_count=chunk.token_count,
            )
            for chunk in context.text_chunks
        ]
        # Replacing rather than appending keeps a re-run idempotent.
        context.persisted_chunks = self._chunks.replace_for_document(
            context.document.id, rows
        )
        context.document.chunk_count = context.persisted_chunks
        return StageOutcome(items_processed=context.persisted_chunks)


class EmbedStage(PipelineStage):
    """Embeds chunks in batches and stores the resulting vectors."""

    stage = ProcessingStage.EMBED

    def __init__(
        self,
        embedder: EmbeddingProvider,
        chunks: SyncChunkRepository,
        usage: SyncApiUsageRepository,
        settings: IngestionSettings,
    ) -> None:
        self._embedder = embedder
        self._chunks = chunks
        self._usage = usage
        self._settings = settings

    def execute(self, context: PipelineContext) -> StageOutcome:
        pending = self._chunks.pending_embeddings(context.document.id)
        if not pending:
            return StageOutcome(items_processed=0)

        batch_size = self._settings.embed_batch_size
        embedded = 0

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            result = self._embedder.embed_documents([row.content for row in batch])

            if len(result.vectors) != len(batch):
                raise ProcessingError(
                    f"Embedding provider returned {len(result.vectors)} vectors "
                    f"for {len(batch)} chunks."
                )

            for row, vector in zip(batch, result.vectors, strict=True):
                row.embedding = vector

            context.usage = context.usage + result.usage
            self._usage.record(
                operation=AIOperation.EMBEDDING,
                model=result.model,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=0,
                cost_usd=result.usage.cost_usd,
                document_id=context.document.id,
            )
            # Flush per batch so progress survives a mid-document failure.
            self._chunks.flush()
            embedded += len(batch)

        context.embedded_chunks = embedded
        return StageOutcome(items_processed=embedded)


class StageRunner:
    """Executes stages in order, timing each one and recording its outcome."""

    def __init__(self, metrics: SyncProcessingMetricRepository) -> None:
        self._metrics = metrics

    def run(self, stages: list[PipelineStage], context: PipelineContext) -> None:
        for stage in stages:
            self._run_one(stage, context)

    def _run_one(self, stage: PipelineStage, context: PipelineContext) -> None:
        started = time.perf_counter()
        try:
            outcome = stage.execute(context)
        except Exception as exc:
            code, message = failure_details(exc)
            self._metrics.record(
                document_id=context.document.id,
                stage=stage.stage,
                status=StageStatus.FAILED,
                duration_ms=self._elapsed_ms(started),
                error_code=code,
                error_message=message,
            )
            logger.error(
                "Pipeline stage failed",
                extra={
                    "document_id": str(context.document.id),
                    "stage": stage.stage.value,
                    "error": message,
                },
            )
            raise

        duration_ms = self._elapsed_ms(started)
        self._metrics.record(
            document_id=context.document.id,
            stage=stage.stage,
            status=StageStatus.SUCCEEDED,
            duration_ms=duration_ms,
            items_processed=outcome.items_processed,
        )
        logger.info(
            "Pipeline stage completed",
            extra={
                "document_id": str(context.document.id),
                "stage": stage.stage.value,
                "duration_ms": duration_ms,
                "items": outcome.items_processed,
            },
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary of a completed ingestion run, returned to the Celery task."""

    document_id: str
    pages: int
    chunks: int
    embedded: int
    cost_usd: float

    @classmethod
    def from_context(cls, context: PipelineContext) -> "IngestionResult":
        return cls(
            document_id=str(context.document.id),
            pages=context.document.page_count or 0,
            chunks=context.persisted_chunks,
            embedded=context.embedded_chunks,
            cost_usd=float(context.usage.cost_usd),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "pages": self.pages,
            "chunks": self.chunks,
            "embedded": self.embedded,
            "cost_usd": self.cost_usd,
        }


def failure_details(exc: Exception) -> tuple[str, str]:
    """Extract a stable error code and a user-facing reason from an exception."""
    if isinstance(exc, AppError):
        return exc.code, exc.message
    return exc.__class__.__name__, str(exc) or "Processing failed unexpectedly."


class AnalyzeStage(PipelineStage):
    """Generates the structured AI insight record for the document.

    Runs last so that a failure here still leaves a searchable document: the
    chunks are already embedded, and chat works without insights.
    """

    stage = ProcessingStage.ANALYZE

    def __init__(
        self,
        analyzer: DocumentAnalyzer,
        documents: SyncDocumentRepository,
        usage: SyncApiUsageRepository,
    ) -> None:
        self._analyzer = analyzer
        self._documents = documents
        self._usage = usage

    def execute(self, context: PipelineContext) -> StageOutcome:
        document = context.document
        analysis = self._analyzer.analyze(
            context.require_extracted().text, filename=document.filename
        )

        self._documents.upsert_insight(
            DocumentInsight(
                document_id=document.id,
                summary=analysis.summary,
                key_points=analysis.key_points,
                tags=analysis.tags,
                entities=analysis.entities,
                category=analysis.category,
                sentiment=analysis.sentiment,
                language=analysis.language,
                model=analysis.model,
                token_usage=analysis.usage.as_dict(),
                was_truncated=analysis.was_truncated,
            )
        )

        context.usage = context.usage + analysis.usage
        self._usage.record(
            operation=AIOperation.ANALYSIS,
            model=analysis.model,
            prompt_tokens=analysis.usage.prompt_tokens,
            completion_tokens=analysis.usage.completion_tokens,
            cost_usd=analysis.usage.cost_usd,
            document_id=document.id,
        )
        return StageOutcome(items_processed=1)
