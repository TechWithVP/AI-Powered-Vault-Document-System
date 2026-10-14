"""Worker-side orchestration of document ingestion.

Owns the transactional boundary for one ingestion run: it builds the stage
pipeline, marks the document processing, and guarantees the document always
reaches a terminal state - ready or failed with a precise reason - even when a
stage raises.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.ai.analyzer import DocumentAnalyzer
from app.ai.base import EmbeddingProvider, LLMProvider
from app.core.config import Settings
from app.core.constants import DocumentStatus
from app.core.logging import get_logger
from app.processing.chunker import DocumentChunker
from app.processing.extractors import ExtractorRegistry
from app.processing.pipeline import (
    AnalyzeStage,
    ChunkStage,
    EmbedStage,
    ExtractStage,
    IngestionResult,
    PipelineContext,
    PipelineStage,
    StageRunner,
    failure_details,
)
from app.repositories.chunk import SyncChunkRepository
from app.repositories.document import SyncDocumentRepository
from app.repositories.metrics import (
    SyncApiUsageRepository,
    SyncProcessingMetricRepository,
)
from app.storage.base import StorageBackend

logger = get_logger(__name__)


class IngestionService:
    """Runs the ingestion pipeline for a single document."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings,
        storage: StorageBackend,
        extractors: ExtractorRegistry,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        chunker: DocumentChunker | None = None,
        analyzer: DocumentAnalyzer | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._documents = SyncDocumentRepository(session)
        self._chunks = SyncChunkRepository(session)
        self._metrics = SyncProcessingMetricRepository(session)
        self._usage = SyncApiUsageRepository(session)
        self._storage = storage
        self._extractors = extractors
        self._embedder = embedder
        self._chunker = chunker or DocumentChunker(settings.ingestion)
        self._analyzer = analyzer or DocumentAnalyzer(llm, settings.ingestion)

    def build_stages(self) -> list[PipelineStage]:
        """Compose the stage sequence for an ingestion run."""
        return [
            ExtractStage(self._storage, self._extractors),
            ChunkStage(self._chunker, self._chunks),
            EmbedStage(
                self._embedder, self._chunks, self._usage, self._settings.ingestion
            ),
            AnalyzeStage(self._analyzer, self._documents, self._usage),
        ]

    def process(self, document_id: uuid.UUID) -> IngestionResult:
        """Process one document end to end.

        Always leaves the document in a terminal state. Failures are recorded
        against the document with a code the API can surface verbatim, then
        re-raised so Celery can apply its retry policy.
        """
        document = self._documents.get_or_fail(document_id)
        context = PipelineContext(document=document)

        document.mark_processing()
        self._session.flush()

        logger.info(
            "Ingestion started",
            extra={
                "document_id": str(document_id),
                "filename": document.filename,
                "file_type": document.file_type.value,
            },
        )

        runner = StageRunner(self._metrics)
        try:
            runner.run(self.build_stages(), context)
        except Exception as exc:
            code, reason = failure_details(exc)
            document.mark_failed(code=code, reason=reason)
            # Commit the failure state before propagating, so the API can report
            # why processing failed even though the task itself errored.
            self._session.commit()
            logger.error(
                "Ingestion failed",
                extra={
                    "document_id": str(document_id),
                    "failure_code": code,
                    "failure_reason": reason,
                },
            )
            raise

        document.mark_ready()
        self._session.flush()

        result = IngestionResult.from_context(context)
        logger.info("Ingestion completed", extra=result.as_dict())
        return result

    def mark_failed(self, document_id: uuid.UUID, *, code: str, reason: str) -> None:
        """Record a terminal failure raised outside the stage pipeline."""
        document = self._documents.get(document_id)
        if document is None:
            return
        document.mark_failed(code=code, reason=reason)

    def reset_for_retry(self, document_id: uuid.UUID) -> None:
        """Return a failed document to the queue so it can be processed again."""
        document = self._documents.get_or_fail(document_id)
        document.status = DocumentStatus.QUEUED
        document.failure_code = None
        document.failure_reason = None
