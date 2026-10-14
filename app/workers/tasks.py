"""Celery tasks for asynchronous document processing."""

from __future__ import annotations

import uuid
from typing import Any

from celery import Task

from app.ai import get_embedding_provider, get_llm_provider
from app.core.config import get_settings
from app.core.exceptions import (
    LLMRateLimitError,
    LLMTimeoutError,
    StorageError,
    TextExtractionError,
)
from app.core.logging import CorrelationId, get_logger
from app.db.session import db_manager
from app.processing.extractors import get_extractor_registry
from app.services.ingestion_service import IngestionService
from app.storage import get_storage_backend
from app.workers.celery_app import INGESTION_QUEUE, celery_app

logger = get_logger(__name__)

#: Transient infrastructure failures worth retrying. Extraction errors are
#: excluded deliberately: a corrupt PDF will never parse, so retrying only
#: delays the user's feedback.
RETRYABLE_ERRORS = (LLMRateLimitError, LLMTimeoutError, StorageError, ConnectionError)

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 10


@celery_app.task(
    bind=True,
    name="documents.process",
    queue=INGESTION_QUEUE,
    max_retries=MAX_RETRIES,
    autoretry_for=RETRYABLE_ERRORS,
    retry_backoff=RETRY_BACKOFF_SECONDS,
    retry_backoff_max=300,
    retry_jitter=True,
)
def process_document(
    self: Task, document_id: str, correlation_id: str | None = None
) -> dict[str, Any]:
    """Run the ingestion pipeline for one uploaded document.

    Returns a summary of the run. Permanent failures are recorded on the
    document and re-raised without retrying; transient failures are retried by
    Celery using the policy declared above.
    """
    token = CorrelationId.set(correlation_id)
    identifier = uuid.UUID(document_id)

    try:
        with db_manager.sync_session() as session:
            service = IngestionService(
                session,
                settings=get_settings(),
                storage=get_storage_backend(),
                extractors=get_extractor_registry(),
                embedder=get_embedding_provider(),
                llm=get_llm_provider(),
            )
            return service.process(identifier).as_dict()
    except TextExtractionError:
        # Already recorded against the document; retrying cannot help.
        logger.info(
            "Document rejected as unreadable",
            extra={"document_id": document_id, "attempt": self.request.retries + 1},
        )
        raise
    except RETRYABLE_ERRORS as exc:
        logger.warning(
            "Transient ingestion failure; will retry",
            extra={
                "document_id": document_id,
                "attempt": self.request.retries + 1,
                "error": str(exc),
            },
        )
        raise
    finally:
        CorrelationId.reset(token)


@celery_app.task(name="documents.reprocess", queue=INGESTION_QUEUE)
def reprocess_document(document_id: str) -> dict[str, Any]:
    """Requeue a previously failed document for another processing attempt."""
    identifier = uuid.UUID(document_id)
    with db_manager.sync_session() as session:
        IngestionService(
            session,
            settings=get_settings(),
            storage=get_storage_backend(),
            extractors=get_extractor_registry(),
            embedder=get_embedding_provider(),
            llm=get_llm_provider(),
        ).reset_for_retry(identifier)

    process_document.delay(document_id)
    return {"document_id": document_id, "status": "requeued"}
