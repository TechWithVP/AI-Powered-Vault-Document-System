"""Celery application used for asynchronous document processing."""

from __future__ import annotations

from celery import Celery
from celery.signals import setup_logging, worker_ready

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

#: Queue names. Ingestion is separated from short interactive jobs so a backlog
#: of large documents cannot starve latency-sensitive work.
INGESTION_QUEUE = "ingestion"
DEFAULT_QUEUE = "default"


def create_celery_app(settings: Settings | None = None) -> Celery:
    """Build the configured Celery application."""
    settings = settings or get_settings()

    celery_app = Celery(
        "vault",
        broker=settings.redis.broker_url,
        backend=settings.redis.broker_url,
        include=["app.workers.tasks"],
    )

    celery_app.conf.update(
        task_default_queue=DEFAULT_QUEUE,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # Fetch one task at a time: document processing is long-running, so
        # prefetching would leave work queued behind a busy worker.
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_track_started=True,
        task_time_limit=30 * 60,
        task_soft_time_limit=25 * 60,
        result_expires=24 * 60 * 60,
        broker_connection_retry_on_startup=True,
    )
    return celery_app


celery_app = create_celery_app()


@setup_logging.connect
def _configure_worker_logging(**_: object) -> None:
    """Use the application's structured logging inside workers too."""
    settings = get_settings()
    configure_logging(
        level=settings.app.log_level, json_output=settings.app.log_json
    )


@worker_ready.connect
def _on_worker_ready(**_: object) -> None:
    logger.info("Celery worker ready")
