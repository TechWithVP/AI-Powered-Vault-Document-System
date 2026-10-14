"""Shared FastAPI dependencies.

Wiring lives here so routers declare what they need and never construct
infrastructure themselves, which keeps endpoints thin and services testable.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.cache import EmbeddingCache, RedisEmbeddingCache
from app.core.config import Settings, get_settings
from app.core.redis import get_async_redis
from app.core.security import CurrentPrincipal
from app.db.session import get_db_session
from app.schemas.common import PageParams
from app.ai import EmbeddingProvider, LLMProvider, get_embedding_provider, get_llm_provider
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.metrics_service import MetricsService
from app.storage import StorageBackend, get_storage_backend

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
AppSettingsDep = Annotated[Settings, Depends(get_settings)]
StorageDep = Annotated[StorageBackend, Depends(get_storage_backend)]
Pagination = Annotated[PageParams, Depends()]


def get_embedding_cache() -> EmbeddingCache:
    """Query embeddings are deterministic, so repeats need no API call."""
    return RedisEmbeddingCache(get_async_redis())


CacheDep = Annotated[EmbeddingCache, Depends(get_embedding_cache)]
LLMDep = Annotated[LLMProvider, Depends(get_llm_provider)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]


def get_document_service(
    session: DbSession,
    settings: AppSettingsDep,
    storage: StorageDep,
) -> DocumentService:
    return DocumentService(session, settings=settings, storage=storage)


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


def build_chat_service(
    session: AsyncSession,
    *,
    settings: Settings,
    storage: StorageBackend,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    cache: EmbeddingCache | None = None,
) -> ChatService:
    """Assemble a ChatService around an explicit session.

    Streaming responses outlive the request-scoped session, so the SSE endpoint
    opens its own unit of work and builds the service with this rather than
    through Depends.
    """
    return ChatService(
        session,
        settings=settings,
        llm=llm,
        embedder=embedder,
        documents=DocumentService(session, settings=settings, storage=storage),
        cache=cache,
    )


def get_chat_service(
    session: DbSession,
    settings: AppSettingsDep,
    documents: DocumentServiceDep,
    llm: LLMDep,
    embedder: EmbedderDep,
    cache: CacheDep,
) -> ChatService:
    return ChatService(
        session,
        settings=settings,
        llm=llm,
        embedder=embedder,
        documents=documents,
        cache=cache,
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


def get_metrics_service(session: DbSession) -> MetricsService:
    return MetricsService(session)


MetricsServiceDep = Annotated[MetricsService, Depends(get_metrics_service)]

__all__ = [
    "AppSettingsDep",
    "ChatServiceDep",
    "build_chat_service",
    "CurrentPrincipal",
    "DbSession",
    "DocumentServiceDep",
    "MetricsServiceDep",
    "Pagination",
    "StorageDep",
]
