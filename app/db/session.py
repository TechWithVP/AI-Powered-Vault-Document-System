"""Database engine and session management.

Two engines are maintained against the same database:

* an **async** engine used by the FastAPI request path, and
* a **sync** engine used by Celery workers and Alembic.

Both are created lazily and owned by :class:`DatabaseSessionManager`, which also
exposes transactional context managers so callers never hand-roll
commit/rollback logic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import PostgresSettings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DatabaseSessionManager:
    """Owns the engines and session factories for the process."""

    def __init__(self, settings: PostgresSettings) -> None:
        self._settings = settings
        self._async_engine: AsyncEngine | None = None
        self._sync_engine: Engine | None = None
        self._async_factory: async_sessionmaker[AsyncSession] | None = None
        self._sync_factory: sessionmaker[Session] | None = None

    # -- engines ----------------------------------------------------------- #
    @property
    def async_engine(self) -> AsyncEngine:
        if self._async_engine is None:
            self._async_engine = create_async_engine(
                self._settings.async_dsn,
                echo=self._settings.echo,
                pool_size=self._settings.pool_size,
                max_overflow=self._settings.max_overflow,
                pool_pre_ping=True,
            )
        return self._async_engine

    @property
    def sync_engine(self) -> Engine:
        if self._sync_engine is None:
            self._sync_engine = create_engine(
                self._settings.sync_dsn,
                echo=self._settings.echo,
                pool_size=self._settings.pool_size,
                max_overflow=self._settings.max_overflow,
                pool_pre_ping=True,
            )
        return self._sync_engine

    # -- session factories -------------------------------------------------- #
    @property
    def async_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._async_factory is None:
            self._async_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
        return self._async_factory

    @property
    def sync_session_factory(self) -> sessionmaker[Session]:
        if self._sync_factory is None:
            self._sync_factory = sessionmaker(
                bind=self.sync_engine,
                expire_on_commit=False,
                autoflush=False,
            )
        return self._sync_factory

    # -- unit-of-work helpers ---------------------------------------------- #
    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Async transactional scope: commits on success, rolls back on error."""
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    @contextmanager
    def sync_session(self) -> Iterator[Session]:
        """Sync transactional scope used by Celery tasks."""
        with self.sync_session_factory() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    # -- lifecycle ---------------------------------------------------------- #
    async def verify_connection(self) -> None:
        """Raise if the database is unreachable (used by the health check)."""
        async with self.async_engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def dispose(self) -> None:
        """Release pooled connections on shutdown."""
        if self._async_engine is not None:
            await self._async_engine.dispose()
            self._async_engine = None
        if self._sync_engine is not None:
            self._sync_engine.dispose()
            self._sync_engine = None
        logger.info("Database connections disposed")


#: Process-wide manager. Engines are created on first use.
db_manager = DatabaseSessionManager(get_settings().postgres)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a transactional async session."""
    async with db_manager.session() as session:
        yield session
