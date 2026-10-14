"""Generic repository base classes.

Repositories isolate persistence from business logic: services talk in terms of
domain operations while SQL construction stays here. Async and sync variants
exist because the API runs on asyncio while Celery workers are synchronous.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from sqlalchemy import Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class _RepositoryBase(Generic[ModelT]):
    """Shared query construction for both session flavours."""

    model: type[ModelT]

    def _select(self) -> Select[tuple[ModelT]]:
        return select(self.model)

    def _by_id(self, entity_id: uuid.UUID) -> Select[tuple[ModelT]]:
        return self._select().where(self.model.id == entity_id)

    @staticmethod
    def _count_of(statement: Select[Any]) -> Select[tuple[int]]:
        """Wrap a statement in a COUNT, discarding ordering and pagination."""
        return select(func.count()).select_from(
            statement.order_by(None).limit(None).offset(None).subquery()
        )


class AsyncRepository(_RepositoryBase[ModelT]):
    """Async CRUD operations used by the API request path."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def add(self, entity: ModelT) -> ModelT:
        """Stage a new entity; the surrounding unit of work commits it."""
        self.session.add(entity)
        return entity

    async def get(self, entity_id: uuid.UUID) -> ModelT | None:
        return await self.session.scalar(self._by_id(entity_id))

    async def list(self, statement: Select[tuple[ModelT]]) -> list[ModelT]:
        result = await self.session.scalars(statement)
        return list(result.unique().all())

    async def count(self, statement: Select[Any]) -> int:
        return await self.session.scalar(self._count_of(statement)) or 0

    async def delete(self, entity: ModelT) -> None:
        await self.session.delete(entity)

    async def delete_by_id(self, entity_id: uuid.UUID) -> int:
        result = await self.session.execute(
            delete(self.model).where(self.model.id == entity_id)
        )
        return result.rowcount or 0

    async def flush(self) -> None:
        """Push pending changes so server-side defaults become readable."""
        await self.session.flush()


class SyncRepository(_RepositoryBase[ModelT]):
    """Sync CRUD operations used by Celery workers."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, entity: ModelT) -> ModelT:
        self.session.add(entity)
        return entity

    def add_all(self, entities: Sequence[ModelT]) -> None:
        self.session.add_all(entities)

    def get(self, entity_id: uuid.UUID) -> ModelT | None:
        return self.session.scalar(self._by_id(entity_id))

    def list(self, statement: Select[tuple[ModelT]]) -> list[ModelT]:
        return list(self.session.scalars(statement).unique().all())

    def flush(self) -> None:
        self.session.flush()
