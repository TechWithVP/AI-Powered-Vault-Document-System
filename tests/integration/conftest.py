"""Fixtures for tests that exercise a real PostgreSQL database.

These run against the compose Postgres service. Each test gets a session whose
transaction is rolled back at the end, so tests never see each other's rows and
the database needs no cleanup between runs.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.constants import DocumentStatus, FileType
from app.db.session import db_manager
from app.models.chat import ChatSession
from app.models.document import Document


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncSession]:
    """An isolated session; everything it writes is rolled back afterwards."""
    connection = await db_manager.async_engine.connect()
    transaction = await connection.begin()
    factory = async_sessionmaker(bind=connection, expire_on_commit=False)

    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()
            await connection.close()


@pytest.fixture
def user_id() -> str:
    """A unique owner per test, so rows can never collide."""
    return f"test-user-{uuid.uuid4().hex[:12]}"


def make_document(
    user_id: str,
    *,
    filename: str = "policy.pdf",
    file_type: FileType = FileType.PDF,
    status: DocumentStatus = DocumentStatus.READY,
    content_hash: str | None = None,
) -> Document:
    return Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        file_size_bytes=2048,
        content_hash=content_hash,
        storage_key=f"documents/test/{uuid.uuid4()}.{file_type.value}",
        status=status,
        page_count=3,
        chunk_count=2,
    )


def make_chat_session(user_id: str, *, title: str | None = None) -> ChatSession:
    """A persisted conversation, for rows that hang off a session."""
    return ChatSession(id=uuid.uuid4(), user_id=user_id, title=title)
