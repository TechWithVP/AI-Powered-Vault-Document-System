"""Chat session and message persistence."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.constants import MessageRole
from app.models.chat import ChatMessage, ChatSession
from app.repositories.base import AsyncRepository


class ChatSessionRepository(AsyncRepository[ChatSession]):
    """Async access to chat sessions."""

    model = ChatSession

    async def get_for_user(
        self, session_id: uuid.UUID, user_id: str
    ) -> ChatSession | None:
        statement = (
            select(ChatSession)
            .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .options(selectinload(ChatSession.documents))
        )
        return await self.session.scalar(statement)

    async def list_page(
        self, user_id: str, *, limit: int, offset: int
    ) -> tuple[list[ChatSession], int]:
        statement = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .options(selectinload(ChatSession.documents))
        )
        total = await self.count(statement)
        page = await self.list(statement.limit(limit).offset(offset))
        return page, total


class ChatMessageRepository(AsyncRepository[ChatMessage]):
    """Async access to chat messages."""

    model = ChatMessage

    async def list_for_session(
        self, session_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[ChatMessage], int]:
        statement = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        total = await self.count(statement)
        page = await self.list(statement.limit(limit).offset(offset))
        return page, total

    async def recent_turns(
        self, session_id: uuid.UUID, *, turns: int
    ) -> list[ChatMessage]:
        """The latest messages in chronological order, for prompt history.

        Fetched newest-first then reversed so the window stays anchored to the
        end of the conversation as it grows.
        """
        statement = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(turns * 2)  # a turn is a user message plus its answer
        )
        messages = await self.list(statement)
        return list(reversed(messages))

    async def last_question(self, session_id: uuid.UUID) -> ChatMessage | None:
        statement = (
            select(ChatMessage)
            .where(
                ChatMessage.session_id == session_id,
                ChatMessage.role == MessageRole.USER,
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        )
        return await self.session.scalar(statement)
