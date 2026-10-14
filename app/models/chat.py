"""Chat session and message ORM models."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import MessageRole
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import StrEnumType
from app.models.document import Document

#: Association table scoping a chat session to a set of documents. A real table
#: (rather than a JSON array of ids) preserves referential integrity and lets
#: retrieval filter by join.
chat_session_documents = Table(
    "chat_session_documents",
    Base.metadata,
    Column(
        "session_id",
        PgUUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "document_id",
        PgUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    UniqueConstraint("session_id", "document_id", name="uq_session_document"),
)


class ChatSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A multi-turn conversation scoped to one or more documents."""

    __tablename__ = "chat_sessions"
    __table_args__ = (Index("ix_sessions_user_updated", "user_id", "updated_at"),)

    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(256))
    message_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    documents: Mapped[list[Document]] = relationship(
        secondary=chat_session_documents,
        lazy="selectin",
    )
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessage.created_at",
    )

    @property
    def document_ids(self) -> list[uuid.UUID]:
        return [document.id for document in self.documents]

    @property
    def is_multi_document(self) -> bool:
        return len(self.documents) > 1


class ChatMessage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A single turn in a chat session.

    Assistant turns carry the citations that ground the answer plus the token
    usage and latency captured when it was generated.
    """

    __tablename__ = "chat_messages"
    __table_args__ = (Index("ix_messages_session_created", "session_id", "created_at"),)

    session_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[MessageRole] = mapped_column(
        StrEnumType(MessageRole, 16), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    suggested_questions: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    token_usage: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(String(64))
    grounded: Mapped[bool | None] = mapped_column()

    session: Mapped[ChatSession] = relationship(back_populates="messages")

    @property
    def is_assistant(self) -> bool:
        return self.role is MessageRole.ASSISTANT
