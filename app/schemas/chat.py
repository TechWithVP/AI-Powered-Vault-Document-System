"""Chat API schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.ai.rag import Citation
from app.core.constants import MessageRole
from app.models.chat import ChatMessage, ChatSession
from app.services.chat_service import AnswerResult, AnswerSource


class CreateSessionRequest(BaseModel):
    """Start a chat session over selected documents, or the whole vault."""

    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        description=(
            "Documents to chat with. Omit or send null to include every "
            "processed document you own."
        ),
    )
    title: Annotated[str | None, Field(max_length=256)] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"document_ids": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"]},
                {"document_ids": None, "title": "Policy questions"},
            ]
        }
    )


class AskRequest(BaseModel):
    """Ask a question within a session."""

    question: Annotated[str, Field(min_length=1, max_length=4000)]
    suggest_follow_ups: bool = Field(
        default=True,
        description="Return suggested follow-up questions alongside the answer.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"question": "How many remote days are allowed?"}]
        }
    )


class DocumentRef(BaseModel):
    """A document attached to a chat session."""

    id: uuid.UUID
    filename: str


class SessionResponse(BaseModel):
    """A chat session and the documents it covers."""

    id: uuid.UUID
    title: str | None
    message_count: int
    documents: list[DocumentRef]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, session: ChatSession) -> "SessionResponse":
        return cls(
            id=session.id,
            title=session.title,
            message_count=session.message_count,
            documents=[
                DocumentRef(id=document.id, filename=document.filename)
                for document in session.documents
            ],
            created_at=session.created_at,
            updated_at=session.updated_at,
        )


class CitationResponse(BaseModel):
    """A source excerpt supporting part of an answer."""

    marker: int = Field(description="The [n] marker used in the answer text.")
    document_id: uuid.UUID
    document_name: str
    page_number: int | None = None
    snippet: str = Field(description="The quoted passage the claim rests on.")
    similarity: float = Field(
        description="Cosine similarity; zero if found only by exact-term match."
    )
    lexical_rank: float = Field(
        default=0.0,
        description="Full-text rank; zero if found only by vector similarity.",
    )

    @classmethod
    def from_domain(cls, citation: Citation) -> "CitationResponse":
        return cls(**citation.as_dict())


class MessageResponse(BaseModel):
    """One message in a conversation."""

    id: uuid.UUID
    role: MessageRole
    content: str
    citations: list[CitationResponse] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)
    grounded: bool | None = None
    latency_ms: int | None = None
    model: str | None = None
    created_at: datetime

    @classmethod
    def from_model(cls, message: ChatMessage) -> "MessageResponse":
        return cls(
            id=message.id,
            role=message.role,
            content=message.content,
            citations=[
                CitationResponse(**citation) for citation in message.citations
            ],
            suggested_questions=message.suggested_questions,
            grounded=message.grounded,
            latency_ms=message.latency_ms,
            model=message.model,
            created_at=message.created_at,
        )


class AnswerResponse(BaseModel):
    """An answer with the sources that support it."""

    message_id: uuid.UUID
    answer: str
    citations: list[CitationResponse]
    suggested_questions: list[str]
    grounded: bool = Field(
        description=(
            "True when the answer is supported by cited document excerpts. "
            "False for an honest refusal, or for an answer drawn from the "
            "conversation rather than the documents - see `source`."
        )
    )
    source: AnswerSource = Field(
        default=AnswerSource.DOCUMENTS,
        description=(
            "Where the answer came from: the documents, or this conversation "
            "(for questions about the exchange itself)."
        ),
    )
    latency_ms: int | None = None
    model: str | None = None

    @classmethod
    def from_result(cls, result: AnswerResult) -> "AnswerResponse":
        return cls(
            message_id=result.message.id,
            answer=result.message.content,
            citations=[
                CitationResponse.from_domain(citation)
                for citation in result.citations
            ],
            suggested_questions=result.suggestions,
            grounded=result.grounded,
            source=result.source,
            latency_ms=result.message.latency_ms,
            model=result.message.model,
        )
