"""Retrieval-augmented chat.

Orchestrates one question end to end: condense the follow-up into a standalone
query, retrieve the supporting passages, refuse honestly when none are relevant,
answer with mandatory citations, then persist the turn with its usage.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.base import (
    EmbeddingProvider,
    LLMProvider,
    StreamAccumulator,
    TokenUsage,
)
from app.ai.cache import EmbeddingCache
from app.ai.prompts import (
    AGGREGATE_CAVEAT,
    build_answer_messages,
    build_conversation_messages,
    build_condense_messages,
    build_suggestion_messages,
    build_title_messages,
)
from app.ai.question import QuestionClassifier, QuestionKind
from app.ai.rag import Citation, CitationMapper, ChunkRetriever, ContextBuilder
from app.ai.schemas import FOLLOW_UP_SCHEMA
from app.core.config import Settings
from app.core.constants import AIOperation, MessageRole
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.core.security import Principal
from app.models.chat import ChatMessage, ChatSession
from app.models.document import Document
from app.repositories.chat import ChatMessageRepository, ChatSessionRepository
from app.repositories.chunk import ChunkRepository
from app.repositories.metrics import ApiUsageRepository
from app.services.document_service import DocumentService

logger = get_logger(__name__)

#: Returned when retrieval finds nothing above the relevance threshold.
NO_ANSWER_TEMPLATE = (
    "I could not find anything about that in {scope}. The documents available "
    "to this conversation cover other topics, so I do not want to guess at an "
    "answer. Try rephrasing the question, or ask about something these "
    "documents address."
)


class AnswerSource(StrEnum):
    """Where an answer came from."""

    #: Grounded in retrieved document excerpts.
    DOCUMENTS = "documents"
    #: Answered from the conversation transcript, not the documents.
    CONVERSATION = "conversation"


@dataclass
class AnswerResult:
    """A completed question-and-answer turn."""

    message: ChatMessage
    citations: list[Citation] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    grounded: bool = True
    usage: TokenUsage = field(default_factory=TokenUsage)
    source: AnswerSource = AnswerSource.DOCUMENTS


class StreamEventType(StrEnum):
    """Server-sent event names emitted while answering."""

    TOKEN = "token"
    CITATIONS = "citations"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One server-sent event in a streamed answer."""

    type: StreamEventType
    data: dict[str, Any]

    @classmethod
    def token(cls, text: str) -> "StreamEvent":
        return cls(StreamEventType.TOKEN, {"text": text})

    @classmethod
    def citations(cls, citations: list[Citation]) -> "StreamEvent":
        return cls(
            StreamEventType.CITATIONS,
            {"citations": [citation.as_dict() for citation in citations]},
        )

    @classmethod
    def done(cls, result: "AnswerResult") -> "StreamEvent":
        return cls(
            StreamEventType.DONE,
            {
                "message_id": str(result.message.id),
                # The streamed tokens carry the model's original markers; the
                # final text has them renumbered, so clients re-render from it.
                "answer": result.message.content,
                "grounded": result.grounded,
                "source": result.source.value,
                "latency_ms": result.message.latency_ms,
                "model": result.message.model,
            },
        )


class ChatService:
    """Creates sessions and answers questions against their documents."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        settings: Settings,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        documents: DocumentService,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._llm = llm
        self._documents = documents
        self._sessions = ChatSessionRepository(session)
        self._messages = ChatMessageRepository(session)
        self._usage = ApiUsageRepository(session)
        self._retriever = ChunkRetriever(
            ChunkRepository(session), embedder, settings.rag, cache=cache
        )
        self._context_builder = ContextBuilder(max_chars=settings.rag.max_context_chars)
        self._classifier = QuestionClassifier()

    # -- sessions ----------------------------------------------------------- #
    async def create_session(
        self,
        principal: Principal,
        *,
        document_ids: list[uuid.UUID] | None,
        title: str | None = None,
    ) -> ChatSession:
        """Start a session scoped to the given documents, or to the whole vault."""
        if document_ids:
            documents = await self._documents.get_ready_documents(
                document_ids, principal
            )
        else:
            documents = await self._documents.list_all_ready(principal)
            if not documents:
                raise NotFoundError(
                    "You have no processed documents to chat with yet. Upload a "
                    "document and wait for its status to become ready."
                )

        session = ChatSession(user_id=principal.user_id, title=title)
        session.documents = documents
        self._sessions.add(session)
        await self._sessions.flush()

        logger.info(
            "Chat session created",
            extra={
                "session_id": str(session.id),
                "documents": len(documents),
            },
        )
        return session

    async def get_session(
        self, session_id: uuid.UUID, principal: Principal
    ) -> ChatSession:
        session = await self._sessions.get_for_user(session_id, principal.user_id)
        if session is None:
            raise NotFoundError.for_resource("Chat session", session_id)
        return session

    async def list_sessions(
        self, principal: Principal, *, limit: int, offset: int
    ) -> tuple[list[ChatSession], int]:
        return await self._sessions.list_page(
            principal.user_id, limit=limit, offset=offset
        )

    async def list_messages(
        self, session_id: uuid.UUID, principal: Principal, *, limit: int, offset: int
    ) -> tuple[list[ChatMessage], int]:
        await self.get_session(session_id, principal)
        return await self._messages.list_for_session(
            session_id, limit=limit, offset=offset
        )

    async def delete_session(
        self, session_id: uuid.UUID, principal: Principal
    ) -> None:
        session = await self.get_session(session_id, principal)
        await self._sessions.delete(session)

    # -- question answering -------------------------------------------------- #
    async def ask(
        self,
        session_id: uuid.UUID,
        principal: Principal,
        question: str,
        *,
        suggest_follow_ups: bool = True,
    ) -> AnswerResult:
        """Answer a question within a session and persist both turns."""
        started = time.perf_counter()
        session = await self.get_session(session_id, principal)
        document_ids = session.document_ids

        history = await self._history(session_id)
        self._messages.add(
            ChatMessage(
                session_id=session_id,
                role=MessageRole.USER,
                content=question,
            )
        )

        kind = self._classifier.classify(question, has_history=bool(history))
        if kind is QuestionKind.CONVERSATION:
            return await self._answer_from_history(
                session, question, history, started
            )

        search_query = await self._condense(question, history)
        outcome = await self._retriever.retrieve(
            search_query,
            document_ids=document_ids,
            top_k=self._top_k_for(kind),
        )

        if not outcome.is_relevant:
            self._record_retrieval_usage(outcome, session_id)
            return await self._refuse(session, question, started)

        names = {document.id: document.filename for document in session.documents}
        context = self._context_builder.build(
            outcome.results, names, query=question
        )

        completion = await asyncio.to_thread(
            self._llm.complete,
            build_answer_messages(
                question,
                context_block=context.block,
                history=history,
                caveat=AGGREGATE_CAVEAT if kind is QuestionKind.AGGREGATE else "",
            ),
        )
        grounded = CitationMapper.is_grounded(
            completion.content, context.citations
        )
        answer, cited = CitationMapper.finalise(
            completion.content.strip(), context.citations
        )
        usage = completion.usage + outcome.usage

        suggestions: list[str] = []
        if suggest_follow_ups and grounded:
            suggestions, suggestion_usage = await self._suggest(
                question, answer, context.block
            )
            usage = usage + suggestion_usage

        message = ChatMessage(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer,
            citations=[citation.as_dict() for citation in cited],
            suggested_questions=suggestions,
            token_usage=usage.as_dict(),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=completion.model,
            grounded=grounded,
        )
        self._messages.add(message)
        self._record_retrieval_usage(outcome, session_id)
        self._usage.record(
            operation=AIOperation.CHAT,
            model=completion.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            session_id=session_id,
            latency_ms=message.latency_ms,
        )
        await self._finalise(session, question, answer, increment=2)

        logger.info(
            "Question answered",
            extra={
                "session_id": str(session_id),
                "citations": len(cited),
                "grounded": grounded,
                "best_similarity": round(outcome.best_similarity, 4),
                "latency_ms": message.latency_ms,
                "cost_usd": float(usage.cost_usd),
            },
        )
        return AnswerResult(
            message=message,
            citations=cited,
            suggestions=suggestions,
            grounded=grounded,
            usage=usage,
        )

    def _top_k_for(self, kind: QuestionKind) -> int:
        """Counting and listing questions need a wider view of the document."""
        if kind is QuestionKind.AGGREGATE:
            return self._settings.rag.aggregate_top_k
        return self._settings.rag.top_k

    async def _answer_from_history(
        self,
        session: ChatSession,
        question: str,
        history: list[tuple[str, str]],
        started: float,
    ) -> AnswerResult:
        """Answer a question about the conversation, not the documents.

        The transcript is already in memory, so refusing these - as the
        retrieval gate otherwise would - withholds an answer the system plainly
        has.
        """
        completion = await asyncio.to_thread(
            self._llm.complete,
            build_conversation_messages(question, history),
            max_tokens=600,
        )
        answer = completion.content.strip()

        message = ChatMessage(
            session_id=session.id,
            role=MessageRole.ASSISTANT,
            content=answer,
            citations=[],
            suggested_questions=[],
            token_usage=completion.usage.as_dict(),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=completion.model,
            # Grounding describes support by document excerpts; this answer is
            # supported by the transcript instead, which `source` records.
            grounded=False,
        )
        self._messages.add(message)
        self._usage.record(
            operation=AIOperation.CHAT,
            model=completion.model,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            cost_usd=completion.usage.cost_usd,
            session_id=session.id,
            latency_ms=message.latency_ms,
        )
        await self._finalise(session, question, answer, increment=2)

        logger.info(
            "Answered from conversation history",
            extra={"session_id": str(session.id)},
        )
        return AnswerResult(
            message=message,
            grounded=False,
            usage=completion.usage,
            source=AnswerSource.CONVERSATION,
        )

    def _record_retrieval_usage(self, outcome, session_id: uuid.UUID) -> None:
        """Attribute the query-embedding cost to the session."""
        if outcome.usage.total_tokens:
            self._usage.record(
                operation=AIOperation.EMBEDDING,
                model=self._settings.openai.embedding_model,
                prompt_tokens=outcome.usage.prompt_tokens,
                completion_tokens=0,
                cost_usd=outcome.usage.cost_usd,
                session_id=session_id,
            )

    async def _refuse(
        self, session: ChatSession, question: str, started: float
    ) -> AnswerResult:
        """Record an honest "not in your documents" answer."""
        scope = (
            "your documents"
            if session.is_multi_document
            else f"'{session.documents[0].filename}'"
            if session.documents
            else "your documents"
        )
        answer = NO_ANSWER_TEMPLATE.format(scope=scope)

        message = ChatMessage(
            session_id=session.id,
            role=MessageRole.ASSISTANT,
            content=answer,
            citations=[],
            suggested_questions=[],
            token_usage=TokenUsage().as_dict(),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=self._llm.model,
            grounded=False,
        )
        self._messages.add(message)
        await self._finalise(session, question, answer, increment=2)

        logger.info(
            "Question refused as unsupported by the documents",
            extra={"session_id": str(session.id)},
        )
        return AnswerResult(message=message, grounded=False)

    # -- helpers ------------------------------------------------------------- #
    async def _history(self, session_id: uuid.UUID) -> list[tuple[str, str]]:
        """Recent turns as (role, content) pairs for prompt construction."""
        messages = await self._messages.recent_turns(
            session_id, turns=self._settings.rag.history_turns
        )
        return [(message.role.value, message.content) for message in messages]

    async def _condense(
        self, question: str, history: list[tuple[str, str]]
    ) -> str:
        """Rewrite a follow-up into a standalone search query.

        Skipped on the first turn, where the question is already standalone and
        the extra call would only add latency and cost.
        """
        if not history:
            return question

        try:
            result = await asyncio.to_thread(
                self._llm.complete,
                build_condense_messages(question, history),
                max_tokens=120,
            )
        except Exception as exc:
            # Retrieval with the raw question is far better than failing the
            # request outright, so a condenser failure degrades silently.
            logger.warning(
                "Question condensation failed; using the raw question",
                extra={"error": str(exc)},
            )
            return question

        condensed = result.content.strip().strip('"')
        return condensed or question

    async def _suggest(
        self, question: str, answer: str, context_block: str
    ) -> tuple[list[str], TokenUsage]:
        count = self._settings.rag.suggestion_count
        try:
            result = await asyncio.to_thread(
                self._llm.complete_structured,
                build_suggestion_messages(
                    question, answer, context_block=context_block, count=count
                ),
                schema=FOLLOW_UP_SCHEMA,
                schema_name="follow_up_questions",
                max_tokens=400,
            )
        except Exception as exc:
            logger.warning(
                "Follow-up suggestion failed", extra={"error": str(exc)}
            )
            return [], TokenUsage()

        raw = result.data.get("questions", [])
        questions = [str(item).strip() for item in raw if str(item).strip()]
        return questions[:count], result.usage

    async def _finalise(
        self, session: ChatSession, question: str, answer: str, *, increment: int
    ) -> None:
        """Update session counters and name the session on its first exchange."""
        session.message_count += increment
        if session.title is None:
            session.title = await self._title(question, answer)
        await self._messages.flush()

    async def _title(self, question: str, answer: str) -> str:
        try:
            result = await asyncio.to_thread(
                self._llm.complete,
                build_title_messages(question, answer),
                max_tokens=32,
            )
            title = result.content.strip().strip('"').rstrip(".")
        except Exception:
            title = ""
        # Fall back to the question itself so a session is always identifiable.
        return (title or question)[:256]

    # -- streaming ----------------------------------------------------------- #
    async def ask_stream(
        self,
        session_id: uuid.UUID,
        principal: Principal,
        question: str,
    ) -> AsyncIterator[StreamEvent]:
        """Answer a question, emitting events as the answer is generated.

        Mirrors :meth:`ask` but yields the answer progressively. Citations are
        only known once generation finishes, so they arrive as a final event
        rather than alongside the text.
        """
        started = time.perf_counter()
        session = await self.get_session(session_id, principal)
        history = await self._history(session_id)

        self._messages.add(
            ChatMessage(
                session_id=session_id, role=MessageRole.USER, content=question
            )
        )

        search_query = await self._condense(question, history)
        outcome = await self._retriever.retrieve(
            search_query, document_ids=session.document_ids
        )

        if not outcome.is_relevant:
            self._record_retrieval_usage(outcome, session_id)
            refusal = await self._refuse(session, question, started)
            yield StreamEvent.token(refusal.message.content)
            yield StreamEvent.done(refusal)
            return

        names = {document.id: document.filename for document in session.documents}
        context = self._context_builder.build(
            outcome.results, names, query=question
        )
        accumulator = StreamAccumulator()

        async for delta in self._llm.stream(
            build_answer_messages(
                question, context_block=context.block, history=history
            ),
            accumulator=accumulator,
        ):
            yield StreamEvent.token(delta)

        grounded = CitationMapper.is_grounded(
            accumulator.content, context.citations
        )
        answer, cited = CitationMapper.finalise(
            accumulator.content.strip(), context.citations
        )
        usage = accumulator.usage + outcome.usage

        message = ChatMessage(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer,
            citations=[citation.as_dict() for citation in cited],
            suggested_questions=[],
            token_usage=usage.as_dict(),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=accumulator.model or self._llm.model,
            grounded=grounded,
        )
        self._messages.add(message)
        self._record_retrieval_usage(outcome, session_id)
        self._usage.record(
            operation=AIOperation.CHAT,
            model=message.model or self._llm.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            session_id=session_id,
            latency_ms=message.latency_ms,
        )
        await self._finalise(session, question, answer, increment=2)

        result = AnswerResult(
            message=message, citations=cited, grounded=grounded, usage=usage
        )
        yield StreamEvent.citations(cited)
        yield StreamEvent.done(result)

        logger.info(
            "Streamed answer completed",
            extra={
                "session_id": str(session_id),
                "citations": len(cited),
                "grounded": grounded,
                "latency_ms": message.latency_ms,
            },
        )
