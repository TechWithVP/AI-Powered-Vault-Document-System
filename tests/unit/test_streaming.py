"""Streaming answers: event sequence, caching and usage capture."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ai.base import ChatMessage as ProviderMessage
from app.ai.base import (
    CompletionResult,
    EmbeddingResult,
    LLMProvider,
    StreamAccumulator,
    StructuredResult,
    TokenUsage,
)
from app.ai.rag.retriever import ChunkRetriever, RetrievalOutcome
from app.core.config import RagSettings, get_settings
from app.core.security import Principal
from app.models.chat import ChatSession
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.repositories.chunk import ScoredChunk
from app.ai.prompts import AGGREGATE_CAVEAT
from app.core.constants import MessageRole
from app.models.chat import ChatMessage
from app.services.chat_service import (
    AnswerSource,
    ChatService,
    StreamEventType,
)

DOCUMENT_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()


class StreamingLLM(LLMProvider):
    """LLM double that yields a canned answer one word at a time."""

    def __init__(self, answer: str = "The limit is 500 USD [1].") -> None:
        self._answer = answer
        self.completions: list[Sequence[ProviderMessage]] = []
        self.streamed: list[Sequence[ProviderMessage]] = []

    @property
    def model(self) -> str:
        return "test-model"

    def complete(self, messages, *, max_tokens=None, temperature=None):
        self.completions.append(messages)
        return CompletionResult(
            content="standalone question", usage=TokenUsage(), model=self.model
        )

    def complete_structured(self, messages, *, schema, schema_name, max_tokens=None):
        return StructuredResult(
            data={"questions": []}, usage=TokenUsage(), model=self.model
        )

    async def stream(
        self,
        messages: Sequence[ProviderMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        accumulator: StreamAccumulator | None = None,
    ) -> AsyncIterator[str]:
        self.streamed.append(messages)
        if accumulator is not None:
            accumulator.model = self.model
        for word in self._answer.split(" "):
            chunk = word + " "
            if accumulator is not None:
                accumulator.append(chunk)
            yield chunk
        if accumulator is not None:
            accumulator.usage = TokenUsage(
                prompt_tokens=80, completion_tokens=12, cost_usd=Decimal("0.0004")
            )


def make_session() -> ChatSession:
    document = Document(
        id=DOCUMENT_ID,
        user_id="user-1",
        filename="policy.pdf",
        file_type="pdf",
        file_size_bytes=10,
        storage_key="k",
    )
    session = ChatSession(id=SESSION_ID, user_id="user-1", title="Existing")
    session.documents = [document]
    session.message_count = 0
    return session


def make_message(role: MessageRole, content: str) -> ChatMessage:
    return ChatMessage(session_id=SESSION_ID, role=role, content=content)


def scored(similarity: float = 0.7) -> ScoredChunk:
    chunk = DocumentChunk(
        document_id=DOCUMENT_ID,
        chunk_index=0,
        content="Employees may claim up to 500 USD per year.",
        page_number=2,
        token_count=12,
    )
    return ScoredChunk(chunk=chunk, similarity=similarity)


@pytest.fixture
def service() -> ChatService:
    chat = ChatService(
        MagicMock(),
        settings=get_settings(),
        llm=StreamingLLM(),
        embedder=MagicMock(),
        documents=MagicMock(),
    )
    chat.get_session = AsyncMock(return_value=make_session())
    chat._messages = MagicMock()
    chat._messages.recent_turns = AsyncMock(return_value=[])
    chat._messages.flush = AsyncMock()
    chat._messages.add = MagicMock(side_effect=lambda message: message)
    chat._usage = MagicMock()
    return chat


@pytest.fixture
def principal() -> Principal:
    return Principal(user_id="user-1")


async def collect(service: ChatService, principal: Principal, question: str) -> list:
    return [event async for event in service.ask_stream(SESSION_ID, principal, question)]


class TestEventSequence:
    async def test_tokens_arrive_before_citations_and_done(
        self, service: ChatService, principal: Principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        events = await collect(service, principal, "What is the limit?")
        kinds = [event.type for event in events]

        assert kinds[0] is StreamEventType.TOKEN
        assert kinds[-2] is StreamEventType.CITATIONS
        assert kinds[-1] is StreamEventType.DONE

    async def test_concatenated_tokens_form_the_answer(
        self, service: ChatService, principal: Principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        events = await collect(service, principal, "What is the limit?")
        streamed = "".join(
            event.data["text"] for event in events if event.type is StreamEventType.TOKEN
        )
        assert "500 USD" in streamed

    async def test_citations_are_resolved_after_generation(
        self, service: ChatService, principal: Principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        events = await collect(service, principal, "What is the limit?")
        citations = next(
            event for event in events if event.type is StreamEventType.CITATIONS
        )
        assert citations.data["citations"][0]["page_number"] == 2

    async def test_done_reports_grounding_and_model(
        self, service: ChatService, principal: Principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        done = (await collect(service, principal, "What is the limit?"))[-1]

        assert done.data["grounded"] is True
        assert done.data["model"] == "test-model"


class TestStreamedRefusal:
    async def test_weak_retrieval_streams_the_refusal(
        self, service: ChatService, principal: Principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([], 0.0, 0.25)
        )

        events = await collect(service, principal, "Unrelated?")

        assert [event.type for event in events] == [
            StreamEventType.TOKEN,
            StreamEventType.DONE,
        ]
        assert events[-1].data["grounded"] is False
        assert "could not find" in events[0].data["text"]


class TestUsageCapture:
    async def test_streamed_usage_is_persisted_for_costing(
        self, service: ChatService, principal: Principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await collect(service, principal, "What is the limit?")

        chat_call = next(
            call
            for call in service._usage.record.call_args_list
            if call.kwargs["operation"].value == "chat"
        )
        assert chat_call.kwargs["completion_tokens"] == 12
        assert chat_call.kwargs["cost_usd"] == Decimal("0.0004")


class TestRetrieverCaching:
    @staticmethod
    def _embedder() -> MagicMock:
        embedder = MagicMock()
        embedder.model = "stub-model"
        embedder.embed_documents = MagicMock(
            return_value=EmbeddingResult(
                vectors=[[0.5] * 4],
                usage=TokenUsage(prompt_tokens=9, cost_usd=Decimal("0.000001")),
                model="stub-model",
            )
        )
        return embedder

    @staticmethod
    def _chunks() -> MagicMock:
        chunks = MagicMock()
        chunks.search = AsyncMock(return_value=[scored()])
        return chunks

    async def test_a_cache_hit_skips_the_provider_and_costs_nothing(self) -> None:
        cache = MagicMock()
        cache.get = AsyncMock(return_value=[0.5] * 4)
        cache.set = AsyncMock()
        embedder = self._embedder()

        outcome = await ChunkRetriever(
            self._chunks(), embedder, RagSettings(), cache=cache
        ).retrieve("q", document_ids=[DOCUMENT_ID])

        embedder.embed_documents.assert_not_called()
        assert outcome.usage.total_tokens == 0

    async def test_a_miss_embeds_then_stores_the_vector(self) -> None:
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        embedder = self._embedder()

        outcome = await ChunkRetriever(
            self._chunks(), embedder, RagSettings(), cache=cache
        ).retrieve("q", document_ids=[DOCUMENT_ID])

        embedder.embed_documents.assert_called_once()
        cache.set.assert_awaited_once()
        assert outcome.usage.prompt_tokens == 9


class TestQuestionKindRouting:
    """Streaming must route questions the same way the blocking path does.

    These diverged once: ask() classified and ask_stream() did not, so the
    conversation and counting behaviour existed only on the endpoint the web
    UI does not use.
    """

    async def test_a_conversation_question_is_answered_from_the_transcript(
        self, service: ChatService, principal: Principal
    ) -> None:
        service._messages.recent_turns = AsyncMock(
            return_value=[
                make_message(MessageRole.USER, "What is the excess?"),
                make_message(MessageRole.ASSISTANT, "It is 10 percent [1]."),
            ]
        )
        service._retriever.retrieve = AsyncMock()

        events = await collect(service, principal, "what did I just ask you?")

        service._retriever.retrieve.assert_not_awaited()
        done = events[-1]
        assert done.type is StreamEventType.DONE
        assert done.data["source"] == AnswerSource.CONVERSATION.value

    async def test_the_transcript_answer_reaches_the_client(
        self, service: ChatService, principal: Principal
    ) -> None:
        service._messages.recent_turns = AsyncMock(
            return_value=[make_message(MessageRole.USER, "What is the excess?")]
        )
        service._retriever.retrieve = AsyncMock()

        events = await collect(service, principal, "what did I ask?")

        tokens = [e.data["text"] for e in events if e.type is StreamEventType.TOKEN]
        assert "".join(tokens) == "standalone question"

    async def test_a_counting_question_retrieves_a_wider_window(
        self, service: ChatService, principal: Principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await collect(service, principal, "how many exclusions are there?")

        settings = get_settings()
        assert (
            service._retriever.retrieve.await_args.kwargs["top_k"]
            == settings.rag.aggregate_top_k
        )

    async def test_a_counting_question_is_told_the_excerpts_are_a_sample(
        self, service: ChatService, principal: Principal
    ) -> None:
        """Without this a total counted from six chunks reads as the real one."""
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await collect(service, principal, "list all the exclusions")

        prompts = [
            message.content
            for messages in service._llm.streamed
            for message in messages
        ]
        assert any(AGGREGATE_CAVEAT in prompt for prompt in prompts)

    async def test_an_ordinary_question_uses_the_normal_window_and_no_caveat(
        self, service: ChatService, principal: Principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await collect(service, principal, "What is the limit?")

        settings = get_settings()
        assert (
            service._retriever.retrieve.await_args.kwargs["top_k"]
            == settings.rag.top_k
        )
        prompts = [
            message.content
            for messages in service._llm.streamed
            for message in messages
        ]
        assert not any(AGGREGATE_CAVEAT in prompt for prompt in prompts)
