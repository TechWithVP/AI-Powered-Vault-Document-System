"""Chat orchestration: condensation, grounding and refusal."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ai.base import ChatMessage as ProviderMessage
from app.ai.base import CompletionResult, LLMProvider, StructuredResult, TokenUsage
from app.ai.question import QuestionKind
from app.ai.rag.retriever import RetrievalOutcome
from app.core.config import Settings
from app.core.constants import MessageRole
from app.models.chat import ChatSession
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.repositories.chunk import ScoredChunk
from app.services.chat_service import ChatService

DOCUMENT_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()


class RecordingLLM(LLMProvider):
    """LLM double returning canned answers and recording every call."""

    def __init__(self, answer: str = "The limit is 500 USD [1].") -> None:
        self.answer = answer
        self.completions: list[list[ProviderMessage]] = []
        self.structured: list[str] = []

    @property
    def model(self) -> str:
        return "test-model"

    def complete(
        self,
        messages: Sequence[ProviderMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        self.completions.append(list(messages))
        # Short budgets belong to the condenser and title calls.
        content = "standalone question" if max_tokens and max_tokens <= 120 else self.answer
        return CompletionResult(
            content=content,
            usage=TokenUsage(
                prompt_tokens=50, completion_tokens=20, cost_usd=Decimal("0.0005")
            ),
            model=self.model,
        )

    def complete_structured(
        self,
        messages: Sequence[ProviderMessage],
        *,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
    ) -> StructuredResult:
        self.structured.append(schema_name)
        return StructuredResult(
            data={"questions": ["What about contractors?", "Who approves it?"]},
            usage=TokenUsage(prompt_tokens=20, cost_usd=Decimal("0.0001")),
            model=self.model,
        )

    def stream(self, messages, *, max_tokens=None, temperature=None):
        raise NotImplementedError


def make_session() -> ChatSession:
    document = Document(
        id=DOCUMENT_ID,
        user_id="user-1",
        filename="policy.pdf",
        file_type="pdf",
        file_size_bytes=10,
        storage_key="k",
    )
    session = ChatSession(id=SESSION_ID, user_id="user-1", title=None)
    session.documents = [document]
    session.message_count = 0
    return session


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
    from app.core.config import get_settings

    settings: Settings = get_settings()
    chat = ChatService(
        MagicMock(),
        settings=settings,
        llm=RecordingLLM(),
        embedder=MagicMock(),
        documents=MagicMock(),
    )
    chat.get_session = AsyncMock(return_value=make_session())
    chat._messages = MagicMock()
    chat._messages.recent_turns = AsyncMock(return_value=[])
    chat._messages.flush = AsyncMock()
    chat._messages.add = MagicMock(side_effect=lambda message: message)
    chat._retrieval_metrics = MagicMock()
    chat._retrieval_metrics.record = MagicMock()
    return chat


@pytest.fixture
def principal():
    from app.core.security import Principal

    return Principal(user_id="user-1")


class TestGroundedAnswers:
    async def test_answer_is_returned_with_its_citation(
        self, service: ChatService, principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        result = await service.ask(SESSION_ID, principal, "What is the limit?")

        assert result.grounded is True
        assert result.message.content == "The limit is 500 USD [1]."
        assert [c.marker for c in result.citations] == [1]
        assert result.citations[0].page_number == 2

    async def test_follow_up_suggestions_are_attached(
        self, service: ChatService, principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        result = await service.ask(SESSION_ID, principal, "What is the limit?")

        assert result.suggestions == [
            "What about contractors?",
            "Who approves it?",
        ]

    async def test_suggestions_can_be_disabled(
        self, service: ChatService, principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        result = await service.ask(
            SESSION_ID, principal, "What is the limit?", suggest_follow_ups=False
        )

        assert result.suggestions == []
        assert service._llm.structured == []

    async def test_usage_and_latency_are_recorded(
        self, service: ChatService, principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        result = await service.ask(SESSION_ID, principal, "What is the limit?")

        assert result.message.latency_ms is not None
        assert result.message.token_usage["total_tokens"] > 0
        assert result.message.model == "test-model"

    async def test_an_uncited_answer_is_marked_ungrounded(
        self, service: ChatService, principal
    ) -> None:
        service._llm.answer = "The limit is 500 USD."  # no [n] marker
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        result = await service.ask(SESSION_ID, principal, "What is the limit?")

        assert result.grounded is False
        assert result.citations == []


class TestRefusal:
    async def test_irrelevant_retrieval_produces_an_honest_refusal(
        self, service: ChatService, principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored(0.05)], 0.05, 0.25)
        )

        result = await service.ask(SESSION_ID, principal, "Pet insurance policy?")

        assert result.grounded is False
        assert result.citations == []
        assert "could not find" in result.message.content

    async def test_the_model_is_never_asked_to_answer_from_weak_context(
        self, service: ChatService, principal
    ) -> None:
        """The refusal must not reach the answering prompt at all.

        Titling still runs so the session stays identifiable, so the invariant
        is checked against the answer prompt rather than call count.
        """
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([], 0.0, 0.25)
        )

        await service.ask(SESSION_ID, principal, "Unrelated question?")

        prompts = [
            message.content
            for call in service._llm.completions
            for message in call
        ]
        assert not any("Excerpts from my documents" in text for text in prompts)

    async def test_the_refusal_names_the_single_document(
        self, service: ChatService, principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([], 0.0, 0.25)
        )

        result = await service.ask(SESSION_ID, principal, "Unrelated?")

        assert "policy.pdf" in result.message.content


class TestMultiTurnContext:
    async def test_the_first_question_is_not_condensed(
        self, service: ChatService, principal
    ) -> None:
        service._messages.recent_turns = AsyncMock(return_value=[])
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await service.ask(SESSION_ID, principal, "What is the limit?")

        assert service._retriever.retrieve.await_args.args[0] == "What is the limit?"

    async def test_a_follow_up_is_condensed_before_retrieval(
        self, service: ChatService, principal
    ) -> None:
        history = [
            MagicMock(role=MessageRole.USER, content="What is the furniture limit?"),
            MagicMock(role=MessageRole.ASSISTANT, content="500 USD per year."),
        ]
        service._messages.recent_turns = AsyncMock(return_value=history)
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await service.ask(SESSION_ID, principal, "And internet?")

        assert service._retriever.retrieve.await_args.args[0] == "standalone question"

    async def test_condenser_failure_falls_back_to_the_raw_question(
        self, service: ChatService, principal
    ) -> None:
        history = [MagicMock(role=MessageRole.USER, content="Earlier question")]
        service._messages.recent_turns = AsyncMock(return_value=history)
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        def explode(messages, **kwargs):
            if kwargs.get("max_tokens") == 120:
                raise RuntimeError("provider down")
            return CompletionResult(
                content="Answer [1].", usage=TokenUsage(), model="test-model"
            )

        service._llm.complete = explode

        result = await service.ask(SESSION_ID, principal, "And internet?")

        assert service._retriever.retrieve.await_args.args[0] == "And internet?"
        assert result.grounded is True


class TestSessionTitling:
    async def test_the_first_exchange_names_the_session(
        self, service: ChatService, principal
    ) -> None:
        session = make_session()
        service.get_session = AsyncMock(return_value=session)
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await service.ask(SESSION_ID, principal, "What is the limit?")

        assert session.title is not None
        assert session.message_count == 2

    async def test_an_existing_title_is_left_alone(
        self, service: ChatService, principal
    ) -> None:
        session = make_session()
        session.title = "Chosen by the user"
        service.get_session = AsyncMock(return_value=session)
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await service.ask(SESSION_ID, principal, "What is the limit?")

        assert session.title == "Chosen by the user"


class TestCitationNumbering:
    async def test_markers_are_renumbered_to_match_the_listed_sources(
        self, service: ChatService, principal
    ) -> None:
        """An answer citing retrieval slots 2 and 3 must display [1] and [2]."""
        service._llm.answer = "Both apply [2][3]."
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome(
                [scored(0.7), scored(0.6), scored(0.5)], 0.7, 0.25
            )
        )

        result = await service.ask(SESSION_ID, principal, "What applies?")

        assert result.message.content == "Both apply [1][2]."
        assert [citation.marker for citation in result.citations] == [1, 2]

    async def test_a_marker_for_an_excerpt_that_was_never_supplied_is_dropped(
        self, service: ChatService, principal
    ) -> None:
        service._llm.answer = "Supported [1] and invented [9]."
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        result = await service.ask(SESSION_ID, principal, "What applies?")

        assert "[9]" not in result.message.content
        assert len(result.citations) == 1


class TestConversationQuestions:
    async def test_a_question_about_the_chat_is_answered_from_history(
        self, service: ChatService, principal
    ) -> None:
        """Previously refused, even though the transcript held the answer."""
        history = [
            MagicMock(role=MessageRole.USER, content="What is the limit?"),
            MagicMock(role=MessageRole.ASSISTANT, content="500 USD per year."),
        ]
        service._messages.recent_turns = AsyncMock(return_value=history)
        service._retriever.retrieve = AsyncMock()

        result = await service.ask(
            SESSION_ID, principal, "what was the first question i asked?"
        )

        assert result.source.value == "conversation"
        assert result.citations == []
        service._retriever.retrieve.assert_not_awaited()

    async def test_document_questions_still_use_retrieval(
        self, service: ChatService, principal
    ) -> None:
        service._messages.recent_turns = AsyncMock(
            return_value=[MagicMock(role=MessageRole.USER, content="Earlier")]
        )
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        result = await service.ask(SESSION_ID, principal, "What is the limit?")

        assert result.source.value == "documents"
        service._retriever.retrieve.assert_awaited()


class TestAggregateQuestions:
    async def test_counting_questions_retrieve_more_widely(
        self, service: ChatService, principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await service.ask(SESSION_ID, principal, "How many exclusions are listed?")

        requested = service._retriever.retrieve.await_args.kwargs["top_k"]
        assert requested == service._settings.rag.aggregate_top_k
        assert requested > service._settings.rag.top_k

    async def test_the_model_is_told_the_excerpts_may_be_incomplete(
        self, service: ChatService, principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await service.ask(SESSION_ID, principal, "List all the exclusions")

        # The turn also makes a title call, so scan every system prompt sent.
        prompts = [call[0].content for call in service._llm.completions]
        assert any("may be incomplete" in prompt for prompt in prompts)

    async def test_ordinary_questions_carry_no_caveat(
        self, service: ChatService, principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await service.ask(SESSION_ID, principal, "What is the limit?")

        prompts = [call[0].content for call in service._llm.completions]
        assert not any("may be incomplete" in prompt for prompt in prompts)


class TestRetrievalAttribution:
    """Every search records which arm it leaned on."""

    async def test_an_answered_question_records_its_split(
        self, service: ChatService, principal
    ) -> None:
        outcome = RetrievalOutcome(
            [scored().with_fusion(semantic=0.75, lexical=0.25)], 0.7, 0.25
        )
        service._retriever.retrieve = AsyncMock(return_value=outcome)

        await service.ask(SESSION_ID, principal, "What is the limit?")

        service._retrieval_metrics.record.assert_called_once()
        call = service._retrieval_metrics.record.call_args
        assert call.args[0] is outcome
        assert call.kwargs["question_kind"] == QuestionKind.DOCUMENT.value
        assert call.kwargs["session_id"] == SESSION_ID

    async def test_a_refusal_records_its_split_too(
        self, service: ChatService, principal
    ) -> None:
        """A search that found nothing is still a search worth counting."""
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([], 0.0, 0.25)
        )

        await service.ask(SESSION_ID, principal, "Who won the world cup?")

        service._retrieval_metrics.record.assert_called_once()
        assert service._retrieval_metrics.record.call_args.args[0].is_relevant is False

    async def test_the_kind_recorded_is_the_classified_one(
        self, service: ChatService, principal
    ) -> None:
        service._retriever.retrieve = AsyncMock(
            return_value=RetrievalOutcome([scored()], 0.7, 0.25)
        )

        await service.ask(SESSION_ID, principal, "How many exclusions are there?")

        kind = service._retrieval_metrics.record.call_args.kwargs["question_kind"]
        assert kind == QuestionKind.AGGREGATE.value
