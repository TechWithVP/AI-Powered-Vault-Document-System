"""AI provider interfaces and value objects.

Everything the application needs from an AI vendor is expressed here, so the
rest of the codebase never imports a vendor SDK. Swapping providers means
adding one implementation module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts and cost for a single provider call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: Decimal = Decimal("0")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": float(self.cost_usd),
        }


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a provider conversation."""

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """A chat completion plus the usage it consumed."""

    content: str
    usage: TokenUsage
    model: str
    latency_ms: int = 0


@dataclass(frozen=True, slots=True)
class StructuredResult:
    """A schema-validated JSON completion."""

    data: dict[str, Any]
    usage: TokenUsage
    model: str
    latency_ms: int = 0


@dataclass
class StreamAccumulator:
    """Collects the full result of a streamed completion.

    Streaming yields text as it arrives, but callers still need the finished
    text, the model and the token usage afterwards to persist and cost the
    turn. The accumulator carries those out of the stream.
    """

    content: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""

    def append(self, delta: str) -> None:
        self.content += delta


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Embedding vectors for a batch of inputs, in request order."""

    vectors: list[list[float]] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""

    def __len__(self) -> int:
        return len(self.vectors)


class EmbeddingProvider(ABC):
    """Turns text into vectors for semantic search."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the embedding model in use."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Width of the produced vectors."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed a batch of passages for storage."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""


class LLMProvider(ABC):
    """Generates completions, optionally constrained to a JSON schema."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the chat model in use."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        """Produce a free-form completion."""

    @abstractmethod
    def complete_structured(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
    ) -> StructuredResult:
        """Produce a completion guaranteed to match the given JSON schema."""

    @abstractmethod
    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        accumulator: StreamAccumulator | None = None,
    ) -> AsyncIterator[str]:
        """Yield completion text incrementally.

        When an accumulator is supplied it receives the full text, the model
        and the token usage once the stream completes.
        """
