"""OpenAI implementations of the AI provider interfaces."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai
from openai import AsyncOpenAI, OpenAI

from app.ai.base import (
    ChatMessage,
    CompletionResult,
    EmbeddingProvider,
    EmbeddingResult,
    LLMProvider,
    StreamAccumulator,
    StructuredResult,
    TokenUsage,
)
from app.ai.pricing import cost_calculator
from app.core.config import OpenAISettings
from app.core.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Default ceiling for one completion; callers may override per request.
DEFAULT_MAX_TOKENS = 4096


def translate_provider_error(exc: Exception) -> LLMError:
    """Map vendor exceptions onto the application error hierarchy."""
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError()
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError()
    if isinstance(exc, openai.AuthenticationError):
        return LLMError("The AI provider rejected the configured API key.")
    if isinstance(exc, openai.BadRequestError):
        return LLMError(f"The AI provider rejected the request: {exc}")
    if isinstance(exc, openai.APIError):
        return LLMError(f"The AI provider returned an error: {exc}")
    return LLMError(f"Unexpected AI provider failure: {exc}")


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings via the OpenAI embeddings endpoint.

    The SDK retries transient failures itself, so this class concentrates on
    ordering, usage accounting and error translation.
    """

    def __init__(self, settings: OpenAISettings, client: OpenAI | None = None) -> None:
        self._settings = settings
        self._client = client or OpenAI(
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    @property
    def model(self) -> str:
        return self._settings.embedding_model

    @property
    def dimensions(self) -> int:
        return self._settings.embedding_dimensions

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(model=self.model)

        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=list(texts),
                dimensions=self.dimensions,
            )
        except Exception as exc:
            raise translate_provider_error(exc) from exc

        # Responses may arrive out of order; the index field is authoritative.
        ordered = sorted(response.data, key=lambda item: item.index)
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0

        return EmbeddingResult(
            vectors=[item.embedding for item in ordered],
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                cost_usd=cost_calculator.calculate(
                    self.model, prompt_tokens=prompt_tokens
                ),
            ),
            model=self.model,
        )

    def embed_query(self, text: str) -> list[float]:
        result = self.embed_documents([text])
        if not result.vectors:
            raise LLMError("The embedding provider returned no vector.")
        return result.vectors[0]


class OpenAILLMProvider(LLMProvider):
    """Chat completions, including schema-constrained structured output."""

    def __init__(
        self,
        settings: OpenAISettings,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or OpenAI(
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )
        self._async_client = async_client or AsyncOpenAI(
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    @property
    def model(self) -> str:
        return self._settings.chat_model

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[message.as_dict() for message in messages],
                max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
                temperature=self._resolve_temperature(temperature),
            )
        except Exception as exc:
            raise translate_provider_error(exc) from exc

        return CompletionResult(
            content=response.choices[0].message.content or "",
            usage=self._usage_from(response),
            model=self.model,
            latency_ms=self._elapsed_ms(started),
        )

    def complete_structured(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
    ) -> StructuredResult:
        """Return JSON guaranteed to satisfy the supplied schema.

        Strict structured outputs remove the need to defensively parse or repair
        model output, so analysis never retries on malformed JSON.
        """
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[message.as_dict() for message in messages],
                max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
                temperature=self._settings.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
        except Exception as exc:
            raise translate_provider_error(exc) from exc

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(
                "The response was truncated before completing. Retry with a "
                "larger token budget."
            )

        try:
            data = json.loads(choice.message.content or "{}")
        except json.JSONDecodeError as exc:
            raise LLMError(f"The AI provider returned malformed JSON: {exc}") from exc

        return StructuredResult(
            data=data,
            usage=self._usage_from(response),
            model=self.model,
            latency_ms=self._elapsed_ms(started),
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        accumulator: StreamAccumulator | None = None,
    ) -> AsyncIterator[str]:
        """Yield answer text as it arrives, capturing usage for costing.

        ``include_usage`` makes the provider emit a final chunk carrying token
        counts, so a streamed answer is billed as accurately as a buffered one.
        """
        if accumulator is not None:
            accumulator.model = self.model

        try:
            stream = await self._async_client.chat.completions.create(
                model=self.model,
                messages=[message.as_dict() for message in messages],
                max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
                temperature=self._resolve_temperature(temperature),
                stream=True,
                stream_options={"include_usage": True},
            )
            async for event in stream:
                if event.choices and (delta := event.choices[0].delta.content):
                    if accumulator is not None:
                        accumulator.append(delta)
                    yield delta
                # The usage chunk arrives last and carries no choices.
                if accumulator is not None and getattr(event, "usage", None):
                    accumulator.usage = self._usage_from(event)
        except Exception as exc:
            raise translate_provider_error(exc) from exc

    def _resolve_temperature(self, override: float | None) -> float:
        return override if override is not None else self._settings.temperature

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)

    def _usage_from(self, response: Any) -> TokenUsage:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_calculator.calculate(
                self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )
