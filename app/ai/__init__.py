"""AI package: provider interfaces and their OpenAI implementations."""

from functools import lru_cache

from app.ai.base import (
    ChatMessage,
    CompletionResult,
    EmbeddingProvider,
    EmbeddingResult,
    LLMProvider,
    StructuredResult,
    TokenUsage,
)
from app.ai.openai_provider import OpenAIEmbeddingProvider, OpenAILLMProvider
from app.core.config import get_settings

__all__ = [
    "ChatMessage",
    "CompletionResult",
    "EmbeddingProvider",
    "EmbeddingResult",
    "LLMProvider",
    "StructuredResult",
    "TokenUsage",
    "get_embedding_provider",
    "get_llm_provider",
]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Return the process-wide embedding provider."""
    return OpenAIEmbeddingProvider(get_settings().openai)


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Return the process-wide chat provider."""
    return OpenAILLMProvider(get_settings().openai)
