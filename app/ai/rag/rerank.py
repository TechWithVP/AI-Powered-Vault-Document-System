"""Reranking the fused candidate list.

Retrieval optimises for recall: cast a wide net so the answering passage is
somewhere in the candidates. Ranking that net well is a different problem.
Cosine distance and ts_rank both score a passage without ever considering the
question as a whole - one compares an averaged embedding, the other counts
stems - so a passage that merely shares vocabulary with the question can
outrank the one that answers it.

A reranker reads each candidate against the question and scores it directly.
It cannot find what retrieval missed, so its ceiling is the recall of the
candidate pool, not of the corpus.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.ai.base import ChatMessage, LLMProvider, StructuredResult, TokenUsage
from app.ai.prompts import build_rerank_messages
from app.ai.schemas import RERANK_SCHEMA
from app.core.config import RagSettings
from app.core.logging import get_logger
from app.repositories.chunk import ScoredChunk

logger = get_logger(__name__)

#: Scores below this are treated as the model declaring a passage irrelevant.
#: It is deliberately low: the relevance gate is the component that refuses, and
#: a reranker that silently discards everything would turn a ranking problem
#: into a refusal.
MIN_RELEVANCE_SCORE = 1

#: Fraction of passages that must come back scored for the ranking to be used.
#: Structured output guarantees the shape of the response, not that the model
#: filled it in: asked to score 40 passages it has been measured returning 5,
#: silently. A partial ranking is not a weaker ranking, it is a different one -
#: most candidates would be placed by an opinion the model never gave - so
#: below this threshold the retrieval order stands instead.
MIN_SCORE_COVERAGE = 0.9

#: Characters of each candidate shown to the reranker. Full chunks are the
#: point - truncation would hide the passage the score depends on - but a
#: pathological chunk must not blow the context budget on its own.
MAX_CANDIDATE_CHARS = 2_000


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    """Reordered candidates plus what the reordering cost."""

    results: list[ScoredChunk]
    usage: TokenUsage = field(default_factory=TokenUsage)
    #: False when reranking was skipped or failed and the input order stands.
    applied: bool = False


class Reranker(ABC):
    """Reorders candidates by how well each answers the question."""

    @abstractmethod
    async def rerank(
        self, query: str, candidates: list[ScoredChunk], *, limit: int
    ) -> RerankOutcome:
        """Return the best `limit` candidates, most relevant first."""


class NullReranker(Reranker):
    """Keeps the fused order. The default, and the fallback on failure."""

    async def rerank(
        self, query: str, candidates: list[ScoredChunk], *, limit: int
    ) -> RerankOutcome:
        return RerankOutcome(results=candidates[:limit])


class LLMReranker(Reranker):
    """Scores candidates with a single structured LLM call.

    One call for the whole list rather than one per candidate: the cost is
    dominated by the passages themselves, and scoring them together lets the
    model judge relative relevance instead of guessing an absolute scale.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        max_candidate_chars: int = MAX_CANDIDATE_CHARS,
    ) -> None:
        self._llm = llm
        self._max_candidate_chars = max_candidate_chars

    async def rerank(
        self, query: str, candidates: list[ScoredChunk], *, limit: int
    ) -> RerankOutcome:
        if len(candidates) <= 1:
            return RerankOutcome(results=candidates[:limit])

        passages = [self._render(item) for item in candidates]
        try:
            result = await self._score(build_rerank_messages(query, passages))
        except Exception as exc:
            # A reranker is an optimisation. Losing it degrades ranking; it
            # must not turn a usable answer into an error.
            logger.warning("Reranking failed; keeping fused order",
                           extra={"error": str(exc)})
            return RerankOutcome(results=candidates[:limit])

        scores = self._parse(result.data, len(candidates))
        coverage = len(scores) / len(candidates)
        if coverage < MIN_SCORE_COVERAGE:
            logger.warning(
                "Reranker scored too few passages; keeping fused order",
                extra={
                    "candidates": len(candidates),
                    "scored": len(scores),
                    "coverage": round(coverage, 2),
                },
            )
            return RerankOutcome(
                results=candidates[:limit], usage=result.usage
            )

        ordered = self._apply(candidates, scores)
        logger.info(
            "Reranking completed",
            extra={
                "candidates": len(candidates),
                "scored": len(scores),
                "kept": len(ordered[:limit]),
                "cost_usd": float(result.usage.cost_usd),
            },
        )
        return RerankOutcome(
            results=ordered[:limit], usage=result.usage, applied=True
        )

    async def _score(self, messages: list[ChatMessage]) -> StructuredResult:
        # The provider SDK is blocking; keep the event loop free.
        return await asyncio.to_thread(
            self._llm.complete_structured,
            messages,
            schema=RERANK_SCHEMA,
            schema_name="passage_relevance",
        )

    def _render(self, item: ScoredChunk) -> str:
        text = " ".join(item.chunk.content.split())
        if len(text) <= self._max_candidate_chars:
            return text
        return f"{text[: self._max_candidate_chars].rstrip()}..."

    @staticmethod
    def _parse(data: dict[str, object], count: int) -> dict[int, int]:
        """Extract position -> score, ignoring anything out of range."""
        scores: dict[int, int] = {}
        entries = data.get("scores") or []
        if not isinstance(entries, list):
            return scores
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            position, score = entry.get("position"), entry.get("score")
            if (
                isinstance(position, int)
                and isinstance(score, int)
                and 1 <= position <= count
            ):
                scores[position] = score
        return scores

    @staticmethod
    def _apply(
        candidates: list[ScoredChunk], scores: dict[int, int]
    ) -> list[ScoredChunk]:
        """Order scored candidates by score, then the rest by retrieval rank.

        An unscored position is an absent opinion, so it is neither promoted
        above a judged passage nor dropped - it keeps its retrieval order
        behind everything the model actually ranked. Only an explicit low score
        removes a candidate.
        """
        judged = [
            (position, item)
            for position, item in enumerate(candidates, start=1)
            if position in scores
        ]
        unjudged = [
            item
            for position, item in enumerate(candidates, start=1)
            if position not in scores
        ]
        judged.sort(key=lambda pair: (-scores[pair[0]], pair[0]))
        kept = [
            item
            for position, item in judged
            if scores[position] >= MIN_RELEVANCE_SCORE
        ]
        return kept + unjudged


def build_reranker(settings: RagSettings, llm: LLMProvider) -> Reranker:
    """The reranker configuration asks for, or a no-op when it is disabled."""
    if not settings.rerank_enabled:
        return NullReranker()
    return LLMReranker(llm)
