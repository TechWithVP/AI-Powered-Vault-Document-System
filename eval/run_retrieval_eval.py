"""Measure retrieval quality against the labelled set.

Reports each arm separately rather than one blended number, because a change
that helps one arm while breaking another is invisible in the aggregate - the
reason this harness exists is that exactly that happened.

Recall@k asks whether the answering passage reached the model at all. Rank
metrics ask where it landed: a reranker that moves it from sixth to first
changes nothing about recall but a great deal about the answer, so judging a
reranker on recall alone would be judging it on a metric that cannot see it.

The retrieval arms are deterministic. The reranked arm calls an LLM, so its
numbers move between runs - recall@1 has been measured at both 36 and 45
percent on an unchanged corpus. Read a single reranked column as one sample and
run the harness several times before concluding anything from it.

    docker compose exec api python -m eval.run_retrieval_eval
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field

from app.ai import get_embedding_provider, get_llm_provider
from app.ai.rag.rerank import LLMReranker
from app.ai.rag.retriever import ChunkRetriever
from app.core.config import get_settings
from app.db.session import db_manager
from app.repositories.chunk import ChunkRepository, ScoredChunk
from eval.golden_set import SBI_POLICY, SBI_POLICY_CASES, GoldenCase

#: Fail the run below this. Set from the measured baseline, not aspiration.
MIN_HYBRID_RECALL = 0.90


@dataclass
class ArmResult:
    """Where the answering passage landed, across every case."""

    name: str
    #: 1-based rank of the best gold chunk, or None when it was never returned.
    ranks: list[int | None] = field(default_factory=list)
    cost_usd: float = 0.0
    seconds: float = 0.0

    def recall_at(self, k: int) -> float:
        if not self.ranks:
            return 0.0
        found = sum(1 for rank in self.ranks if rank is not None and rank <= k)
        return found / len(self.ranks)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank: rewards placing the answer near the top."""
        if not self.ranks:
            return 0.0
        return sum(1 / rank for rank in self.ranks if rank) / len(self.ranks)


def gold_rank(results: list[ScoredChunk], case: GoldenCase) -> int | None:
    """Position of the first gold chunk, 1-based."""
    for position, item in enumerate(results, start=1):
        if item.chunk.chunk_index in case.gold_chunks:
            return position
    return None


class RetrievalEvaluator:
    """Runs the labelled set through each retrieval arm."""

    def __init__(self, retriever: ChunkRetriever, chunks: ChunkRepository) -> None:
        self._retriever = retriever
        self._chunks = chunks

    async def run(
        self, cases: tuple[GoldenCase, ...], document_id, *, k: int
    ) -> tuple[list[ArmResult], int, int]:
        settings = get_settings()
        answerable = [case for case in cases if case.answerable]
        controls = [case for case in cases if not case.answerable]
        arms = {
            name: ArmResult(name)
            for name in ("semantic", "lexical", "hybrid", "hybrid+rerank")
        }
        pool = max(settings.rag.candidate_pool, k)
        shortlist = max(settings.rag.rerank_candidates, k)
        reranker = LLMReranker(get_llm_provider())
        cost = elapsed = 0.0

        print(f"{'question':<58} {'sem':>5} {'lex':>5} {'hyb':>5} {'rrk':>5}")
        print("-" * 86)
        for case in answerable:
            vector, _ = await self._retriever._embed(case.question)
            semantic = await self._chunks.search(
                vector,
                document_ids=[document_id],
                limit=pool,
                model=settings.openai.embedding_model,
            )
            lexical = await self._retriever._search_lexical(
                case.question, document_ids=[document_id], limit=pool
            )
            fused = self._retriever._fuse(semantic, lexical, limit=shortlist)

            started = time.perf_counter()
            reranked = await reranker.rerank(case.question, fused, limit=k)
            elapsed += time.perf_counter() - started
            cost += float(reranked.usage.cost_usd)

            selected = {
                "semantic": semantic[:k],
                "lexical": lexical[:k],
                "hybrid": fused[:k],
                "hybrid+rerank": reranked.results,
            }
            cells = []
            for name, results in selected.items():
                rank = gold_rank(results, case)
                arms[name].ranks.append(rank)
                cells.append(str(rank) if rank else ".")
            print(f"{case.question:<58} " + " ".join(f"{c:>5}" for c in cells))

        refused = 0
        for case in controls:
            outcome = await self._retriever.retrieve(
                case.question, document_ids=[document_id]
            )
            ok = not outcome.is_relevant
            refused += ok
            print(f"{case.question:<58} {'REFUSED' if ok else 'LEAKED!':>23}")

        arms["hybrid+rerank"].cost_usd = cost
        arms["hybrid+rerank"].seconds = elapsed
        return list(arms.values()), refused, len(controls)


def report(arms: list[ArmResult], k: int) -> None:
    """Print the metric table. Cells are the rank of the answering passage."""
    print("-" * 86)
    print(f"{'arm':<16} {'r@1':>6} {'r@3':>6} {f'r@{k}':>6} {'MRR':>6}   cost")
    for arm in arms:
        extra = ""
        if arm.cost_usd or arm.seconds:
            per_query = arm.seconds / len(arm.ranks) if arm.ranks else 0.0
            extra = f"   +${arm.cost_usd:.4f}, +{per_query:.2f}s/query"
        print(
            f"{arm.name:<16} {arm.recall_at(1):>6.0%} {arm.recall_at(3):>6.0%} "
            f"{arm.recall_at(k):>6.0%} {arm.mrr:>6.2f}{extra}"
        )


async def main() -> int:
    settings = get_settings()
    k = settings.rag.top_k
    async with db_manager.session() as session:
        chunks = ChunkRepository(session)
        retriever = ChunkRetriever(chunks, get_embedding_provider(), settings.rag)
        arms, refused, controls = await RetrievalEvaluator(retriever, chunks).run(
            SBI_POLICY_CASES, SBI_POLICY, k=k
        )

    report(arms, k)
    print(f"controls refused  {refused}/{controls}")

    hybrid = next(arm for arm in arms if arm.name == "hybrid")
    if hybrid.recall_at(k) < MIN_HYBRID_RECALL:
        print(f"\nFAIL: hybrid recall@{k} {hybrid.recall_at(k):.0%} "
              f"< {MIN_HYBRID_RECALL:.0%}")
        return 1
    if refused != controls:
        print("\nFAIL: a control question was not refused")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
