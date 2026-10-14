"""Measure retrieval quality against the labelled set.

Reports recall@k for each arm separately and for the fusion, so a change that
helps one arm while quietly breaking the other is visible rather than hidden
behind a single aggregate number.

    docker compose exec api python -m eval.run_retrieval_eval
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

from app.ai import get_embedding_provider
from app.ai.rag.retriever import ChunkRetriever
from app.core.config import get_settings
from app.db.session import db_manager
from app.repositories.chunk import ChunkRepository
from eval.golden_set import SBI_POLICY, SBI_POLICY_CASES, GoldenCase

#: Fail the run below this. Set from the measured baseline, not aspiration.
MIN_HYBRID_RECALL = 0.90


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Recall for one retrieval configuration."""

    name: str
    hits: int
    total: int

    @property
    def recall(self) -> float:
        return self.hits / self.total if self.total else 0.0


class RetrievalEvaluator:
    """Runs the labelled set through each retrieval arm."""

    def __init__(self, retriever: ChunkRetriever, chunks: ChunkRepository) -> None:
        self._retriever = retriever
        self._chunks = chunks

    async def run(
        self, cases: tuple[GoldenCase, ...], document_id, *, k: int
    ) -> tuple[list[ArmResult], int, int]:
        answerable = [c for c in cases if c.answerable]
        controls = [c for c in cases if not c.answerable]
        tally = {"semantic": 0, "lexical": 0, "hybrid": 0}
        pool = max(get_settings().rag.candidate_pool, k)

        print(f"{'question':<62} {'sem':>4} {'lex':>4} {'hyb':>4}")
        print("-" * 80)
        for case in answerable:
            vector, _ = await self._retriever._embed(case.question)
            semantic = await self._chunks.search(
                vector, document_ids=[document_id], limit=pool
            )
            lexical = await self._retriever._search_lexical(
                case.question, document_ids=[document_id], limit=pool
            )
            arms = {
                "semantic": semantic[:k],
                "lexical": lexical[:k],
                "hybrid": self._retriever._fuse(semantic, lexical, limit=k),
            }
            marks = []
            for name, results in arms.items():
                hit = bool({r.chunk.chunk_index for r in results} & case.gold_chunks)
                tally[name] += hit
                marks.append("HIT" if hit else " . ")
            print(f"{case.question:<62} {marks[0]:>4} {marks[1]:>4} {marks[2]:>4}")

        refused = 0
        for case in controls:
            outcome = await self._retriever.retrieve(
                case.question, document_ids=[document_id]
            )
            ok = not outcome.is_relevant
            refused += ok
            print(f"{case.question:<62} {'REFUSED' if ok else 'LEAKED!':>14}")

        results = [ArmResult(n, h, len(answerable)) for n, h in tally.items()]
        return results, refused, len(controls)


async def main() -> int:
    settings = get_settings()
    k = settings.rag.top_k
    async with db_manager.session() as session:
        chunks = ChunkRepository(session)
        retriever = ChunkRetriever(chunks, get_embedding_provider(), settings.rag)
        arms, refused, controls = await RetrievalEvaluator(retriever, chunks).run(
            SBI_POLICY_CASES, SBI_POLICY, k=k
        )

    print("-" * 80)
    for arm in arms:
        print(f"recall@{k} {arm.name:<10} {arm.hits}/{arm.total} = {arm.recall:.0%}")
    print(f"controls refused  {refused}/{controls}")

    hybrid = next(a for a in arms if a.name == "hybrid")
    if hybrid.recall < MIN_HYBRID_RECALL:
        print(f"\nFAIL: hybrid recall {hybrid.recall:.0%} < {MIN_HYBRID_RECALL:.0%}")
        return 1
    if refused != controls:
        print("\nFAIL: a control question was not refused")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
