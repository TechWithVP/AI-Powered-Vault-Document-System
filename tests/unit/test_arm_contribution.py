"""Attribution of a result set between the vector and exact-term arms."""

from __future__ import annotations

import uuid

import pytest

from app.ai.rag.retriever import (
    LEXICAL_WEIGHT,
    RRF_K,
    SEMANTIC_WEIGHT,
    ArmContribution,
    ChunkRetriever,
)
from app.models.chunk import DocumentChunk
from app.repositories.chunk import ScoredChunk

DOCUMENT_ID = uuid.uuid4()


def chunk(index: int) -> DocumentChunk:
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=DOCUMENT_ID,
        chunk_index=index,
        content=f"chunk {index}",
        token_count=5,
    )


def semantic(index: int, similarity: float = 0.5) -> ScoredChunk:
    return ScoredChunk(chunk=chunk(index), similarity=similarity)


def lexical(index: int, rank: float = 0.3) -> ScoredChunk:
    return ScoredChunk(chunk=chunk(index), lexical_rank=rank)


class TestScoredChunkAttribution:
    def test_shares_are_zero_before_fusion_runs(self) -> None:
        assert semantic(0).rrf_total == 0.0
        assert semantic(0).semantic_share == 0.0

    def test_a_chunk_found_by_both_splits_its_score(self) -> None:
        item = semantic(0).with_fusion(semantic=0.75, lexical=0.25)

        assert item.rrf_total == pytest.approx(1.0)
        assert item.semantic_share == pytest.approx(0.75)
        assert item.found_by == "both"

    def test_a_purely_lexical_hit_is_labelled_as_such(self) -> None:
        item = lexical(0).with_fusion(semantic=0.0, lexical=0.4)

        assert item.found_by == "lexical"
        assert item.semantic_share == pytest.approx(0.0)

    def test_merging_keeps_the_larger_contribution_from_each_arm(self) -> None:
        left = semantic(0).with_fusion(semantic=0.3, lexical=0.1)
        right = semantic(0).with_fusion(semantic=0.1, lexical=0.5)

        merged = left.merged_with(right)

        assert merged.rrf_semantic == pytest.approx(0.3)
        assert merged.rrf_lexical == pytest.approx(0.5)


class TestFusionRecordsAttribution:
    def test_each_arms_contribution_matches_the_rrf_formula(self) -> None:
        results = ChunkRetriever._fuse([semantic(0)], [], limit=5)

        assert results[0].rrf_semantic == pytest.approx(
            SEMANTIC_WEIGHT / (RRF_K + 1)
        )
        assert results[0].rrf_lexical == 0.0

    def test_a_chunk_in_both_lists_accumulates_both(self) -> None:
        shared = chunk(7)
        both = ChunkRetriever._fuse(
            [ScoredChunk(chunk=shared, similarity=0.4)],
            [ScoredChunk(chunk=shared, lexical_rank=0.2)],
            limit=5,
        )

        assert len(both) == 1
        assert both[0].rrf_semantic == pytest.approx(SEMANTIC_WEIGHT / (RRF_K + 1))
        assert both[0].rrf_lexical == pytest.approx(LEXICAL_WEIGHT / (RRF_K + 1))
        assert both[0].found_by == "both"

    def test_attribution_survives_the_ordering(self) -> None:
        results = ChunkRetriever._fuse(
            [semantic(0), semantic(1)], [lexical(2)], limit=3
        )

        assert all(item.rrf_total > 0 for item in results)


class TestArmContribution:
    def test_an_empty_result_set_has_no_share(self) -> None:
        split = ArmContribution.of([])

        assert split.semantic_share == 0.0
        assert split.lexical_share == 0.0
        assert split.total == 0.0

    def test_shares_sum_to_one(self) -> None:
        split = ArmContribution.of([
            semantic(0).with_fusion(semantic=0.6, lexical=0.0),
            semantic(1).with_fusion(semantic=0.2, lexical=0.2),
        ])

        assert split.semantic_share + split.lexical_share == pytest.approx(1.0)
        assert split.semantic_share == pytest.approx(0.8)

    def test_chunks_are_counted_by_which_arm_found_them(self) -> None:
        split = ArmContribution.of([
            semantic(0).with_fusion(semantic=0.5, lexical=0.0),
            semantic(1).with_fusion(semantic=0.5, lexical=0.0),
            lexical(2).with_fusion(semantic=0.0, lexical=0.4),
            semantic(3).with_fusion(semantic=0.3, lexical=0.3),
        ])

        assert split.semantic_only == 2
        assert split.lexical_only == 1
        assert split.both == 1

    def test_a_single_arm_result_set_reports_a_full_share(self) -> None:
        split = ArmContribution.of([
            semantic(0).with_fusion(semantic=0.5, lexical=0.0)
        ])

        assert split.semantic_share == pytest.approx(1.0)
        assert split.lexical_share == pytest.approx(0.0)
