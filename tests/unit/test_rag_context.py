"""Context assembly and citation mapping."""

from __future__ import annotations

import uuid

import pytest

from app.ai.rag.context import CitationMapper, ContextBuilder
from app.models.chunk import DocumentChunk
from app.repositories.chunk import ScoredChunk

DOC_A = uuid.uuid4()
DOC_B = uuid.uuid4()
NAMES = {DOC_A: "policy.pdf", DOC_B: "handbook.docx"}


def scored(
    text: str,
    *,
    page: int | None = 1,
    doc: uuid.UUID = DOC_A,
    similarity: float = 0.8,
    lexical_rank: float = 0.0,
) -> ScoredChunk:
    chunk = DocumentChunk(
        document_id=doc, chunk_index=0, content=text, page_number=page, token_count=10
    )
    return ScoredChunk(
        chunk=chunk, similarity=similarity, lexical_rank=lexical_rank
    )


@pytest.fixture
def builder() -> ContextBuilder:
    return ContextBuilder(max_chars=1000)


class TestContextBuilder:
    def test_excerpts_are_numbered_from_one(self, builder: ContextBuilder) -> None:
        context = builder.build([scored("First."), scored("Second.")], NAMES)
        assert "[1]" in context.block
        assert "[2]" in context.block
        assert [c.marker for c in context.citations] == [1, 2]

    def test_headers_name_the_document_and_page(
        self, builder: ContextBuilder
    ) -> None:
        context = builder.build([scored("Text.", page=7)], NAMES)
        assert "policy.pdf, page 7" in context.block

    def test_page_is_omitted_when_unknown(self, builder: ContextBuilder) -> None:
        context = builder.build([scored("Text.", page=None)], NAMES)
        assert "policy.pdf" in context.block
        assert "page" not in context.block

    def test_chunk_text_is_included_verbatim(
        self, builder: ContextBuilder
    ) -> None:
        context = builder.build([scored("The limit is 500 USD.")], NAMES)
        assert "The limit is 500 USD." in context.block

    def test_citations_carry_document_identity(
        self, builder: ContextBuilder
    ) -> None:
        context = builder.build([scored("Text.", doc=DOC_B)], NAMES)
        citation = context.citations[0]
        assert citation.document_id == DOC_B
        assert citation.document_name == "handbook.docx"

    def test_unknown_documents_get_a_readable_placeholder(
        self, builder: ContextBuilder
    ) -> None:
        context = builder.build([scored("Text.", doc=uuid.uuid4())], {})
        assert context.citations[0].document_name == "Unknown document"

    def test_long_chunks_are_quoted_in_part_and_marked_as_cut(self) -> None:
        builder = ContextBuilder(max_chars=10_000, snippet_chars=60)
        context = builder.build([scored("word " * 200)], NAMES)
        snippet = context.citations[0].snippet

        assert len(snippet) < len("word " * 200)
        assert "..." in snippet

    def test_snippets_centre_on_the_passage_matching_the_question(self) -> None:
        """A citation is only useful if the quote contains the answer."""
        chunk = (
            "Introductory paragraph about the scheme and its purpose. "
            "A second paragraph of general background information. "
            "The reimbursement limit is 500 USD per calendar year. "
            "A closing paragraph about unrelated administrative detail."
        )
        builder = ContextBuilder(max_chars=10_000, snippet_chars=90)

        context = builder.build(
            [scored(chunk)], NAMES, query="What is the reimbursement limit?"
        )

        assert "500 USD" in context.citations[0].snippet

    def test_whitespace_in_snippets_is_normalised(
        self, builder: ContextBuilder
    ) -> None:
        context = builder.build([scored("line one\n\n   line two")], NAMES)
        assert context.citations[0].snippet == "line one line two"

    def test_budget_stops_further_excerpts(self) -> None:
        builder = ContextBuilder(max_chars=120)
        results = [scored("x" * 100), scored("y" * 100), scored("z" * 100)]
        context = builder.build(results, NAMES)
        assert len(context.citations) < len(results)

    def test_at_least_one_excerpt_survives_a_tiny_budget(self) -> None:
        builder = ContextBuilder(max_chars=10)
        context = builder.build([scored("x" * 500)], NAMES)
        assert len(context.citations) == 1

    def test_no_results_yields_an_empty_context(
        self, builder: ContextBuilder
    ) -> None:
        context = builder.build([], NAMES)
        assert context.is_empty
        assert context.block == ""

    def test_a_lexical_only_hit_is_not_reported_as_irrelevant(
        self, builder: ContextBuilder
    ) -> None:
        """similarity is zero for exact-term hits, so rank must be exposed."""
        context = builder.build(
            [scored("a", similarity=0.0, lexical_rank=0.42)], NAMES
        )

        citation = context.citations[0]
        assert citation.similarity == pytest.approx(0.0)
        assert citation.lexical_rank == pytest.approx(0.42)
        assert citation.as_dict()["lexical_rank"] == pytest.approx(0.42)


class TestCitationMapper:
    def test_markers_are_extracted_from_the_answer(self) -> None:
        assert CitationMapper.used_markers("Claim [1] and also [3].") == {1, 3}

    def test_adjacent_markers_are_both_found(self) -> None:
        assert CitationMapper.used_markers("Both apply [2][4].") == {2, 4}

    def test_answers_without_markers_cite_nothing(self) -> None:
        assert CitationMapper.used_markers("No citation here.") == set()

    def test_only_referenced_citations_are_returned(
        self, builder: ContextBuilder
    ) -> None:
        context = builder.build([scored("a"), scored("b"), scored("c")], NAMES)
        selected = CitationMapper.select("Only the third [3].", context.citations)
        assert [citation.marker for citation in selected] == [3]

    def test_selection_preserves_marker_order(
        self, builder: ContextBuilder
    ) -> None:
        context = builder.build([scored("a"), scored("b"), scored("c")], NAMES)
        selected = CitationMapper.select("Cites [3] then [1].", context.citations)
        assert [citation.marker for citation in selected] == [1, 3]

    def test_markers_beyond_the_supplied_excerpts_are_ignored(
        self, builder: ContextBuilder
    ) -> None:
        context = builder.build([scored("a")], NAMES)
        assert CitationMapper.select("Invented source [9].", context.citations) == []

    def test_grounding_requires_a_valid_marker(
        self, builder: ContextBuilder
    ) -> None:
        context = builder.build([scored("a")], NAMES)
        assert CitationMapper.is_grounded("Backed by [1].", context.citations)
        assert not CitationMapper.is_grounded("Backed by [9].", context.citations)
        assert not CitationMapper.is_grounded("No marker.", context.citations)
