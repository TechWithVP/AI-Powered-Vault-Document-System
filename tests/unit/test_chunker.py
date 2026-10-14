"""Chunking behaviour: sizing, overlap and page attribution."""

from __future__ import annotations

import pytest

from app.core.config import IngestionSettings
from app.processing.chunker import (
    MIN_CHUNK_CHARS,
    DocumentChunker,
    count_tokens,
)
from app.processing.extractors.base import ExtractedDocument, ExtractedPage


@pytest.fixture
def chunker() -> DocumentChunker:
    return DocumentChunker(
        IngestionSettings(chunk_tokens=100, chunk_overlap_ratio=0.2)
    )


def make_document(*pages: str) -> ExtractedDocument:
    return ExtractedDocument(
        pages=[
            ExtractedPage(number=index, text=text)
            for index, text in enumerate(pages, start=1)
        ]
    )


class TestTokenCounting:
    def test_counts_are_positive_and_scale_with_length(self) -> None:
        short = count_tokens("hello world")
        long = count_tokens("hello world " * 50)
        assert 0 < short < long

    def test_empty_text_counts_zero(self) -> None:
        assert count_tokens("") == 0


class TestChunking:
    def test_short_page_becomes_a_single_chunk(
        self, chunker: DocumentChunker
    ) -> None:
        chunks = chunker.chunk(make_document("A short policy statement here."))
        assert len(chunks) == 1
        assert chunks[0].page_number == 1

    def test_long_page_is_split_into_several_chunks(
        self, chunker: DocumentChunker
    ) -> None:
        chunks = chunker.chunk(make_document("This is a sentence. " * 200))
        assert len(chunks) > 1

    def test_chunk_indices_are_contiguous_across_pages(
        self, chunker: DocumentChunker
    ) -> None:
        chunks = chunker.chunk(
            make_document("First page content. " * 60, "Second page content. " * 60)
        )
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_page_numbers_are_preserved_for_citations(
        self, chunker: DocumentChunker
    ) -> None:
        chunks = chunker.chunk(
            make_document("Alpha content here. " * 60, "Beta content here. " * 60)
        )
        pages = {chunk.page_number for chunk in chunks}
        assert pages == {1, 2}

    def test_a_chunk_never_spans_two_pages(self, chunker: DocumentChunker) -> None:
        chunks = chunker.chunk(
            make_document("Alpha unique marker. " * 60, "Beta unique marker. " * 60)
        )
        for chunk in chunks:
            text = chunk.content
            assert not ("Alpha" in text and "Beta" in text)

    def test_blank_pages_are_skipped(self, chunker: DocumentChunker) -> None:
        chunks = chunker.chunk(make_document("Real content here. " * 20, "   "))
        assert all(chunk.page_number == 1 for chunk in chunks)

    def test_trivial_fragments_are_dropped_beside_real_content(
        self, chunker: DocumentChunker
    ) -> None:
        document = make_document("Substantive policy sentence here. " * 30, "7")
        chunks = chunker.chunk(document)
        assert all(len(chunk.content) >= MIN_CHUNK_CHARS for chunk in chunks)
        assert all(chunk.page_number == 1 for chunk in chunks)

    def test_a_very_short_document_still_yields_one_chunk(
        self, chunker: DocumentChunker
    ) -> None:
        """Filtering must never discard a short document entirely."""
        chunks = chunker.chunk(make_document("Rent is 500."))
        assert len(chunks) == 1
        assert chunks[0].content == "Rent is 500."

    def test_every_chunk_meets_the_minimum_size(
        self, chunker: DocumentChunker
    ) -> None:
        chunks = chunker.chunk(make_document("Meaningful sentence here. " * 100))
        assert all(len(c.content) >= MIN_CHUNK_CHARS for c in chunks)

    def test_token_counts_are_recorded(self, chunker: DocumentChunker) -> None:
        chunks = chunker.chunk(make_document("Policy content sentence. " * 50))
        assert all(chunk.token_count > 0 for chunk in chunks)

    def test_chunks_respect_the_configured_token_budget(self) -> None:
        chunker = DocumentChunker(
            IngestionSettings(chunk_tokens=60, chunk_overlap_ratio=0.1)
        )
        chunks = chunker.chunk(make_document("Sentence number one here. " * 120))
        # Allow headroom for the splitter landing on a separator boundary.
        assert all(chunk.token_count <= 90 for chunk in chunks)

    def test_empty_document_produces_no_chunks(
        self, chunker: DocumentChunker
    ) -> None:
        assert chunker.chunk(ExtractedDocument(pages=[])) == []


class TestOverlap:
    def test_consecutive_chunks_share_context(self) -> None:
        chunker = DocumentChunker(
            IngestionSettings(chunk_tokens=50, chunk_overlap_ratio=0.3)
        )
        text = " ".join(f"word{i}" for i in range(400))
        chunks = chunker.chunk(make_document(text))

        assert len(chunks) > 1
        overlapping = sum(
            1
            for first, second in zip(chunks, chunks[1:], strict=False)
            if set(first.content.split()) & set(second.content.split())
        )
        assert overlapping > 0
