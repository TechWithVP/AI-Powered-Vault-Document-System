"""Snippet selection for citations."""

from __future__ import annotations

import pytest

from app.ai.rag.snippets import SnippetSelector

PROSE = (
    "Introductory paragraph describing the scheme and who administers it. "
    "A second paragraph of general background about the arrangement. "
    "The reimbursement limit is 500 USD per calendar year. "
    "A closing paragraph about unrelated administrative detail."
)

TABLE = (
    "1.1 Full Name of the Insured Cement Industries\n"
    "1.2 Address New Delhi\n"
    "1.3 Business Type Cement\n"
    "1.4 Geographical Limits India (no location barrier)\n"
    "1.5 Policy Period One year from inception\n"
)


@pytest.fixture
def selector() -> SnippetSelector:
    return SnippetSelector(120)


class TestShortContent:
    def test_content_within_budget_is_returned_whole(
        self, selector: SnippetSelector
    ) -> None:
        assert selector.select("A short passage.", query="anything") == (
            "A short passage."
        )

    def test_whitespace_is_normalised(self, selector: SnippetSelector) -> None:
        assert selector.select("Spread   over\n\nlines.") == "Spread over lines."


class TestRelevanceCentring:
    def test_the_matching_sentence_is_quoted(
        self, selector: SnippetSelector
    ) -> None:
        snippet = selector.select(PROSE, query="What is the reimbursement limit?")
        assert "500 USD" in snippet

    def test_a_different_question_moves_the_window(
        self, selector: SnippetSelector
    ) -> None:
        snippet = selector.select(PROSE, query="who administers the scheme?")
        assert "administers" in snippet
        assert "500 USD" not in snippet

    def test_line_separated_tables_are_searchable(
        self, selector: SnippetSelector
    ) -> None:
        """Table rows are divided by newlines, not full stops."""
        snippet = selector.select(TABLE, query="What are the Geographical Limits?")
        assert "India (no location barrier)" in snippet

    def test_without_a_query_the_opening_is_quoted(
        self, selector: SnippetSelector
    ) -> None:
        assert selector.select(PROSE).startswith("Introductory")

    def test_a_query_of_only_common_words_falls_back_to_the_opening(
        self, selector: SnippetSelector
    ) -> None:
        assert selector.select(PROSE, query="what is it").startswith("Introductory")


class TestTruncationMarkers:
    def test_a_window_taken_from_the_middle_is_marked_at_both_ends(
        self, selector: SnippetSelector
    ) -> None:
        snippet = selector.select(PROSE, query="reimbursement limit")
        assert snippet.startswith("...")

    def test_a_window_from_the_start_is_not_marked_at_the_start(
        self, selector: SnippetSelector
    ) -> None:
        assert not selector.select(PROSE, query="introductory scheme").startswith("...")

    def test_an_unsplittable_run_is_still_marked_as_cut(self) -> None:
        """Text without punctuation must not look complete after being cut."""
        snippet = SnippetSelector(60).select("word " * 200)
        assert snippet.endswith("...")
        assert len(snippet) <= 70


class TestBudget:
    @pytest.mark.parametrize("budget", [60, 120, 300])
    def test_the_character_budget_is_respected(self, budget: int) -> None:
        snippet = SnippetSelector(budget).select(PROSE, query="reimbursement")
        # Allowing for the ellipsis markers added at each cut end.
        assert len(snippet) <= budget + 6
