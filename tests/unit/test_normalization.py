"""Extraction artefact repair."""

from __future__ import annotations

import pytest

from app.processing.normalization import TextNormalizer


@pytest.fixture
def normalizer() -> TextNormalizer:
    return TextNormalizer()


class TestOverprintedTokens:
    @pytest.mark.parametrize(
        ("overprinted", "expected"),
        [
            ("SSSSBBBBIIII", "SBI"),
            ("GGGGeeeennnneeeerrrraaaallll", "General"),
            ("IIIInnnnssssuuuurrrraaaannnncccceeee", "Insurance"),
            ("CCCCoooommmmppppaaaannnnyyyy", "Company"),
            ("LLLLiiiimmmmiiiitttteeeedddd", "Limited"),
        ],
    )
    def test_uniform_repeats_are_collapsed(
        self, normalizer: TextNormalizer, overprinted: str, expected: str
    ) -> None:
        assert normalizer.collapse_overprinted(overprinted) == expected

    @pytest.mark.parametrize(
        "word",
        [
            "committee",
            "bookkeeper",
            "letter",
            "success",
            "addressees",
            "aggressive",
            "Hello",
            "aaa",
            "AAAA",
            "zz",
        ],
    )
    def test_ordinary_words_are_untouched(
        self, normalizer: TextNormalizer, word: str
    ) -> None:
        """Only a token whose every letter repeats identically is suspicious."""
        assert normalizer.collapse_overprinted(word) == word

    @pytest.mark.parametrize("token", ["9900112233", "1800", "2024", "11223344"])
    def test_numbers_are_never_collapsed(
        self, normalizer: TextNormalizer, token: str
    ) -> None:
        """A phone number can be uniformly doubled and still be correct."""
        assert normalizer.collapse_overprinted(token) == token

    def test_mixed_tokens_are_left_alone(self, normalizer: TextNormalizer) -> None:
        assert normalizer.collapse_overprinted("AAAA-1111") == "AAAA-1111"

    def test_a_doubled_word_containing_a_double_letter_is_left_alone(
        self, normalizer: TextNormalizer
    ) -> None:
        """A known limit of the rule, and the safe side of it.

        Doubling "Hello" yields "HHeelllloo", where the doubled "ll" becomes a
        run of four among runs of two. The repeat is no longer uniform, so the
        token is left untouched rather than guessed at - declining to repair is
        always preferable to corrupting real text.
        """
        assert normalizer.collapse_overprinted("HHeelllloo") == "HHeelllloo"


class TestDocumentText:
    def test_a_real_overprinted_header_is_repaired(
        self, normalizer: TextNormalizer
    ) -> None:
        header = (
            "SSSSBBBBIIII GGGGeeeennnneeeerrrraaaallll "
            "IIIInnnnssssuuuurrrraaaannnncccceeee"
        )
        cleaned, report = normalizer.normalize_text(header)

        assert cleaned == "SBI General Insurance"
        assert report.tokens_collapsed == 3
        assert report.changed

    def test_surrounding_text_is_preserved(
        self, normalizer: TextNormalizer
    ) -> None:
        cleaned, _ = normalizer.normalize_text(
            "SSSSBBBBIIII policy number 9900112233 issued to the committee."
        )
        assert cleaned == "SBI policy number 9900112233 issued to the committee."

    def test_line_structure_survives(self, normalizer: TextNormalizer) -> None:
        cleaned, _ = normalizer.normalize_text("First line\nSecond line")
        assert cleaned == "First line\nSecond line"

    def test_excess_blank_lines_are_collapsed(
        self, normalizer: TextNormalizer
    ) -> None:
        cleaned, _ = normalizer.normalize_text("One\n\n\n\n\nTwo")
        assert cleaned == "One\n\nTwo"

    def test_trailing_spaces_are_removed(self, normalizer: TextNormalizer) -> None:
        cleaned, _ = normalizer.normalize_text("Line with trailing   \nNext")
        assert cleaned == "Line with trailing\nNext"

    def test_clean_text_reports_no_change(self, normalizer: TextNormalizer) -> None:
        _, report = normalizer.normalize_text("An ordinary sentence of text.")
        assert not report.changed
