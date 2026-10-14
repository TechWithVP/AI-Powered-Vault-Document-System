"""Snippet selection for citations.

A citation is only useful if the quoted text actually contains what the answer
claims. Quoting the opening of a chunk fails that test whenever the relevant
passage sits further in, which is the common case for long chunks. The selector
instead finds the sentence that best matches the question and quotes around it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Sentence boundaries. Deliberately simple: the goal is a readable quote, not
#: linguistically perfect segmentation.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

#: Words this short carry no signal when matching a question to a sentence.
MIN_MATCH_TERM = 3

#: Very common words would match everything, so they are ignored when scoring.
STOP_WORDS = frozenset(
    """
    the and for are but not you all any can her was one our out his has had
    how what when where which who whom why that this these those with from
    into than then they them there their will would should could does did
    done have been were is of to in on at it as by or if be a an i
    """.split()
)


@dataclass(frozen=True, slots=True)
class Snippet:
    """A quoted passage, flagged where it was cut."""

    text: str
    truncated_start: bool = False
    truncated_end: bool = False

    def render(self) -> str:
        prefix = "..." if self.truncated_start else ""
        suffix = "..." if self.truncated_end else ""
        return f"{prefix}{self.text}{suffix}"


class SnippetSelector:
    """Chooses the most relevant window of a chunk to quote."""

    def __init__(self, max_chars: int = 280) -> None:
        self._max_chars = max_chars

    def select(self, content: str, *, query: str = "") -> str:
        """Return the passage of ``content`` that best answers ``query``."""
        flat = " ".join(content.split())
        if len(flat) <= self._max_chars:
            return flat

        # Split before collapsing whitespace: tabular extracts are separated by
        # line breaks rather than full stops, so flattening first would leave
        # the whole chunk as a single unsplittable run.
        units = self._split(content)
        if not units:
            return Snippet(flat[: self._max_chars].rstrip(), truncated_end=True).render()

        focus = self._best_sentence(units, query)
        return self._window(units, focus).render()

    # -- internals ----------------------------------------------------------- #
    @staticmethod
    def _split(text: str) -> list[str]:
        """Split into sentences, treating line breaks as boundaries too."""
        pieces = (" ".join(piece.split()) for piece in SENTENCE_BOUNDARY.split(text))
        return [piece for piece in pieces if piece]

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {
            word for word in words if len(word) >= MIN_MATCH_TERM and word not in STOP_WORDS
        }

    def _best_sentence(self, sentences: list[str], query: str) -> int:
        """Index of the sentence sharing the most distinctive terms with the query.

        Falls back to the first sentence when the question shares nothing, which
        keeps behaviour sensible for queries made entirely of common words.
        """
        wanted = self._terms(query)
        if not wanted:
            return 0

        best_index, best_score = 0, 0
        for index, sentence in enumerate(sentences):
            score = len(wanted & self._terms(sentence))
            if score > best_score:
                best_index, best_score = index, score
        return best_index

    def _window(self, sentences: list[str], focus: int) -> Snippet:
        """Grow outwards from the focus sentence until the budget is used."""
        selected = [focus]
        length = len(sentences[focus])
        before, after = focus - 1, focus + 1

        while length < self._max_chars and (before >= 0 or after < len(sentences)):
            # Prefer following context: it usually completes the thought.
            if after < len(sentences) and length + len(sentences[after]) + 1 <= self._max_chars:
                length += len(sentences[after]) + 1
                selected.append(after)
                after += 1
            elif before >= 0 and length + len(sentences[before]) + 1 <= self._max_chars:
                length += len(sentences[before]) + 1
                selected.insert(0, before)
                before -= 1
            else:
                break

        start, end = min(selected), max(selected)
        text = " ".join(sentences[start : end + 1])

        # A single unit longer than the budget - common for tables and for text
        # without sentence punctuation - still has to be cut, and saying so
        # matters: an unmarked cut reads as though the quote were complete.
        hard_cut = len(text) > self._max_chars
        if hard_cut:
            text = text[: self._max_chars].rstrip()

        return Snippet(
            text=text,
            truncated_start=start > 0,
            truncated_end=hard_cut or end < len(sentences) - 1,
        )
