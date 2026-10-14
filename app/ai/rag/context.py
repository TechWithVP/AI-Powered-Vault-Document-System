"""Context assembly and citation mapping.

Retrieved chunks are rendered as numbered excerpts, and the numbers the model
cites are mapped back to the exact document, page and snippet. This pairing is
what makes an answer verifiable rather than merely plausible.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, replace

from app.ai.rag.snippets import SnippetSelector
from app.models.chunk import DocumentChunk
from app.repositories.chunk import ScoredChunk

#: Matches citation markers such as [1] or [2][3] in model output.
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

#: Characters of a chunk quoted back in a citation.
SNIPPET_CHARS = 280


@dataclass(frozen=True, slots=True)
class Citation:
    """A source reference attached to an answer."""

    marker: int
    document_id: uuid.UUID
    document_name: str
    page_number: int | None
    snippet: str
    similarity: float
    #: Reported alongside similarity because a chunk found only by exact-term
    #: matching has a similarity of zero, which would otherwise advertise a
    #: perfectly good citation as irrelevant.
    lexical_rank: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "document_id": str(self.document_id),
            "document_name": self.document_name,
            "page_number": self.page_number,
            "snippet": self.snippet,
            "similarity": round(self.similarity, 4),
            "lexical_rank": round(self.lexical_rank, 4),
        }


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    """The numbered excerpts given to the model, with their citations."""

    block: str
    citations: list[Citation]

    @property
    def is_empty(self) -> bool:
        return not self.citations


class ContextBuilder:
    """Renders retrieved chunks into a numbered context block."""

    def __init__(self, *, max_chars: int, snippet_chars: int = SNIPPET_CHARS) -> None:
        self._max_chars = max_chars
        self._snippets = SnippetSelector(snippet_chars)

    def build(
        self,
        results: list[ScoredChunk],
        names: dict[uuid.UUID, str],
        *,
        query: str = "",
    ) -> RetrievedContext:
        """Assemble excerpts, stopping before the context budget is exceeded."""
        parts: list[str] = []
        citations: list[Citation] = []
        used = 0

        for marker, scored in enumerate(results, start=1):
            chunk = scored.chunk
            header = self._header(marker, chunk, names)
            excerpt = f"{header}\n{chunk.content}"

            if used + len(excerpt) > self._max_chars and citations:
                # Keep whole excerpts only: a truncated passage can change
                # meaning, and a citation must quote what the model saw.
                break

            parts.append(excerpt)
            used += len(excerpt)
            citations.append(
                Citation(
                    marker=marker,
                    document_id=chunk.document_id,
                    document_name=names.get(chunk.document_id, "Unknown document"),
                    page_number=chunk.page_number,
                    snippet=self._snippets.select(chunk.content, query=query),
                    similarity=scored.similarity,
                    lexical_rank=scored.lexical_rank,
                )
            )

        return RetrievedContext(block="\n\n".join(parts), citations=citations)

    @staticmethod
    def _header(
        marker: int, chunk: DocumentChunk, names: dict[uuid.UUID, str]
    ) -> str:
        name = names.get(chunk.document_id, "Unknown document")
        location = f", page {chunk.page_number}" if chunk.page_number else ""
        return f"[{marker}] {name}{location}"

class CitationMapper:
    """Resolves the markers an answer actually cites."""

    @staticmethod
    def used_markers(answer: str) -> set[int]:
        return {int(match) for match in CITATION_PATTERN.findall(answer)}

    @classmethod
    def select(cls, answer: str, citations: list[Citation]) -> list[Citation]:
        """Return only the citations the answer references, in marker order.

        Unused excerpts are dropped so the citation list reflects the evidence
        behind the answer rather than everything retrieval happened to return.
        """
        used = cls.used_markers(answer)
        if not used:
            return []
        return [citation for citation in citations if citation.marker in used]

    @classmethod
    def is_grounded(cls, answer: str, citations: list[Citation]) -> bool:
        """Whether the answer cites at least one supplied excerpt."""
        valid = {citation.marker for citation in citations}
        return bool(cls.used_markers(answer) & valid)

    @classmethod
    def finalise(
        cls, answer: str, citations: list[Citation]
    ) -> tuple[str, list[Citation]]:
        """Renumber the cited sources 1..N and rewrite the answer to match.

        The model cites retrieval positions, so an answer that uses excerpts 4,
        5 and 6 would otherwise display "[4]" above a list of three sources with
        no [1] at all. Markers referring to excerpts that were never supplied
        are dropped rather than left dangling.
        """
        selected = cls.select(answer, citations)
        if not selected:
            return cls._strip_markers(answer), []

        renumbered = [
            replace(citation, marker=position)
            for position, citation in enumerate(selected, start=1)
        ]
        mapping = {
            citation.marker: position
            for position, citation in enumerate(selected, start=1)
        }

        def rewrite(match: re.Match[str]) -> str:
            new = mapping.get(int(match.group(1)))
            return f"[{new}]" if new else ""

        return cls._tidy(CITATION_PATTERN.sub(rewrite, answer)), renumbered

    @classmethod
    def _strip_markers(cls, answer: str) -> str:
        """Remove markers from an answer that cites nothing valid."""
        return cls._tidy(CITATION_PATTERN.sub("", answer))

    @staticmethod
    def _tidy(answer: str) -> str:
        """Close the gaps left where a marker was removed."""
        answer = re.sub(" +([.,;:!?])", lambda match: match.group(1), answer)
        answer = re.sub("[ ]{2,}", " ", answer)
        return answer.strip()
