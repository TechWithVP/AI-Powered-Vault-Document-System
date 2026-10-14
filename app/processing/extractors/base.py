"""Text extraction interface and value objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.core.constants import FileType
from app.core.exceptions import NoExtractableTextError


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """One page (or logical section) of a document.

    Page numbers are carried through the whole pipeline so an answer can cite
    the exact page a passage came from.
    """

    number: int
    text: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """The full text of a document, split into pages."""

    pages: list[ExtractedPage] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        """All page text joined, blank pages omitted."""
        return "\n\n".join(page.text for page in self.pages if not page.is_empty)

    @property
    def character_count(self) -> int:
        return sum(len(page.text) for page in self.pages)

    def require_text(self) -> "ExtractedDocument":
        """Return self, or raise when nothing readable was found."""
        if not self.text.strip():
            raise NoExtractableTextError()
        return self


class TextExtractor(ABC):
    """Converts raw file bytes into page-aware text."""

    #: File types this extractor can handle.
    supported_types: frozenset[FileType]

    @abstractmethod
    def extract(self, data: bytes, *, filename: str) -> ExtractedDocument:
        """Parse the bytes, raising a TextExtractionError subclass on failure."""

    def supports(self, file_type: FileType) -> bool:
        return file_type in self.supported_types
