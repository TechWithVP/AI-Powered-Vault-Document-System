"""Plain-text and Markdown extraction."""

from __future__ import annotations

from app.core.constants import FileType
from app.core.exceptions import TextExtractionError
from app.processing.extractors.base import (
    ExtractedDocument,
    ExtractedPage,
    TextExtractor,
)

#: Encodings tried in order: UTF-8 first, then common Windows/legacy fallbacks.
CANDIDATE_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

#: Plain files have no pages, so text is split into blocks of this size.
SYNTHETIC_PAGE_CHARS = 3000


class PlainTextExtractor(TextExtractor):
    """Decodes text and Markdown files, tolerating non-UTF-8 encodings."""

    supported_types = frozenset({FileType.TXT, FileType.MD})

    def extract(self, data: bytes, *, filename: str) -> ExtractedDocument:
        pages = self._paginate(self._decode(data))
        return ExtractedDocument(pages=pages).require_text()

    @staticmethod
    def _decode(data: bytes) -> str:
        for encoding in CANDIDATE_ENCODINGS:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise TextExtractionError(
            "The file could not be decoded as text. It may be binary or use an "
            "unsupported encoding."
        )

    @staticmethod
    def _paginate(text: str) -> list[ExtractedPage]:
        normalised = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalised:
            return []

        pages: list[ExtractedPage] = []
        buffer: list[str] = []
        length = 0

        for paragraph in normalised.split("\n\n"):
            buffer.append(paragraph)
            length += len(paragraph)
            if length >= SYNTHETIC_PAGE_CHARS:
                pages.append(
                    ExtractedPage(number=len(pages) + 1, text="\n\n".join(buffer))
                )
                buffer, length = [], 0

        if buffer:
            pages.append(ExtractedPage(number=len(pages) + 1, text="\n\n".join(buffer)))
        return pages
