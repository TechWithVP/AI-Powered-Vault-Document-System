"""Text extraction package.

The registry maps a file type to the extractor that handles it, so adding a new
format means adding one class and one registry entry - no caller changes.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.constants import FileType
from app.core.exceptions import UnsupportedFileTypeError
from app.core.logging import get_logger
from app.processing.extractors.base import (
    ExtractedDocument,
    ExtractedPage,
    TextExtractor,
)
from app.processing.extractors.docx import DocxExtractor
from app.processing.extractors.pdf import PdfExtractor
from app.processing.extractors.plain import PlainTextExtractor
from app.processing.normalization import TextNormalizer

logger = get_logger(__name__)

__all__ = [
    "DocxExtractor",
    "TextNormalizer",
    "ExtractedDocument",
    "ExtractedPage",
    "ExtractorRegistry",
    "PdfExtractor",
    "PlainTextExtractor",
    "TextExtractor",
    "get_extractor_registry",
]


class ExtractorRegistry:
    """Resolves the extractor responsible for a given file type."""

    def __init__(
        self,
        extractors: list[TextExtractor] | None = None,
        normalizer: TextNormalizer | None = None,
    ) -> None:
        self._extractors = extractors or [
            PdfExtractor(),
            DocxExtractor(),
            PlainTextExtractor(),
        ]
        self._normalizer = normalizer or TextNormalizer()

    def resolve(self, file_type: FileType) -> TextExtractor:
        for extractor in self._extractors:
            if extractor.supports(file_type):
                return extractor
        raise UnsupportedFileTypeError.for_filename(
            f"file.{file_type.value}", FileType.supported_extensions()
        )

    def extract(
        self, data: bytes, *, file_type: FileType, filename: str
    ) -> ExtractedDocument:
        """Resolve the extractor, run it, then repair extraction artefacts.

        Normalising here rather than inside each extractor keeps parsers focused
        on their format and guarantees every format gets the same treatment.
        """
        extracted = self.resolve(file_type).extract(data, filename=filename)
        return self._normalise(extracted)

    def _normalise(self, document: ExtractedDocument) -> ExtractedDocument:
        """Repair extraction artefacts, preserving page numbers and metadata."""
        pages: list[ExtractedPage] = []
        collapsed = 0

        for page in document.pages:
            cleaned, report = self._normalizer.normalize_text(page.text)
            collapsed += report.tokens_collapsed
            pages.append(ExtractedPage(number=page.number, text=cleaned))

        if collapsed:
            logger.info(
                "Repaired extraction artefacts",
                extra={"tokens_collapsed": collapsed, "pages": len(pages)},
            )

        return ExtractedDocument(pages=pages, metadata=document.metadata)

    @property
    def supported_types(self) -> frozenset[FileType]:
        return frozenset().union(
            *(extractor.supported_types for extractor in self._extractors)
        )


@lru_cache(maxsize=1)
def get_extractor_registry() -> ExtractorRegistry:
    """Return the process-wide extractor registry."""
    return ExtractorRegistry()
