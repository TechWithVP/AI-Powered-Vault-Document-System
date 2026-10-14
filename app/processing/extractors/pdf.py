"""PDF text extraction."""

from __future__ import annotations

import io

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError

from app.core.constants import FileType
from app.core.exceptions import EncryptedDocumentError, TextExtractionError
from app.core.logging import get_logger
from app.processing.extractors.base import (
    ExtractedDocument,
    ExtractedPage,
    TextExtractor,
)

logger = get_logger(__name__)


class PdfExtractor(TextExtractor):
    """Extracts page-aware text from PDFs using pdfplumber.

    pdfplumber preserves layout well and exposes real page boundaries, which the
    citation pipeline depends on. A page that fails to parse is skipped rather
    than failing the whole document: one malformed page should not cost the user
    the other ninety-nine.
    """

    supported_types = frozenset({FileType.PDF})

    def extract(self, data: bytes, *, filename: str) -> ExtractedDocument:
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages = self._read_pages(pdf, filename=filename)
                metadata = self._read_metadata(pdf)
        except PDFPasswordIncorrect as exc:
            raise EncryptedDocumentError() from exc
        except PDFSyntaxError as exc:
            raise TextExtractionError(
                "The file is not a readable PDF (corrupt or truncated)."
            ) from exc
        except Exception as exc:
            if self._looks_encrypted(exc):
                raise EncryptedDocumentError() from exc
            raise TextExtractionError(f"Could not read the PDF: {exc}") from exc

        return ExtractedDocument(pages=pages, metadata=metadata).require_text()

    def _read_pages(self, pdf: object, *, filename: str) -> list[ExtractedPage]:
        pages: list[ExtractedPage] = []
        for index, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning(
                    "Skipping unreadable PDF page",
                    extra={"filename": filename, "page": index, "error": str(exc)},
                )
                continue
            pages.append(ExtractedPage(number=index, text=text.strip()))
        return pages

    @staticmethod
    def _read_metadata(pdf: object) -> dict[str, str]:
        raw = pdf.metadata or {}
        wanted = ("Title", "Author", "Subject", "Creator", "Producer")
        return {
            key.lower(): str(raw[key])[:500]
            for key in wanted
            if raw.get(key) not in (None, "")
        }

    @staticmethod
    def _looks_encrypted(exc: Exception) -> bool:
        text = str(exc).lower()
        return "password" in text or "encrypt" in text
