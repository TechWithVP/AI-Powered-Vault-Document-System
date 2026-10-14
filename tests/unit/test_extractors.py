"""Text extraction across formats and failure modes."""

from __future__ import annotations

import io
import zipfile

import pytest

from app.core.constants import FileType
from app.core.exceptions import (
    NoExtractableTextError,
    TextExtractionError,
    UnsupportedFileTypeError,
)
from app.processing.extractors import ExtractorRegistry
from app.processing.extractors.plain import PlainTextExtractor


@pytest.fixture
def registry() -> ExtractorRegistry:
    return ExtractorRegistry()


class TestRegistry:
    @pytest.mark.parametrize(
        ("file_type", "expected"),
        [
            (FileType.PDF, "PdfExtractor"),
            (FileType.DOCX, "DocxExtractor"),
            (FileType.TXT, "PlainTextExtractor"),
            (FileType.MD, "PlainTextExtractor"),
        ],
    )
    def test_resolves_the_right_extractor(
        self, registry: ExtractorRegistry, file_type: FileType, expected: str
    ) -> None:
        assert type(registry.resolve(file_type)).__name__ == expected

    def test_every_supported_type_is_covered(
        self, registry: ExtractorRegistry
    ) -> None:
        assert registry.supported_types == frozenset(FileType)


class TestPlainText:
    def test_extracts_utf8_text(self, registry: ExtractorRegistry) -> None:
        result = registry.extract(
            "Hello world. This is a policy document.".encode(),
            file_type=FileType.TXT,
            filename="a.txt",
        )
        assert "policy document" in result.text
        assert result.page_count == 1

    @pytest.mark.parametrize(
        ("encoding", "text"),
        [("cp1252", "Caf\u00e9 r\u00e9sum\u00e9"), ("utf-8", "\u4e2d\u6587\u6587\u4ef6")],
    )
    def test_handles_non_ascii_encodings(
        self, registry: ExtractorRegistry, encoding: str, text: str
    ) -> None:
        payload = (text + " " * 60).encode(encoding)
        result = registry.extract(
            payload, file_type=FileType.TXT, filename="a.txt"
        )
        assert result.text.strip()

    def test_empty_content_is_rejected(self, registry: ExtractorRegistry) -> None:
        with pytest.raises(NoExtractableTextError):
            registry.extract(b"   \n  ", file_type=FileType.TXT, filename="a.txt")

    def test_long_text_is_split_into_pages(self) -> None:
        paragraph = "This sentence provides realistic density. " * 30
        payload = "\n\n".join([paragraph] * 6).encode()
        result = PlainTextExtractor().extract(payload, filename="long.txt")
        assert result.page_count > 1

    def test_page_numbers_are_sequential_from_one(self) -> None:
        paragraph = "Content line with substance. " * 40
        payload = "\n\n".join([paragraph] * 8).encode()
        result = PlainTextExtractor().extract(payload, filename="long.txt")
        assert [page.number for page in result.pages] == list(
            range(1, result.page_count + 1)
        )

    def test_character_count_reflects_content(self) -> None:
        result = PlainTextExtractor().extract(b"abcdefghij" * 10, filename="a.txt")
        assert result.character_count == 100


class TestDocxFailures:
    def test_non_docx_bytes_are_rejected(self, registry: ExtractorRegistry) -> None:
        with pytest.raises(TextExtractionError, match="not a valid .docx"):
            registry.extract(
                b"this is plain text, not a zip",
                file_type=FileType.DOCX,
                filename="fake.docx",
            )

    def test_zip_without_word_parts_is_rejected(
        self, registry: ExtractorRegistry
    ) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("hello.txt", "not a word document")

        with pytest.raises(TextExtractionError):
            registry.extract(
                buffer.getvalue(), file_type=FileType.DOCX, filename="fake.docx"
            )


class TestPdfFailures:
    def test_corrupt_pdf_is_rejected(self, registry: ExtractorRegistry) -> None:
        with pytest.raises(TextExtractionError):
            registry.extract(
                b"%PDF-1.4 truncated garbage",
                file_type=FileType.PDF,
                filename="broken.pdf",
            )


class TestFileTypeDetection:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("report.pdf", FileType.PDF),
            ("REPORT.PDF", FileType.PDF),
            ("notes.MD", FileType.MD),
            ("a.b.c.txt", FileType.TXT),
        ],
    )
    def test_detects_type_from_extension(
        self, filename: str, expected: FileType
    ) -> None:
        assert FileType.from_filename(filename) is expected

    @pytest.mark.parametrize("filename", ["virus.exe", "noextension", "archive.zip"])
    def test_unknown_extensions_return_none(self, filename: str) -> None:
        assert FileType.from_filename(filename) is None

    def test_unsupported_type_raises_from_registry(self) -> None:
        registry = ExtractorRegistry(extractors=[PlainTextExtractor()])
        with pytest.raises(UnsupportedFileTypeError):
            registry.resolve(FileType.PDF)
