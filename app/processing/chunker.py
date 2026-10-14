"""Document chunking.

Chunks are produced per page so every chunk keeps the page number it came from,
which is what lets an answer cite a precise location. Sizing is done in tokens
rather than characters so chunks line up with the embedding model's budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import IngestionSettings
from app.core.logging import get_logger
from app.processing.extractors.base import ExtractedDocument

logger = get_logger(__name__)

#: Encoding used by the current OpenAI embedding and chat models.
TOKENIZER_ENCODING = "cl100k_base"

#: Separators ordered from strongest to weakest semantic boundary, so splits
#: land on paragraph breaks before falling back to sentences, then words.
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

#: Chunks shorter than this carry too little signal to embed usefully.
MIN_CHUNK_CHARS = 40


@lru_cache(maxsize=1)
def get_tokenizer() -> tiktoken.Encoding:
    """Return the cached tokenizer; loading it is expensive."""
    return tiktoken.get_encoding(TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    """Exact token count for the configured model family."""
    return len(get_tokenizer().encode(text, disallowed_special=()))


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A chunk of document text ready to be embedded."""

    index: int
    content: str
    page_number: int | None
    token_count: int

    @property
    def is_substantive(self) -> bool:
        return len(self.content.strip()) >= MIN_CHUNK_CHARS


class DocumentChunker:
    """Splits an extracted document into token-sized, page-attributed chunks."""

    def __init__(self, settings: IngestionSettings) -> None:
        self._settings = settings
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=TOKENIZER_ENCODING,
            chunk_size=settings.chunk_tokens,
            chunk_overlap=settings.chunk_overlap_tokens,
            separators=SEPARATORS,
        )

    def chunk(self, document: ExtractedDocument) -> list[TextChunk]:
        """Split every page, returning chunks numbered across the document."""
        candidates = self._split_pages(document)
        substantive = [piece for piece in candidates if self._is_substantive(piece)]

        # The minimum-size filter exists to drop noise such as page numbers and
        # running headers. If it would discard everything, the document is
        # simply short - keep what there is rather than failing the ingestion.
        selected = substantive or candidates
        chunks = [
            TextChunk(
                index=index,
                content=text,
                page_number=page_number,
                token_count=count_tokens(text),
            )
            for index, (text, page_number) in enumerate(selected)
        ]

        logger.info(
            "Document chunked",
            extra={
                "pages": document.page_count,
                "chunks": len(chunks),
                "chunk_tokens": self._settings.chunk_tokens,
                "fragments_dropped": len(candidates) - len(chunks),
            },
        )
        return chunks

    def _split_pages(
        self, document: ExtractedDocument
    ) -> list[tuple[str, int | None]]:
        """Split each page, keeping every piece paired with its page number."""
        pieces: list[tuple[str, int | None]] = []
        for page in document.pages:
            if page.is_empty:
                continue
            for raw in self._splitter.split_text(page.text):
                text = raw.strip()
                if text:
                    pieces.append((text, page.number))
        return pieces

    @staticmethod
    def _is_substantive(piece: tuple[str, int | None]) -> bool:
        return len(piece[0]) >= MIN_CHUNK_CHARS
