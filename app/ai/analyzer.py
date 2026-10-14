"""Document analysis.

Produces the structured insight record for a document. Documents that exceed
the single-pass budget are condensed section by section and then analysed from
those condensations, so a large document still yields a faithful summary rather
than one based only on its opening pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ai.base import LLMProvider, StructuredResult, TokenUsage
from app.ai.prompts import (
    build_analysis_messages,
    build_reduce_messages,
    build_section_messages,
)
from app.ai.schemas import DOCUMENT_ANALYSIS_SCHEMA, SECTION_SUMMARY_SCHEMA
from app.core.config import IngestionSettings
from app.core.constants import DocumentCategory, Sentiment
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Characters per section when a document must be condensed first. Sized well
#: inside the model context so a section plus its instructions always fit.
SECTION_CHARS = 24_000

#: Upper bound on sections, so a pathologically large upload cannot fan out
#: into an unbounded number of provider calls.
MAX_SECTIONS = 12


@dataclass(frozen=True, slots=True)
class DocumentAnalysis:
    """Validated analysis of one document."""

    summary: str
    key_points: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    entities: list[dict[str, str]] = field(default_factory=list)
    category: DocumentCategory = DocumentCategory.OTHER
    sentiment: Sentiment = Sentiment.NEUTRAL
    language: str = "en"
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    was_truncated: bool = False

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        model: str,
        usage: TokenUsage,
        was_truncated: bool,
    ) -> "DocumentAnalysis":
        """Build from provider output, normalising the free-text fields.

        The schema guarantees the shape; this guards the values - lowercasing
        tags, dropping blanks and falling back on an unrecognised enum - so one
        odd value never fails an otherwise good analysis.
        """
        return cls(
            summary=str(payload.get("summary", "")).strip(),
            key_points=_clean_list(payload.get("key_points")),
            tags=[tag.lower() for tag in _clean_list(payload.get("tags"))],
            entities=_clean_entities(payload.get("entities")),
            category=_coerce(
                DocumentCategory, payload.get("category"), DocumentCategory.OTHER
            ),
            sentiment=_coerce(Sentiment, payload.get("sentiment"), Sentiment.NEUTRAL),
            language=str(payload.get("language") or "en").lower()[:16],
            model=model,
            usage=usage,
            was_truncated=was_truncated,
        )


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _clean_entities(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    entities: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            entities.append(
                {"name": name, "type": str(item.get("type", "other")).strip()}
            )
    return entities


def _coerce(enum_class: type, value: Any, default: Any) -> Any:
    try:
        return enum_class(str(value).lower())
    except (ValueError, AttributeError):
        return default


class DocumentAnalyzer:
    """Turns document text into a structured insight record."""

    def __init__(self, llm: LLMProvider, settings: IngestionSettings) -> None:
        self._llm = llm
        self._settings = settings

    def analyze(self, text: str, *, filename: str) -> DocumentAnalysis:
        """Analyse a document, condensing it first when it is too large."""
        normalised = text.strip()
        if not normalised:
            raise ValueError("Cannot analyse an empty document.")

        if len(normalised) <= self._settings.max_analysis_chars:
            return self._analyze_directly(normalised, filename=filename)
        return self._analyze_by_sections(normalised, filename=filename)

    def _analyze_directly(self, text: str, *, filename: str) -> DocumentAnalysis:
        result = self._structured(
            build_analysis_messages(text, filename=filename),
            schema=DOCUMENT_ANALYSIS_SCHEMA,
            schema_name="document_analysis",
        )
        return DocumentAnalysis.from_payload(
            result.data,
            model=result.model,
            usage=result.usage,
            was_truncated=False,
        )

    def _analyze_by_sections(self, text: str, *, filename: str) -> DocumentAnalysis:
        sections = self._split_sections(text)
        logger.info(
            "Analysing large document by sections",
            extra={"filename": filename, "sections": len(sections)},
        )

        condensations: list[str] = []
        total_usage = TokenUsage()

        for position, section in enumerate(sections, start=1):
            result = self._structured(
                build_section_messages(
                    section, position=position, total=len(sections)
                ),
                schema=SECTION_SUMMARY_SCHEMA,
                schema_name="section_summary",
                max_tokens=1024,
            )
            total_usage = total_usage + result.usage
            condensations.append(self._render_section(result.data))

        final = self._structured(
            build_reduce_messages(condensations, filename=filename),
            schema=DOCUMENT_ANALYSIS_SCHEMA,
            schema_name="document_analysis",
        )
        total_usage = total_usage + final.usage

        return DocumentAnalysis.from_payload(
            final.data,
            model=final.model,
            usage=total_usage,
            # Truncated only if the section cap forced content to be dropped.
            was_truncated=len(sections) >= MAX_SECTIONS,
        )

    def _structured(
        self,
        messages: list,
        *,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
    ) -> StructuredResult:
        return self._llm.complete_structured(
            messages,
            schema=schema,
            schema_name=schema_name,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _render_section(payload: dict[str, Any]) -> str:
        summary = str(payload.get("summary", "")).strip()
        points = _clean_list(payload.get("key_points"))
        bullets = "\n".join(f"- {point}" for point in points)
        return f"{summary}\n{bullets}".strip()

    @staticmethod
    def _split_sections(text: str) -> list[str]:
        """Split on paragraph boundaries into analysis-sized sections."""
        sections: list[str] = []
        buffer: list[str] = []
        length = 0

        for paragraph in text.split("\n\n"):
            buffer.append(paragraph)
            length += len(paragraph) + 2
            if length >= SECTION_CHARS:
                sections.append("\n\n".join(buffer))
                buffer, length = [], 0
            if len(sections) >= MAX_SECTIONS:
                return sections

        if buffer and len(sections) < MAX_SECTIONS:
            sections.append("\n\n".join(buffer))
        return sections
