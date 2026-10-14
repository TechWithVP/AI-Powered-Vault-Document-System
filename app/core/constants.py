"""Domain enumerations shared by models, schemas and services."""

from __future__ import annotations

from enum import StrEnum


class DocumentStatus(StrEnum):
    """Lifecycle of an uploaded document."""

    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {DocumentStatus.READY, DocumentStatus.FAILED}

    @property
    def is_queryable(self) -> bool:
        """Only ``ready`` documents can take part in a chat session."""
        return self is DocumentStatus.READY


class ProcessingStage(StrEnum):
    """Checkpointed stages of the asynchronous ingestion pipeline."""

    EXTRACT = "extract"
    CHUNK = "chunk"
    EMBED = "embed"
    ANALYZE = "analyze"


class StageStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class AIOperation(StrEnum):
    """Billable AI operations, recorded for cost tracking."""

    EMBEDDING = "embedding"
    ANALYSIS = "analysis"
    CHAT = "chat"
    CONDENSE = "condense"
    SUGGEST = "suggest"
    TITLE = "title"


class DocumentCategory(StrEnum):
    """Controlled vocabulary the analyser must choose from."""

    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    RESEARCH = "research"
    MARKETING = "marketing"
    HR = "hr"
    CORRESPONDENCE = "correspondence"
    OTHER = "other"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class FileType(StrEnum):
    """Supported upload formats."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"

    @property
    def media_type(self) -> str:
        return _MEDIA_TYPES[self]

    @classmethod
    def from_filename(cls, filename: str) -> "FileType | None":
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        try:
            return cls(suffix)
        except ValueError:
            return None

    @classmethod
    def supported_extensions(cls) -> tuple[str, ...]:
        return tuple(f".{member.value}" for member in cls)


_MEDIA_TYPES: dict[FileType, str] = {
    FileType.PDF: "application/pdf",
    FileType.DOCX: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    FileType.TXT: "text/plain",
    FileType.MD: "text/markdown",
}
