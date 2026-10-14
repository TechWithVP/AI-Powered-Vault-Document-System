"""Domain exception hierarchy.

Every error raised by the application derives from :class:`AppError`, which
carries the machine-readable ``code``, the HTTP ``status_code`` and optional
structured ``details``. The API layer renders these uniformly, so clients always
receive the same error envelope regardless of which layer failed.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any


class AppError(Exception):
    """Base class for all application errors."""

    code: str = "internal_error"
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.__class__.message
        self.details = details or {}
        super().__init__(self.message)

    def to_payload(self) -> dict[str, Any]:
        """Render the error as the API's response envelope."""
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            error["details"] = self.details
        return {"error": error}


# --------------------------------------------------------------------------- #
# Client errors
# --------------------------------------------------------------------------- #
class ValidationError(AppError):
    code = "validation_error"
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    message = "The request payload is invalid."


class NotFoundError(AppError):
    code = "not_found"
    status_code = HTTPStatus.NOT_FOUND
    message = "The requested resource was not found."

    @classmethod
    def for_resource(cls, resource: str, identifier: Any) -> "NotFoundError":
        return cls(
            f"{resource} '{identifier}' was not found.",
            details={"resource": resource, "id": str(identifier)},
        )


class AuthenticationError(AppError):
    code = "authentication_error"
    status_code = HTTPStatus.UNAUTHORIZED
    message = "A valid API key is required."


class ConflictError(AppError):
    code = "conflict"
    status_code = HTTPStatus.CONFLICT
    message = "The request conflicts with the current state of the resource."


class RateLimitedError(AppError):
    code = "rate_limited"
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    message = "Too many requests. Please retry shortly."


# --------------------------------------------------------------------------- #
# Upload / document errors
# --------------------------------------------------------------------------- #
class UnsupportedFileTypeError(ValidationError):
    code = "unsupported_file_type"
    message = "This file type is not supported."

    @classmethod
    def for_filename(
        cls, filename: str, supported: tuple[str, ...]
    ) -> "UnsupportedFileTypeError":
        return cls(
            f"'{filename}' is not a supported document. "
            f"Supported types: {', '.join(supported)}.",
            details={"filename": filename, "supported_extensions": list(supported)},
        )


class FileTooLargeError(ValidationError):
    code = "file_too_large"
    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    message = "The uploaded file exceeds the maximum allowed size."

    @classmethod
    def for_size(cls, size_bytes: int, limit_bytes: int) -> "FileTooLargeError":
        to_mb = 1024 * 1024
        return cls(
            f"File is {size_bytes / to_mb:.1f} MB; the limit is "
            f"{limit_bytes / to_mb:.0f} MB.",
            details={"size_bytes": size_bytes, "limit_bytes": limit_bytes},
        )


class EmptyFileError(ValidationError):
    code = "empty_file"
    message = "The uploaded file is empty."


class DocumentNotReadyError(ConflictError):
    code = "document_not_ready"
    message = "The document is still being processed and cannot be queried yet."

    @classmethod
    def for_documents(cls, identifiers: list[str]) -> "DocumentNotReadyError":
        return cls(
            "These documents are not ready yet: " + ", ".join(identifiers),
            details={"document_ids": identifiers},
        )


# --------------------------------------------------------------------------- #
# Pipeline errors
# --------------------------------------------------------------------------- #
class ProcessingError(AppError):
    """Base class for failures inside the ingestion pipeline."""

    code = "processing_error"
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    message = "Document processing failed."


class TextExtractionError(ProcessingError):
    code = "text_extraction_failed"
    message = "The document's text could not be extracted."


class EncryptedDocumentError(TextExtractionError):
    code = "document_encrypted"
    message = "The document is password-protected and cannot be read."


class NoExtractableTextError(TextExtractionError):
    code = "no_extractable_text"
    message = (
        "No readable text was found. Scanned images are not supported without OCR."
    )


# --------------------------------------------------------------------------- #
# Infrastructure errors
# --------------------------------------------------------------------------- #
class StorageError(AppError):
    code = "storage_error"
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    message = "The document store is unavailable."


class LLMError(AppError):
    code = "llm_error"
    status_code = HTTPStatus.BAD_GATEWAY
    message = "The AI provider request failed."


class LLMTimeoutError(LLMError):
    code = "llm_timeout"
    status_code = HTTPStatus.GATEWAY_TIMEOUT
    message = "The AI provider timed out."


class LLMRateLimitError(LLMError):
    code = "llm_rate_limited"
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    message = "The AI provider rate limit was reached. Please retry shortly."
