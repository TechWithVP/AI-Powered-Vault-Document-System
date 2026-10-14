"""Domain exception contract."""

from __future__ import annotations

from http import HTTPStatus

import pytest

from app.core.exceptions import (
    AppError,
    AuthenticationError,
    DocumentNotReadyError,
    FileTooLargeError,
    NotFoundError,
    UnsupportedFileTypeError,
    ValidationError,
)


class TestErrorEnvelope:
    def test_payload_contains_code_and_message(self) -> None:
        payload = AppError("boom").to_payload()
        assert payload == {"error": {"code": "internal_error", "message": "boom"}}

    def test_details_are_included_when_present(self) -> None:
        payload = NotFoundError("missing", details={"id": "42"}).to_payload()
        assert payload["error"]["details"] == {"id": "42"}

    def test_details_are_omitted_when_empty(self) -> None:
        assert "details" not in AppError("boom").to_payload()["error"]


class TestStatusCodes:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (NotFoundError(), HTTPStatus.NOT_FOUND),
            (AuthenticationError(), HTTPStatus.UNAUTHORIZED),
            (ValidationError(), HTTPStatus.UNPROCESSABLE_ENTITY),
            (FileTooLargeError(), HTTPStatus.REQUEST_ENTITY_TOO_LARGE),
            (DocumentNotReadyError(), HTTPStatus.CONFLICT),
        ],
    )
    def test_status_code_matches_semantics(
        self, error: AppError, expected: HTTPStatus
    ) -> None:
        assert error.status_code == expected


class TestFactories:
    def test_not_found_names_the_resource(self) -> None:
        error = NotFoundError.for_resource("Document", "abc")
        assert "Document" in error.message
        assert error.details == {"resource": "Document", "id": "abc"}

    def test_unsupported_type_lists_supported_extensions(self) -> None:
        error = UnsupportedFileTypeError.for_filename("virus.exe", (".pdf", ".txt"))
        assert ".pdf" in error.message
        assert error.details["supported_extensions"] == [".pdf", ".txt"]

    def test_file_too_large_reports_both_sizes(self) -> None:
        error = FileTooLargeError.for_size(30 * 1024 * 1024, 25 * 1024 * 1024)
        assert error.details["limit_bytes"] == 25 * 1024 * 1024
        assert "25 MB" in error.message
