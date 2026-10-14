"""Exception handlers producing a single, uniform error envelope.

Every failure - domain error, framework validation error or unexpected crash -
is rendered as {"error": {"code", "message", "details"}} so clients only ever
parse one shape.
"""

from __future__ import annotations

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError, ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Fallback codes for framework-raised HTTP errors that carry no domain code.
_STATUS_CODES: dict[int, str] = {
    HTTPStatus.BAD_REQUEST: "bad_request",
    HTTPStatus.UNAUTHORIZED: "authentication_error",
    HTTPStatus.FORBIDDEN: "forbidden",
    HTTPStatus.NOT_FOUND: "not_found",
    HTTPStatus.METHOD_NOT_ALLOWED: "method_not_allowed",
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: "file_too_large",
    HTTPStatus.UNSUPPORTED_MEDIA_TYPE: "unsupported_media_type",
    HTTPStatus.UNPROCESSABLE_ENTITY: "validation_error",
    HTTPStatus.TOO_MANY_REQUESTS: "rate_limited",
}


def _envelope(code: str, message: str, details: dict | None = None) -> dict:
    error: dict = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"error": error}


async def handle_app_error(_: Request, exc: Exception) -> JSONResponse:
    """Render a domain error using its own code and status."""
    assert isinstance(exc, AppError)
    if exc.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        logger.error(
            "Domain error", extra={"code": exc.code, "message": exc.message}
        )
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


async def handle_validation_error(_: Request, exc: Exception) -> JSONResponse:
    """Flatten FastAPI request-validation errors into the standard envelope."""
    assert isinstance(exc, RequestValidationError)
    fields = [
        {
            "field": ".".join(str(part) for part in error["loc"][1:]) or "body",
            "issue": error["msg"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=ValidationError.status_code,
        content=_envelope(
            ValidationError.code,
            "The request payload is invalid.",
            {"fields": fields},
        ),
    )


async def handle_http_exception(_: Request, exc: Exception) -> JSONResponse:
    """Render Starlette HTTP errors (404s, 405s) in the standard envelope."""
    assert isinstance(exc, StarletteHTTPException)
    code = _STATUS_CODES.get(exc.status_code, "http_error")
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(code, detail),
        headers=getattr(exc, "headers", None),
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: log the traceback, return a generic message."""
    logger.exception(
        "Unhandled exception",
        extra={"path": request.url.path, "method": request.method},
    )
    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        content=_envelope(
            "internal_error",
            "An unexpected error occurred. The incident has been logged.",
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler to the application."""
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(Exception, handle_unexpected_error)
