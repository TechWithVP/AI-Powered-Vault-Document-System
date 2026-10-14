"""Structured logging.

Emits single-line JSON in deployed environments (CloudWatch/Docker friendly)
and a readable console format locally. A context-local correlation id is
attached to every record so one request's logs can be traced end to end across
the API and the Celery worker.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

#: Key under which caller-supplied ``extra=`` fields are stored on the record.
#: Nesting them avoids collisions with reserved ``LogRecord`` attributes such as
#: ``filename``, ``module`` or ``args``, which otherwise raise a KeyError.
CONTEXT_KEY = "context"


class ContextLogger(logging.Logger):
    """Logger that stores ``extra=`` fields in a nested context dictionary."""

    def makeRecord(  # noqa: PLR0913 - signature fixed by the stdlib
        self,
        name: str,
        level: int,
        fn: str,
        lno: int,
        msg: object,
        args: object,
        exc_info: object,
        func: str | None = None,
        extra: dict[str, Any] | None = None,
        sinfo: str | None = None,
    ) -> logging.LogRecord:
        record = super().makeRecord(
            name, level, fn, lno, msg, args, exc_info, func, None, sinfo
        )
        setattr(record, CONTEXT_KEY, dict(extra) if extra else {})
        return record


logging.setLoggerClass(ContextLogger)


class CorrelationId:
    """Accessor and scope manager for the context-local correlation id."""

    @staticmethod
    def get() -> str | None:
        return _correlation_id.get()

    @staticmethod
    def set(value: str | None = None) -> Token[str | None]:
        return _correlation_id.set(value or uuid4().hex)

    @staticmethod
    def reset(token: Token[str | None]) -> None:
        _correlation_id.reset(token)


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if correlation_id := _correlation_id.get():
            payload["correlation_id"] = correlation_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        payload.update(_extras(record))
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable format for local development."""

    _TEMPLATE = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    def __init__(self) -> None:
        super().__init__(self._TEMPLATE, datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        suffix: list[str] = []
        if correlation_id := _correlation_id.get():
            suffix.append(f"cid={correlation_id[:8]}")
        suffix.extend(f"{key}={value}" for key, value in _extras(record).items())
        return f"{base} [{' '.join(suffix)}]" if suffix else base


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    """Return caller-supplied ``extra=`` fields attached to the record."""
    return getattr(record, CONTEXT_KEY, None) or {}


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Install the root logging configuration. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else ConsoleFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Access logs are emitted by our own middleware with richer context.
    for noisy in ("uvicorn.access", "botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
