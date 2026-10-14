"""Custom SQLAlchemy column types."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeVar

from sqlalchemy import Dialect, String
from sqlalchemy.types import TypeDecorator

EnumT = TypeVar("EnumT", bound=StrEnum)


class StrEnumType(TypeDecorator[EnumT]):
    """Persists a StrEnum as VARCHAR and loads it back as the enum.

    A native database ENUM is deliberately avoided: adding a value to a
    Postgres enum requires a migration and locks the type, whereas VARCHAR
    keeps the vocabulary owned by the application. The decorator restores the
    Python type on load so callers always work with real enum members.
    """

    impl = String
    cache_ok = True

    def __init__(self, enum_class: type[EnumT], length: int = 32) -> None:
        self._enum_class = enum_class
        super().__init__(length=length)

    @property
    def python_type(self) -> type[EnumT]:
        return self._enum_class

    def process_bind_param(
        self, value: EnumT | str | None, dialect: Dialect
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, self._enum_class):
            return value.value
        # Validate loose strings so an invalid value fails here rather than
        # silently entering the database.
        return self._enum_class(value).value

    def process_result_value(
        self, value: Any, dialect: Dialect
    ) -> EnumT | None:
        if value is None:
            return None
        return self._enum_class(value)
