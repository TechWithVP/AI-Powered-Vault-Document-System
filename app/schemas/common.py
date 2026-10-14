"""Shared response schemas: error envelope and pagination."""

from __future__ import annotations

from typing import Annotated, Any, Generic, TypeVar

from fastapi import Query
from pydantic import BaseModel, ConfigDict, Field

ItemT = TypeVar("ItemT")


class ErrorDetail(BaseModel):
    """Machine-readable description of a failure."""

    code: str = Field(description="Stable, machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    details: dict[str, Any] | None = Field(
        default=None, description="Optional structured context."
    )


class ErrorResponse(BaseModel):
    """Envelope returned for every non-2xx response."""

    error: ErrorDetail

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "not_found",
                    "message": "Document 'a1b2' was not found.",
                    "details": {"resource": "Document", "id": "a1b2"},
                }
            }
        }
    )


class PageParams(BaseModel):
    """Offset pagination parameters shared by list endpoints."""

    limit: Annotated[int, Query(ge=1, le=100)] = 20
    offset: Annotated[int, Query(ge=0)] = 0


class Page(BaseModel, Generic[ItemT]):
    """A single page of results plus the totals needed to paginate."""

    items: list[ItemT]
    total: int = Field(description="Total records matching the query.")
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total

    @classmethod
    def build(
        cls, items: list[ItemT], *, total: int, params: PageParams
    ) -> "Page[ItemT]":
        return cls(
            items=items, total=total, limit=params.limit, offset=params.offset
        )
