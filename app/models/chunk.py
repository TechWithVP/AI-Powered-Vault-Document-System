"""Chunk ORM model holding the pgvector embedding used for retrieval."""

from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.document import Document

#: Embedding width is fixed by the configured model and must match the
#: dimension recorded in the Alembic migration.
EMBEDDING_DIMENSIONS: int = get_settings().openai.embedding_dimensions


class DocumentChunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A contiguous slice of a document, embedded for semantic search.

    ``page_number`` is preserved during extraction so answers can cite the exact
    page a passage came from.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),
        Index("ix_chunks_document", "document_id"),
        # Retrieval filters on the model, so it belongs in the scoping index.
        Index("ix_chunks_document_model", "document_id", "embedding_model"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS)
    )
    #: Which model produced the vector. Embeddings from different models are
    #: not comparable, but nothing about a stored vector reveals that: swap the
    #: model and cosine distance keeps returning confident, meaningless scores.
    #: Recording it lets retrieval skip vectors it cannot interpret, and lets
    #: a re-embed find the rows that need redoing.
    embedding_model: Mapped[str | None] = mapped_column(String(64))

    #: Lexical index maintained by PostgreSQL. Dense embeddings generalise well
    #: but dilute rare, specific terms inside heterogeneous text such as tables;
    #: full-text search covers exactly that case, so retrieval uses both.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=True,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    @property
    def is_embedded(self) -> bool:
        return self.embedding is not None

    def is_usable_with(self, model: str) -> bool:
        """Whether this vector can be compared against `model`'s output."""
        return self.embedding is not None and self.embedding_model == model

    def preview(self, limit: int = 240) -> str:
        """Short snippet used when rendering a citation."""
        text = " ".join(self.content.split())
        return text if len(text) <= limit else f"{text[:limit].rstrip()}..."
