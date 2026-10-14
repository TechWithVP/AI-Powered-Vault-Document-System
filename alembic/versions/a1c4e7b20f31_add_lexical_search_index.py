"""add lexical search index to chunks

Adds a generated tsvector column and GIN index so retrieval can combine dense
vector similarity with exact-term matching. Dense embeddings dilute rare terms
inside heterogeneous chunks (policy schedules, tables), which caused answerable
questions to be refused.

Revision ID: a1c4e7b20f31
Revises: 0f5557063e1f
Create Date: 2026-10-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1c4e7b20f31"
down_revision: str | None = "0f5557063e1f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column(
            "search_vector",
            sa.dialects.postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_document_chunks_search_vector",
        "document_chunks",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_search_vector", table_name="document_chunks")
    op.drop_column("document_chunks", "search_vector")
