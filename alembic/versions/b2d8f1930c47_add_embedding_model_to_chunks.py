"""record which model produced each embedding

Vectors from different embedding models are not comparable, but a stored
vector carries no evidence of its origin: change OPENAI_EMBEDDING_MODEL and
cosine distance keeps returning confident, meaningless scores against the old
rows. Recording the model lets retrieval skip vectors it cannot interpret and
lets a re-embed find exactly the rows that need redoing.

Existing rows are backfilled with the currently configured model, which is the
model that in fact produced them.

Revision ID: b2d8f1930c47
Revises: a1c4e7b20f31
Create Date: 2026-10-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.core.config import get_settings

revision: str = "b2d8f1930c47"
down_revision: str | None = "a1c4e7b20f31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("embedding_model", sa.String(length=64), nullable=True),
    )
    op.execute(
        sa.text(
            "UPDATE document_chunks SET embedding_model = :model "
            "WHERE embedding IS NOT NULL AND embedding_model IS NULL"
        ).bindparams(model=get_settings().openai.embedding_model)
    )
    op.create_index(
        "ix_chunks_document_model",
        "document_chunks",
        ["document_id", "embedding_model"],
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_document_model", table_name="document_chunks")
    op.drop_column("document_chunks", "embedding_model")
