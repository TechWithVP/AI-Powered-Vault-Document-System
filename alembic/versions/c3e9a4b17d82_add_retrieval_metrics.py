"""record each search's split between the vector and lexical arms

Hybrid retrieval is two systems whose relative value is empirical and changes
with the corpus. Fusion is the only point where both signals share a scale, so
the contribution of each arm is computed there and recorded per search.

Revision ID: c3e9a4b17d82
Revises: b2d8f1930c47
Create Date: 2026-10-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3e9a4b17d82"
down_revision: str | None = "b2d8f1930c47"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "retrieval_metrics",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "session_id",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "message_id",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("chat_messages.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("question_kind", sa.String(length=16), nullable=False),
        sa.Column("semantic_contribution", sa.Float(), nullable=False, server_default="0"),
        sa.Column("lexical_contribution", sa.Float(), nullable=False, server_default="0"),
        sa.Column("semantic_share", sa.Float(), nullable=False, server_default="0"),
        sa.Column("lexical_share", sa.Float(), nullable=False, server_default="0"),
        sa.Column("results_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("semantic_only", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("lexical_only", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("found_by_both", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("best_similarity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("best_lexical_rank", sa.Float(), nullable=False, server_default="0"),
        sa.Column("is_relevant", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reranked", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
    )
    op.create_index("ix_retrieval_created", "retrieval_metrics", ["created_at"])
    op.create_index("ix_retrieval_session", "retrieval_metrics", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_retrieval_session", table_name="retrieval_metrics")
    op.drop_index("ix_retrieval_created", table_name="retrieval_metrics")
    op.drop_table("retrieval_metrics")
