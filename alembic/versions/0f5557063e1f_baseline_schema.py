"""baseline schema

Revision ID: 0f5557063e1f
Revises: 
Create Date: 2026-10-13 17:32:06.912112
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0f5557063e1f'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pgvector must exist before any VECTOR column is created.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table('chat_sessions',
    sa.Column('user_id', sa.String(length=128), nullable=False),
    sa.Column('title', sa.String(length=256), nullable=True),
    sa.Column('message_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_chat_sessions'))
    )
    op.create_index(op.f('ix_chat_sessions_created_at'), 'chat_sessions', ['created_at'], unique=False)
    op.create_index(op.f('ix_chat_sessions_user_id'), 'chat_sessions', ['user_id'], unique=False)
    op.create_index('ix_sessions_user_updated', 'chat_sessions', ['user_id', 'updated_at'], unique=False)
    op.create_table('documents',
    sa.Column('user_id', sa.String(length=128), nullable=False),
    sa.Column('filename', sa.String(length=512), nullable=False),
    sa.Column('file_type', sa.String(length=16), nullable=False),
    sa.Column('file_size_bytes', sa.BigInteger(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=True),
    sa.Column('storage_key', sa.String(length=1024), nullable=False),
    sa.Column('status', sa.String(length=16), server_default='queued', nullable=False),
    sa.Column('failure_reason', sa.Text(), nullable=True),
    sa.Column('failure_code', sa.String(length=64), nullable=True),
    sa.Column('page_count', sa.Integer(), nullable=True),
    sa.Column('chunk_count', sa.Integer(), nullable=True),
    sa.Column('character_count', sa.Integer(), nullable=True),
    sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_documents'))
    )
    op.create_index(op.f('ix_documents_content_hash'), 'documents', ['content_hash'], unique=False)
    op.create_index(op.f('ix_documents_created_at'), 'documents', ['created_at'], unique=False)
    op.create_index(op.f('ix_documents_status'), 'documents', ['status'], unique=False)
    op.create_index('ix_documents_user_created', 'documents', ['user_id', 'created_at'], unique=False)
    op.create_index(op.f('ix_documents_user_id'), 'documents', ['user_id'], unique=False)
    op.create_index('ix_documents_user_status', 'documents', ['user_id', 'status'], unique=False)
    op.create_table('api_usage',
    sa.Column('operation', sa.String(length=16), nullable=False),
    sa.Column('model', sa.String(length=64), nullable=False),
    sa.Column('endpoint', sa.String(length=128), nullable=True),
    sa.Column('document_id', sa.UUID(), nullable=True),
    sa.Column('session_id', sa.UUID(), nullable=True),
    sa.Column('prompt_tokens', sa.Integer(), nullable=False),
    sa.Column('completion_tokens', sa.Integer(), nullable=False),
    sa.Column('total_tokens', sa.Integer(), nullable=False),
    sa.Column('cost_usd', sa.Numeric(precision=12, scale=6), nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=True),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_api_usage_document_id_documents'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], name=op.f('fk_api_usage_session_id_chat_sessions'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_api_usage'))
    )
    op.create_index(op.f('ix_api_usage_created_at'), 'api_usage', ['created_at'], unique=False)
    op.create_index('ix_usage_operation_created', 'api_usage', ['operation', 'created_at'], unique=False)
    op.create_table('chat_messages',
    sa.Column('session_id', sa.UUID(), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('citations', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False),
    sa.Column('suggested_questions', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False),
    sa.Column('token_usage', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=True),
    sa.Column('model', sa.String(length=64), nullable=True),
    sa.Column('grounded', sa.Boolean(), nullable=True),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], name=op.f('fk_chat_messages_session_id_chat_sessions'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_chat_messages'))
    )
    op.create_index(op.f('ix_chat_messages_created_at'), 'chat_messages', ['created_at'], unique=False)
    op.create_index('ix_messages_session_created', 'chat_messages', ['session_id', 'created_at'], unique=False)
    op.create_table('chat_session_documents',
    sa.Column('session_id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_chat_session_documents_document_id_documents'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], name=op.f('fk_chat_session_documents_session_id_chat_sessions'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('session_id', 'document_id', name=op.f('pk_chat_session_documents')),
    sa.UniqueConstraint('session_id', 'document_id', name='uq_session_document')
    )
    op.create_table('document_chunks',
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('page_number', sa.Integer(), nullable=True),
    sa.Column('token_count', sa.Integer(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1536), nullable=True),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_document_chunks_document_id_documents'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_document_chunks')),
    sa.UniqueConstraint('document_id', 'chunk_index', name='uq_chunk_document_index')
    )
    op.create_index('ix_chunks_document', 'document_chunks', ['document_id'], unique=False)
    op.create_index(op.f('ix_document_chunks_created_at'), 'document_chunks', ['created_at'], unique=False)
    op.create_table('document_insights',
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('summary', sa.Text(), nullable=False),
    sa.Column('key_points', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False),
    sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False),
    sa.Column('entities', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False),
    sa.Column('category', sa.String(length=32), nullable=False),
    sa.Column('sentiment', sa.String(length=16), nullable=False),
    sa.Column('language', sa.String(length=16), nullable=False),
    sa.Column('model', sa.String(length=64), nullable=False),
    sa.Column('token_usage', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('was_truncated', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_document_insights_document_id_documents'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_document_insights'))
    )
    op.create_index(op.f('ix_document_insights_created_at'), 'document_insights', ['created_at'], unique=False)
    op.create_index(op.f('ix_document_insights_document_id'), 'document_insights', ['document_id'], unique=True)
    op.create_table('processing_metrics',
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('stage', sa.String(length=16), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('duration_ms', sa.Integer(), nullable=False),
    sa.Column('items_processed', sa.Integer(), nullable=True),
    sa.Column('error_code', sa.String(length=64), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_processing_metrics_document_id_documents'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_processing_metrics'))
    )
    op.create_index('ix_metrics_document_stage', 'processing_metrics', ['document_id', 'stage'], unique=False)
    op.create_index('ix_metrics_stage_status', 'processing_metrics', ['stage', 'status'], unique=False)
    op.create_index(op.f('ix_processing_metrics_created_at'), 'processing_metrics', ['created_at'], unique=False)
    # Approximate-nearest-neighbour index for cosine similarity retrieval.
    # HNSW gives better recall/latency than IVFFlat and needs no training pass.
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.drop_index(op.f('ix_processing_metrics_created_at'), table_name='processing_metrics')
    op.drop_index('ix_metrics_stage_status', table_name='processing_metrics')
    op.drop_index('ix_metrics_document_stage', table_name='processing_metrics')
    op.drop_table('processing_metrics')
    op.drop_index(op.f('ix_document_insights_document_id'), table_name='document_insights')
    op.drop_index(op.f('ix_document_insights_created_at'), table_name='document_insights')
    op.drop_table('document_insights')
    op.drop_index(op.f('ix_document_chunks_created_at'), table_name='document_chunks')
    op.drop_index('ix_chunks_document', table_name='document_chunks')
    op.drop_table('document_chunks')
    op.drop_table('chat_session_documents')
    op.drop_index('ix_messages_session_created', table_name='chat_messages')
    op.drop_index(op.f('ix_chat_messages_created_at'), table_name='chat_messages')
    op.drop_table('chat_messages')
    op.drop_index('ix_usage_operation_created', table_name='api_usage')
    op.drop_index(op.f('ix_api_usage_created_at'), table_name='api_usage')
    op.drop_table('api_usage')
    op.drop_index('ix_documents_user_status', table_name='documents')
    op.drop_index(op.f('ix_documents_user_id'), table_name='documents')
    op.drop_index('ix_documents_user_created', table_name='documents')
    op.drop_index(op.f('ix_documents_status'), table_name='documents')
    op.drop_index(op.f('ix_documents_created_at'), table_name='documents')
    op.drop_index(op.f('ix_documents_content_hash'), table_name='documents')
    op.drop_table('documents')
    op.drop_index('ix_sessions_user_updated', table_name='chat_sessions')
    op.drop_index(op.f('ix_chat_sessions_user_id'), table_name='chat_sessions')
    op.drop_index(op.f('ix_chat_sessions_created_at'), table_name='chat_sessions')
    op.drop_table('chat_sessions')
