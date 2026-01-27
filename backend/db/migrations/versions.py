"""Initial tables

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-01
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Conversations table
    op.create_table(
        'conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', sa.String(255), index=True),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('metadata', postgresql.JSONB(), default={}),
    )
    
    # Messages table
    op.create_table(
        'messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('conversations.id'), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('token_count', sa.Integer(), default=0),
        sa.Column('model', sa.String(100)),
        sa.Column('latency_ms', sa.Float()),
        sa.Column('chunks_used', postgresql.JSONB(), default=[]),
    )
    op.create_index('idx_messages_conversation', 'messages', ['conversation_id'])
    
    # Feedback table
    op.create_table(
        'feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('message_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('messages.id')),
        sa.Column('rating', sa.Integer()),
        sa.Column('feedback_type', sa.String(50)),
        sa.Column('comment', sa.Text()),
        sa.Column('query', sa.Text()),
        sa.Column('response', sa.Text()),
        sa.Column('chunks_used', postgresql.JSONB(), default=[]),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('user_id', sa.String(255)),
    )
    op.create_index('idx_feedback_message', 'feedback', ['message_id'])
    op.create_index('idx_feedback_rating', 'feedback', ['rating'])
    
    # Document chunks table
    op.create_table(
        'document_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', sa.String(255), nullable=False, index=True),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), default=0),
        sa.Column('source_file', sa.String(500)),
        sa.Column('start_char', sa.Integer()),
        sa.Column('end_char', sa.Integer()),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('embedding_model', sa.String(100)),
        sa.Column('is_embedded', sa.Boolean(), default=False),
    )
    op.create_index('idx_chunks_embedded', 'document_chunks', ['is_embedded'])
    
    # Query logs table
    op.create_table(
        'query_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('query_id', sa.String(100), unique=True, index=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('query_type', sa.String(50)),
        sa.Column('chunks_retrieved', postgresql.JSONB(), default=[]),
        sa.Column('retrieval_latency_ms', sa.Float()),
        sa.Column('search_type', sa.String(50)),
        sa.Column('response', sa.Text()),
        sa.Column('model', sa.String(100)),
        sa.Column('generation_latency_ms', sa.Float()),
        sa.Column('prompt_tokens', sa.Integer()),
        sa.Column('completion_tokens', sa.Integer()),
        sa.Column('total_latency_ms', sa.Float()),
        sa.Column('cache_hit', sa.Boolean(), default=False),
        sa.Column('user_id', sa.String(255), index=True),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True)),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now(), index=True),
    )


def downgrade() -> None:
    op.drop_table('query_logs')
    op.drop_table('document_chunks')
    op.drop_table('feedback')
    op.drop_table('messages')
    op.drop_table('conversations')