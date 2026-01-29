"""
SQLAlchemy Models.

FOCUS: Conversations, feedback, document chunks
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Conversation(Base):
    """Conversation/session tracking."""
    
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_metadata = Column(JSONB, default={})  # Renamed from metadata
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """Individual messages in a conversation."""
    
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    token_count = Column(Integer, default=0)
    model = Column(String(100))
    latency_ms = Column(Float)
    
    # For RAG tracking
    chunks_used = Column(JSONB, default=[])  # List of chunk IDs used
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    __table_args__ = (
        Index("idx_messages_conversation", "conversation_id"),
    )


class Feedback(Base):
    """User feedback on responses."""
    
    __tablename__ = "feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"))
    
    # Feedback data
    rating = Column(Integer)  # 1 (thumbs down) or 5 (thumbs up)
    feedback_type = Column(String(50))  # thumbs, rating, text
    comment = Column(Text)
    
    # Context for evaluation dataset
    query = Column(Text)
    response = Column(Text)
    chunks_used = Column(JSONB, default=[])
    
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String(255))
    
    __table_args__ = (
        Index("idx_feedback_message", "message_id"),
        Index("idx_feedback_rating", "rating"),
    )


class DocumentChunk(Base):
    """Document chunks with metadata."""

    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(String(255), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)

    # Content
    content = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)

    # Source tracking
    source_file = Column(String(500))
    start_char = Column(Integer)
    end_char = Column(Integer)

    # Metadata
    chunk_metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Embedding tracking
    embedding_model = Column(String(100))
    is_embedded = Column(Boolean, default=False)
    
    __table_args__ = (
        Index("idx_chunks_document", "document_id"),
        Index("idx_chunks_embedded", "is_embedded"),
    )


class Document(Base):
    """Uploaded documents for RAG ingestion."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(500), nullable=False)
    file_type = Column(String(50))
    file_size = Column(Integer)

    # Processing status
    status = Column(String(50), default="queued", index=True)  # queued, processing, embedding, completed, failed
    chunks_created = Column(Integer, default=0)
    chunks_embedded = Column(Integer, default=0)
    error_message = Column(Text)

    # Metadata
    title = Column(String(500))
    description = Column(Text)
    tags = Column(JSONB, default=[])
    category = Column(String(100))
    source_url = Column(String(1000))
    author = Column(String(255))

    # Chunking configuration used
    chunk_strategy = Column(String(50))
    chunk_size = Column(Integer)
    chunk_overlap = Column(Integer)

    # Tracking
    user_id = Column(String(255), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processing_time_ms = Column(Float)

    __table_args__ = (
        Index("idx_documents_user", "user_id"),
        Index("idx_documents_status", "status"),
    )


class QueryLog(Base):
    """Log of all queries for evaluation dataset building."""

    __tablename__ = "query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(String(100), unique=True, index=True)
    
    # Query data
    query = Column(Text, nullable=False)
    query_type = Column(String(50))  # simple, complex, faq
    
    # Retrieval data
    chunks_retrieved = Column(JSONB, default=[])
    retrieval_latency_ms = Column(Float)
    search_type = Column(String(50))  # hybrid, vector, bm25
    
    # Generation data
    response = Column(Text)
    model = Column(String(100))
    generation_latency_ms = Column(Float)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    
    # Performance
    total_latency_ms = Column(Float)
    cache_hit = Column(Boolean, default=False)
    
    # Context
    user_id = Column(String(255), index=True)
    conversation_id = Column(UUID(as_uuid=True))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("idx_query_logs_created", "created_at"),
    )