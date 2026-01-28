"""
Configuration Management for RAG Backend.

FOCUS: Environment variables, model configs
MUST: Separate dev/staging/prod configs
CRITICAL: Store embedding model version - changing models requires full reindex
"""

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class EmbeddingModelConfig(BaseSettings):
    """
    Embedding model configuration.
    
    CRITICAL: Embedding model version MUST be tracked.
    Changing the model requires full reindexing of all documents.
    Index and query embeddings MUST use the same model.
    """
    
    # Model identification - CRITICAL for consistency
    model_name: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name (OpenAI, Cohere, or local)"
    )
    model_version: str = Field(
        default="v1",
        description="Version tag for tracking index compatibility"
    )
    model_provider: str = Field(
        default="openai",
        description="Provider: openai, cohere, huggingface, local"
    )
    
    # Model specifications
    dimensions: int = Field(
        default=1536,
        description="Embedding vector dimensions"
    )
    max_tokens: int = Field(
        default=8191,
        description="Maximum tokens per embedding request - MUST handle your chunk size"
    )
    
    # Batching configuration
    batch_size: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Batch size for embedding generation (100-500 optimal)"
    )
    
    # Performance
    request_timeout: float = Field(default=30.0)
    max_retries: int = Field(default=3)
    
    @property
    def model_identifier(self) -> str:
        """Unique identifier for this model configuration."""
        return f"{self.model_provider}/{self.model_name}/{self.model_version}"


class ChunkingConfig(BaseSettings):
    """
    Chunking configuration.
    
    FOCUS: Recursive chunking with 512 tokens, 50 overlap
    OPTIONS: Fixed, Semantic, Sentence, Document, Page-level
    """
    
    strategy: str = Field(
        default="recursive",
        description="Chunking strategy: fixed, recursive, semantic, sentence, document, page"
    )
    
    # Size configuration (in tokens)
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Target chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Overlap between chunks in tokens"
    )
    
    # Semantic chunking settings
    semantic_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for semantic chunking"
    )
    
    # Minimum chunk size to avoid tiny fragments
    min_chunk_size: int = Field(
        default=100,
        description="Minimum chunk size in tokens"
    )
    
    # Separators for recursive chunking (in order of priority)
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Text separators for recursive chunking"
    )


class RetrievalConfig(BaseSettings):
    """Retrieval and search configuration."""
    
    # Hybrid search weights - CRITICAL for quality
    vector_weight: float = Field(default=5.0, description="Weight for vector similarity")
    bm25_weight: float = Field(default=3.0, description="Weight for BM25 keyword match")
    recency_weight: float = Field(default=0.2, description="Weight for document recency")
    
    # Search parameters
    top_k_retrieval: int = Field(default=100, description="Initial retrieval count")
    top_k_rerank: int = Field(default=10, description="Final count after reranking")
    
    # HNSW index parameters
    hnsw_m: int = Field(default=16, description="HNSW M parameter")
    hnsw_ef_search: int = Field(default=100, description="HNSW ef_search for 95%+ recall")
    
    # Relevance filtering
    relevance_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score (drop below this)"
    )


class CacheConfig(BaseSettings):
    """
    Caching configuration.
    
    FOCUS: Semantic cache for 50-70% cost reduction
    TARGET: 38%+ cache hit rate
    """
    
    # Semantic cache
    semantic_cache_enabled: bool = Field(default=True)
    semantic_cache_threshold: float = Field(
        default=0.92,
        ge=0.85,
        le=0.99,
        description="Similarity threshold for cache hits (start conservative)"
    )
    semantic_cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    
    # Embedding cache
    embedding_cache_enabled: bool = Field(default=True)
    embedding_cache_ttl: int = Field(
        default=86400,
        description="Embedding cache TTL (24 hours)"
    )
    
    # Redis configuration
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_max_connections: int = Field(default=10)


class LLMConfig(BaseSettings):
    """LLM configuration for generation."""
    
    # Primary model (local fine-tuned)
    local_model_name: str = Field(default="meta-llama/Llama-2-7b-chat-hf")
    local_model_endpoint: str = Field(default="http://localhost:8080/v1")
    
    # Fallback model (OpenAI)
    openai_model: str = Field(default="gpt-4-turbo-preview")
    openai_api_key: Optional[str] = Field(default=None)
    
    # Routing thresholds
    complexity_threshold: float = Field(
        default=0.7,
        description="Route to GPT-4 if complexity > threshold"
    )
    local_model_ratio: float = Field(
        default=0.82,
        description="Target: 82% queries to local model"
    )
    
    # Generation parameters
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.7)
    streaming_enabled: bool = Field(default=True)
    
    # Timeouts and retries
    request_timeout: float = Field(default=30.0)
    max_retries: int = Field(default=3)


class ContextConfig(BaseSettings):
    """Context window management configuration."""
    
    # Dynamic context sizing
    simple_query_tokens: int = Field(default=800, description="Context for simple queries")
    complex_query_tokens: int = Field(default=4000, description="Context for complex queries")
    
    # Conversation history
    full_history_messages: int = Field(
        default=3,
        description="Number of recent messages to keep in full"
    )
    summary_token_limit: int = Field(
        default=150,
        description="Token limit for summarized older messages"
    )


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    log_all_queries: bool = Field(default=True)
    
    # Metrics
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    
    # Tracing
    tracing_enabled: bool = Field(default=True)
    tracing_endpoint: str = Field(default="http://localhost:4317")
    tracing_sample_rate: float = Field(default=1.0)
    
    # Performance targets
    target_recall_at_10: float = Field(default=0.80)
    target_ndcg: float = Field(default=0.75)
    target_p99_latency_ms: int = Field(default=200)


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    # PostgreSQL (metadata + pgvector for small scale)
    postgres_url: str = Field(
        default="postgresql://user:password@localhost:5432/ragdb"
    )
    postgres_pool_size: int = Field(default=10)
    
    # Vector store selection based on scale
    vector_store: str = Field(
        default="pgvector",
        description="Vector store: pgvector (<10M), qdrant (10M-500M), milvus (>500M)"
    )
    
    # Qdrant configuration
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: Optional[str] = Field(default=None)
    
    # Milvus configuration
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)


class Settings(BaseSettings):
    """
    Main application settings.
    
    Loads configuration based on environment.
    Uses Pydantic settings for validation and environment variable loading.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    # Application
    app_name: str = Field(default="RAG Knowledge Assistant")
    app_version: str = Field(default="1.0.0")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    
    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_prefix: str = Field(default="/api/v1")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)
    
    # CORS
    cors_origins: list[str] = Field(default=["*"])
    
    # Sub-configurations
    embedding: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default=None)
    
    @field_validator("data_dir", mode="before")
    @classmethod
    def set_data_dir(cls, v, info):
        if v is None:
            base = info.data.get("base_dir", Path(__file__).parent.parent.parent)
            return base / "data"
        return Path(v)
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    def get_embedding_model_id(self) -> str:
        """
        Get unique embedding model identifier.
        
        CRITICAL: This ID must be stored with every vector index.
        Querying with a different model will produce garbage results.
        """
        return self.embedding.model_identifier


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Settings are loaded once and cached for performance.
    To reload, clear the cache: get_settings.cache_clear()
    """
    return Settings()


# Global settings instance
settings = get_settings()


# Environment-specific overrides
def get_environment_settings(env: Environment) -> dict:
    """
    Get environment-specific setting overrides.
    
    MUST: Different configs for dev/staging/prod
    """
    overrides = {
        Environment.DEVELOPMENT: {
            "debug": True,
            "cache__semantic_cache_threshold": 0.90,  # More permissive for testing
            "monitoring__log_level": "DEBUG",
            "monitoring__tracing_sample_rate": 1.0,
        },
        Environment.STAGING: {
            "debug": False,
            "cache__semantic_cache_threshold": 0.92,
            "monitoring__log_level": "INFO",
            "monitoring__tracing_sample_rate": 0.5,
        },
        Environment.PRODUCTION: {
            "debug": False,
            "cache__semantic_cache_threshold": 0.95,  # Conservative for accuracy
            "monitoring__log_level": "WARNING",
            "monitoring__tracing_sample_rate": 0.1,
            "rate_limit_requests": 50,  # Stricter rate limiting
        },
    }
    return overrides.get(env, {})