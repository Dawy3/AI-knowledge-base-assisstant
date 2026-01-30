"""
Configuration Management for RAG Backend.

All field names match their .env variable names exactly (UPPERCASE).
Lowercase property aliases provided for code compatibility.

Example .env:
    LLM_PROVIDER=openrouter
    LLM_MODEL=openai/gpt-4o-mini
    EMBEDDING_MODEL_NAME=e5-large-v2
    QDRANT_URL=http://localhost:6333
"""

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# =============================================================================
# EMBEDDING CONFIG
# =============================================================================
class EmbeddingModelConfig(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Fields match env var names
    EMBEDDING_MODEL_NAME: str = Field(default="text-embedding-3-small")
    EMBEDDING_MODEL_VERSION: str = Field(default="v1")
    EMBEDDING_MODEL_PROVIDER: str = Field(default="openai")
    EMBEDDING_DIMENSIONS: int = Field(default=1536)
    EMBEDDING_MAX_TOKENS: int = Field(default=8191)
    EMBEDDING_BATCH_SIZE: int = Field(default=100)
    EMBEDDING_TIMEOUT: float = Field(default=30.0)
    EMBEDDING_MAX_RETRIES: int = Field(default=3)

    # Lowercase aliases for code compatibility
    @property
    def model_name(self) -> str:
        return self.EMBEDDING_MODEL_NAME

    @property
    def model_version(self) -> str:
        return self.EMBEDDING_MODEL_VERSION

    @property
    def model_provider(self) -> str:
        return self.EMBEDDING_MODEL_PROVIDER

    @property
    def dimensions(self) -> int:
        return self.EMBEDDING_DIMENSIONS

    @property
    def max_tokens(self) -> int:
        return self.EMBEDDING_MAX_TOKENS

    @property
    def batch_size(self) -> int:
        return self.EMBEDDING_BATCH_SIZE

    @property
    def request_timeout(self) -> float:
        return self.EMBEDDING_TIMEOUT

    @property
    def max_retries(self) -> int:
        return self.EMBEDDING_MAX_RETRIES

    @property
    def model_identifier(self) -> str:
        return f"{self.EMBEDDING_MODEL_PROVIDER}/{self.EMBEDDING_MODEL_NAME}/{self.EMBEDDING_MODEL_VERSION}"


# =============================================================================
# CHUNKING CONFIG
# =============================================================================
class ChunkingConfig(BaseSettings):
    """Chunking configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    CHUNK_STRATEGY: str = Field(default="recursive")
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=50)
    CHUNK_MIN_SIZE: int = Field(default=100)
    CHUNK_SEMANTIC_THRESHOLD: float = Field(default=0.5)

    # Lowercase aliases
    @property
    def strategy(self) -> str:
        return self.CHUNK_STRATEGY

    @property
    def chunk_size(self) -> int:
        return self.CHUNK_SIZE

    @property
    def chunk_overlap(self) -> int:
        return self.CHUNK_OVERLAP

    @property
    def min_chunk_size(self) -> int:
        return self.CHUNK_MIN_SIZE

    @property
    def semantic_threshold(self) -> float:
        return self.CHUNK_SEMANTIC_THRESHOLD

    @property
    def separators(self) -> list:
        return ["\n\n", "\n", ". ", " ", ""]


# =============================================================================
# RETRIEVAL CONFIG
# =============================================================================
class RetrievalConfig(BaseSettings):
    """Retrieval and search configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    VECTOR_WEIGHT: float = Field(default=5.0)
    BM25_WEIGHT: float = Field(default=3.0)
    RECENCY_WEIGHT: float = Field(default=0.2)
    VECTOR_TOP_K: int = Field(default=100)
    RERANK_TOP_K: int = Field(default=10)
    HNSW_M: int = Field(default=16)
    HNSW_EF_SEARCH: int = Field(default=100)
    RELEVANCE_THRESHOLD: float = Field(default=0.6)

    # Lowercase aliases
    @property
    def vector_weight(self) -> float:
        return self.VECTOR_WEIGHT

    @property
    def bm25_weight(self) -> float:
        return self.BM25_WEIGHT

    @property
    def recency_weight(self) -> float:
        return self.RECENCY_WEIGHT

    @property
    def top_k_retrieval(self) -> int:
        return self.VECTOR_TOP_K

    @property
    def top_k_rerank(self) -> int:
        return self.RERANK_TOP_K

    @property
    def hnsw_m(self) -> int:
        return self.HNSW_M

    @property
    def hnsw_ef_search(self) -> int:
        return self.HNSW_EF_SEARCH

    @property
    def relevance_threshold(self) -> float:
        return self.RELEVANCE_THRESHOLD


# =============================================================================
# CACHE CONFIG
# =============================================================================
class CacheConfig(BaseSettings):
    """Caching configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    CACHE_SEMANTIC_ENABLED: bool = Field(default=True)
    CACHE_SEMANTIC_THRESHOLD: float = Field(default=0.92)
    CACHE_SEMANTIC_TTL: int = Field(default=3600)
    CACHE_EMBEDDING_ENABLED: bool = Field(default=True)
    CACHE_EMBEDDING_TTL: int = Field(default=86400)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_MAX_CONNECTIONS: int = Field(default=10)

    # Lowercase aliases
    @property
    def semantic_cache_enabled(self) -> bool:
        return self.CACHE_SEMANTIC_ENABLED

    @property
    def semantic_cache_threshold(self) -> float:
        return self.CACHE_SEMANTIC_THRESHOLD

    @property
    def semantic_cache_ttl(self) -> int:
        return self.CACHE_SEMANTIC_TTL

    @property
    def embedding_cache_enabled(self) -> bool:
        return self.CACHE_EMBEDDING_ENABLED

    @property
    def embedding_cache_ttl(self) -> int:
        return self.CACHE_EMBEDDING_TTL

    @property
    def redis_url(self) -> str:
        return self.REDIS_URL

    @property
    def redis_max_connections(self) -> int:
        return self.REDIS_MAX_CONNECTIONS


# =============================================================================
# LLM CONFIG
# =============================================================================
class LLMConfig(BaseSettings):
    """
    LLM configuration. Supports: openrouter, openai, local

    .env:
        LLM_PROVIDER=openrouter
        LLM_MODEL=openai/gpt-4o-mini
        OPENROUTER_API_KEY=sk-or-xxx
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Provider & Model
    LLM_PROVIDER: str = Field(default="openrouter")
    LLM_MODEL: str = Field(default="openai/gpt-4o-mini")

    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENROUTER_API_KEY: Optional[str] = Field(default=None)

    # Base URLs
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
    LOCAL_LLM_URL: str = Field(default="http://localhost:8080/v1")

    # Tier Models (OpenAI only)
    LLM_TIER1_MODEL: Optional[str] = Field(default=None)
    LLM_TIER2_MODEL: Optional[str] = Field(default=None)
    LLM_TIER3_MODEL: Optional[str] = Field(default=None)

    # Generation Parameters
    LLM_MAX_TOKENS: int = Field(default=1024)
    LLM_TEMPERATURE: float = Field(default=0.7)
    LLM_TIMEOUT: float = Field(default=30.0)
    LLM_MAX_RETRIES: int = Field(default=3)
    LLM_STREAMING: bool = Field(default=True)

    # Lowercase aliases
    @property
    def provider(self) -> str:
        return self.LLM_PROVIDER

    @property
    def model(self) -> str:
        return self.LLM_MODEL

    @property
    def api_key(self) -> Optional[str]:
        if self.LLM_PROVIDER == "openrouter":
            return self.OPENROUTER_API_KEY
        elif self.LLM_PROVIDER == "openai":
            return self.OPENAI_API_KEY
        return None

    @property
    def base_url(self) -> str:
        if self.LLM_PROVIDER == "openrouter":
            return self.OPENROUTER_BASE_URL
        elif self.LLM_PROVIDER == "local":
            return self.LOCAL_LLM_URL
        return self.OPENAI_BASE_URL

    @property
    def effective_tier1_model(self) -> str:
        return self.LLM_TIER1_MODEL or self.LLM_MODEL

    @property
    def effective_tier2_model(self) -> str:
        return self.LLM_TIER2_MODEL or self.LLM_MODEL

    @property
    def effective_tier3_model(self) -> str:
        return self.LLM_TIER3_MODEL or self.LLM_MODEL

    @property
    def max_tokens(self) -> int:
        return self.LLM_MAX_TOKENS

    @property
    def temperature(self) -> float:
        return self.LLM_TEMPERATURE

    @property
    def request_timeout(self) -> float:
        return self.LLM_TIMEOUT

    @property
    def max_retries(self) -> int:
        return self.LLM_MAX_RETRIES

    @property
    def streaming_enabled(self) -> bool:
        return self.LLM_STREAMING

    # Legacy
    @property
    def openai_api_key(self) -> Optional[str]:
        return self.OPENAI_API_KEY

    @property
    def openai_model(self) -> str:
        return self.LLM_MODEL


# =============================================================================
# CONTEXT CONFIG
# =============================================================================
class ContextConfig(BaseSettings):
    """Context window configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    CONTEXT_SIMPLE_TOKENS: int = Field(default=800)
    CONTEXT_COMPLEX_TOKENS: int = Field(default=4000)
    CONTEXT_HISTORY_MESSAGES: int = Field(default=3)
    CONTEXT_SUMMARY_LIMIT: int = Field(default=150)

    # Lowercase aliases
    @property
    def simple_query_tokens(self) -> int:
        return self.CONTEXT_SIMPLE_TOKENS

    @property
    def complex_query_tokens(self) -> int:
        return self.CONTEXT_COMPLEX_TOKENS

    @property
    def full_history_messages(self) -> int:
        return self.CONTEXT_HISTORY_MESSAGES

    @property
    def summary_token_limit(self) -> int:
        return self.CONTEXT_SUMMARY_LIMIT


# =============================================================================
# MONITORING CONFIG
# =============================================================================
class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")
    LOG_ALL_QUERIES: bool = Field(default=True)
    METRICS_ENABLED: bool = Field(default=True)
    METRICS_PORT: int = Field(default=9090)
    TRACING_ENABLED: bool = Field(default=True)
    TRACING_ENDPOINT: str = Field(default="http://localhost:4317")
    TRACING_SAMPLE_RATE: float = Field(default=1.0)

    # Lowercase aliases
    @property
    def log_level(self) -> str:
        return self.LOG_LEVEL

    @property
    def log_format(self) -> str:
        return self.LOG_FORMAT

    @property
    def log_all_queries(self) -> bool:
        return self.LOG_ALL_QUERIES

    @property
    def metrics_enabled(self) -> bool:
        return self.METRICS_ENABLED

    @property
    def metrics_port(self) -> int:
        return self.METRICS_PORT

    @property
    def tracing_enabled(self) -> bool:
        return self.TRACING_ENABLED

    @property
    def tracing_endpoint(self) -> str:
        return self.TRACING_ENDPOINT

    @property
    def tracing_sample_rate(self) -> float:
        return self.TRACING_SAMPLE_RATE


# =============================================================================
# DATABASE CONFIG
# =============================================================================
class DatabaseConfig(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL
    DATABASE_URL: str = Field(default="")
    POSTGRES_POOL_SIZE: int = Field(default=10)

    # Vector Store
    VECTOR_STORE: str = Field(default="qdrant")

    # Qdrant
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = Field(default=None)
    QDRANT_COLLECTION: str = Field(default="documents")

    # Pinecone
    PINECONE_API_KEY: Optional[str] = Field(default=None)
    PINECONE_ENVIRONMENT: str = Field(default="us-east-1")
    PINECONE_INDEX_NAME: str = Field(default="rag-documents")

    # Lowercase aliases
    @property
    def postgres_url(self) -> str:
        return self.DATABASE_URL

    @property
    def postgres_pool_size(self) -> int:
        return self.POSTGRES_POOL_SIZE

    @property
    def vector_store(self) -> str:
        return self.VECTOR_STORE

    @property
    def qdrant_url(self) -> str:
        return self.QDRANT_URL

    @property
    def qdrant_api_key(self) -> Optional[str]:
        return self.QDRANT_API_KEY

    @property
    def qdrant_collection(self) -> str:
        return self.QDRANT_COLLECTION


# =============================================================================
# MAIN SETTINGS
# =============================================================================
class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Application
    APP_NAME: str = Field(default="RAG Knowledge Assistant")
    APP_VERSION: str = Field(default="1.0.0")
    APP_ENV: str = Field(default="development")
    DEBUG: bool = Field(default=False)

    # API
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_PREFIX: str = Field(default="/api/v1")

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100)
    RATE_LIMIT_WINDOW: int = Field(default=60)

    # CORS
    CORS_ORIGINS: str = Field(default="*")

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
    data_dir: Optional[Path] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"

    # Lowercase aliases
    @property
    def app_name(self) -> str:
        return self.APP_NAME

    @property
    def app_version(self) -> str:
        return self.APP_VERSION

    @property
    def environment(self) -> Environment:
        return Environment(self.APP_ENV)

    @property
    def debug(self) -> bool:
        return self.DEBUG

    @property
    def api_host(self) -> str:
        return self.API_HOST

    @property
    def api_port(self) -> int:
        return self.API_PORT

    @property
    def api_prefix(self) -> str:
        return self.API_PREFIX

    @property
    def rate_limit_requests(self) -> int:
        return self.RATE_LIMIT_REQUESTS

    @property
    def rate_limit_window(self) -> int:
        return self.RATE_LIMIT_WINDOW

    @property
    def cors_origins(self) -> list:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"

    def get_embedding_model_id(self) -> str:
        return self.embedding.model_identifier


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
