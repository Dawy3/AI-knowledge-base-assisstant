"""
Configuration Management for RAG backend

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


class Enviornment(str, Enum):
    """Application environement"""
    DEVELPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    
    
class EmbeddingModelConfig(BaseSettings):
    """
    Embedding model configuration.
    
    CRITICAL: Embedding model version must be tracked.
    Changing the model requires full reindexing of all documents.
    Index and query embedding MUST use the same model.
    """
    
    # Model Identification - CRITICAL for consistency 
    model_name: str  = Field(
        default="text-embedding-3-small",
        description="Embedding model name (OpenAI, Cohere, or local)"
    )
    model_version: str = Field(
        default="v1",
        description="Version tag for tracking index compatibility"
    )
    model_provider: str = Field(
        default= "openai",
        description="Provider: openai, cohere, huggingface, local"
    )
    
    # Model specification 
    dimensions: int = Field(
        default= 1536,
        description= "Embedding vector dimensions"
    )
    
    max_tokens: int = Field(
        default=8191,
        description="Maximum token per embedding request - Must handle your chunk size"
    )
    
    # Batching Cnofiguration
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
        description="similarity threshold for semantic chunking"
    )
    
    # Minimum chunk size to avoid tiny fragmets 
    min_chunk_size: int = Field(
        default=100,
        description="Minimum chunk size in tokens"
    )
    
    # Separators for recursive chunking (in order of priority)
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Text separators for recursive chunking"
    )
    