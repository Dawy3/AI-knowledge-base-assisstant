"""
Embedding Generation for RAG Pipeline.

FOCUS: Batch embeddings (100-500 per call)
CRITICAL: NEVER mix models (index + query must match)
MUST: Cache embeddings for unchanged docs

Implementation follows best practices:
1. Batch texts for efficient API usage
2. Cache embeddings keyed by content hash + model ID
3. Retry with exponential backoff
4. Track model version for consistency
"""

import asyncio
import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
import cohere

from ..config import settings
from .models import EmbeddingModel, EmbeddingModelProvider, get_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    
    embeddings: list[list[float]]
    model_id: str
    dimensions: int
    
    # Metadata
    texts_count: int = 0
    total_tokens: int = 0
    cached_count: int = 0
    generated_count: int = 0
    
    # Performance
    latency_ms: float = 0.0
    
    # Cost tracking
    estimated_cost: float = 0.0
    
    
@dataclass
class CachedEmbedding:
    """Cached embedding with metadata."""
    
    embedding: list[float]
    model_id: str
    content_hash: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    

class EmbeddingCache(ABC):
    """Abstract base class for embedding cache."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CachedEmbedding]:
        """Get cached embedding by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, embedding: CachedEmbedding) -> None:
        """Store embedding in cache."""
        pass
    
    @abstractmethod
    async def get_many(self, keys: list[str]) -> dict[str, CachedEmbedding]:
        """Get multiple cached embeddings."""
        pass
    
    @abstractmethod
    async def set_many(self, items: dict[str, CachedEmbedding]) -> None:
        """Store multiple embeddings."""
        pass
    
    
class InMemoryEmbeddingCache(EmbeddingCache):
    """Simple in-memory embedding cache for development/testing."""
    
    def __init__(self, max_size: int = 10000):
        self._cache: dict[str, CachedEmbedding] = {}
        self._max_size = max_size
        
    async def get(self, key: str) -> Optional[CachedEmbedding]:
        cached = self._cache.get(key)
        if cached:
            cached.access_count += 1
        return cached
    
    async def set(self, key: str, embedding: CachedEmbedding) -> None:
        if len(self._cache) >= self._max_size:
            self._evict()
        self._cache[key] = embedding
        
    async def get_many(self, keys: list[str]) -> dict[str, CachedEmbedding]:
        result = {}
        for key in keys:
            if key in self._cache:
                self._cache[key].access_count += 1
                result[key] = self._cache[key]
        return result
    
    async def set_many(self, items: dict[str, CachedEmbedding]) -> None:
        for key, embedding in items.items():
            await self.set(key, embedding)
            
            
    def _evict(self) -> None:
        """Evict least recently used items."""
        # Simple LRU: remove 10% of least accessed items
        evict_count = max(1, len(self._cache) // 10)
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: x[1].access_count
        )
        for key, _ in sorted_items[:evict_count]:
            del self._cache[key]
            

class BaseEmbeddingClient(ABC):
    """Abstract base class for embedding API client."""
    
    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: EmbeddingModel
    ) -> list[list[float]]:
        """Generate embedding for texts."""
        pass
    
    
class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI embedding API client."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    async def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client
    
    async def embed(
        self,
        texts: list[str],
        model: EmbeddingModel,
    ) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        client = await self._get_client()
        
        response = await client.embeddings.create(
            model=model.name,
            input=texts,
            encoding_format="float",
        )

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
    

class CohereEmbeddingClient(BaseEmbeddingClient):
    """Cohere embedding API client."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self._client = None
    
    async def _get_client(self):
        """Lazy initialization of Cohere client."""
        if self._client is None:
            try:
                self._client = cohere.AsyncClient(self.api_key)
            except ImportError:
                raise ImportError("cohere package required: pip install cohere")
        return self._client
    
    async def embed(
        self,
        texts: list[str],
        model: EmbeddingModel,
        input_type: str = "search_document",
    ) -> list[list[float]]:
        """Generate embeddings using Cohere API."""
        client = await self._get_client()
        
        response = await client.embed(
            texts=texts,
            model=model.name,
            input_type=input_type,
        )
        
        return response.embeddings

class HuggingFaceEmbeddingClient(BaseEmbeddingClient):
    """HuggingFace/local model embedding client."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._tokenizer = None
        
        
    async def _load_model(self, model: EmbeddingModel):
        """Lazy load model and tokenizer."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    model.name,
                    device=self.device
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
    
    async def embed(
        self,
        texts: list[str],
        model: EmbeddingModel,
    ) -> list[list[float]]:
        """Generate embeddings using local HuggingFace model."""
        await self._load_model(model)
        
        # Run in executor to avoid blocking 
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                normalize_embeddings=model.normalize_embeddings,
                convert_to_numpy=True,
            ).tolist()
        )
        
        return embeddings
    

class EmbeddingGenerator:
    """
    Main embedding generator with batching, caching, and model consistency.
    
    FOCUS: Batch embeddings (100-500 per call)
    CRITICAL: NEVER mix models (index + query must match)
    MUST: Cache embeddings for unchanged docs
    
    Usage:
        generator = EmbeddingGenerator(model_key="openai/text-embedding-3-small")
        result = await generator.embed_texts(texts)
    """
    def __init__(
        self,
        model_key: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None,
        enable_cache: Optional[bool] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: float = 1.0,
    ):
        """
        Initialize embedding generator.

        Args:
            model_key: Model identifier from supported models (defaults to config)
            cache: Optional cache implementation
            enable_cache: Whether to use caching (defaults to config)
            batch_size: Override default batch size (defaults to config)
            max_retries: Maximum retry attempts (defaults to config)
            retry_delay: Initial retry delay in seconds
        """
        # Use config defaults if not specified
        if model_key is None:
            model_key = f"{settings.embedding.model_provider}/{settings.embedding.model_name}"
        if enable_cache is None:
            enable_cache = settings.cache.embedding_cache_enabled
        if batch_size is None:
            batch_size = settings.embedding.batch_size
        if max_retries is None:
            max_retries = settings.embedding.max_retries

        self.model = get_embedding_model(model_key)
        self.enable_cache = enable_cache
        self.batch_size = batch_size or self.model.batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize Cache
        self._cache = cache if enable_cache else None
        if enable_cache and cache is None:
            self._cache = InMemoryEmbeddingCache()
            
        # Initialize client based on provider
        self._client = self._create_client()
        
        logger.info(
            f"Initialized EmbeddingGenerator with model: {self.model.model_id}"
        )
        
    def _create_client(self) -> BaseEmbeddingClient:
        """Create appropriate client for model provider."""
        clients = {
            EmbeddingModelProvider.OPENAI: OpenAIEmbeddingClient,
            EmbeddingModelProvider.COHERE: CohereEmbeddingClient,
            EmbeddingModelProvider.HUGGINGFACE: HuggingFaceEmbeddingClient,
            EmbeddingModelProvider.LOCAL: HuggingFaceEmbeddingClient,
        }
        
        client_class = clients.get(self.model.provider)
        if not client_class:
            raise ValueError(f"Unsupported provider: {self.model.provider}")
        
        return client_class()
    
    @property
    def model_id(self) -> str:
        """Get current model identifier."""
        return self.model.model_id
    
    def _compute_content_hash(self, text: str) -> str:
        """Compute hash of text content for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key combining content hash and model ID.
        
        CRITICAL: Model ID is included to prevent mixing embeddings
        from different models.
        """
        content_hash = self._compute_content_hash(text)
        # Include model ID to ensure cache invalidation on model change
        model_hash = hashlib.sha256(self.model_id.encode()).hexdigest()[:8]
        return f"em:{model_hash}:{content_hash}"
    
    async def embed_texts(
        self,
        texts: list[str],
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        FOCUS: Batch embeddings (100-500 per call)
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache for this request
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()

        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model_id=self.model_id,
                dimensions=self.model.dimensions,
            )
            
        # Prepare cache keys
        cache_keys = [self._get_cache_key(text) for text in texts]
        
        # Try to get from cache
        cached_embeddings: dict[str, CachedEmbedding] = {}
        texts_to_embed: list[tuple[int, str]] = []  # (index, text)
        
        if use_cache and self._cache:
            cached_embeddings = await self._cache.get_many(cache_keys)
            
            for i, (text, key) in enumerate(zip(texts, cache_keys)):
                if key not in cached_embeddings:
                    texts_to_embed.append((i, text))
                    
        else:
            texts_to_embed = list(enumerate(texts))
            
        # Generate embeddings for uncached texts
        new_embeddings: dict[int, list[float]] = {}

        if texts_to_embed:
            # Process in batches
            new_embeddings = await self._embed_in_batches(texts_to_embed)
            
            # Cache new embeddings
            if use_cache and self._cache:
                cache_items = {}
                for idx , embedding in new_embeddings.items():
                    key = cache_keys[idx]
                    cache_items[key] = CachedEmbedding(
                        embedding = embedding,
                        model_id=self.model_id,
                        content_hash=self._compute_content_hash(texts[idx]),
                    )
                await self._cache.set_many(cache_items)
        
        # Combine cached and new embeddings in original order
        final_embeddings = []
        for i, key in enumerate(cache_keys):
            if key in cached_embeddings:
                final_embeddings.append(cached_embeddings[key].embedding)
            else:
                final_embeddings.append(new_embeddings[i])
                
        latency_ms = (time.time() - start_time) * 1000
        
        # Estimate cost
        total_tokens = sum(len(text.split()) for text in texts)  # Approximate
        estimated_cost = (total_tokens/ 1_000_000) * self.model.cost_per_million_tokens
        
        return EmbeddingResult(
            embeddings=final_embeddings,
            model_id=self.model_id,
            dimensions=self.model.dimensions,
            texts_count=len(texts),
            total_tokens=total_tokens,
            cached_count=len(cached_embeddings),
            generated_count=len(new_embeddings),
            latency_ms=latency_ms,
            estimated_cost=estimated_cost,
        )
        
    async def _embed_in_batches(
        self,
        texts_with_indices: list[tuple[int, str]],
    ) -> dict[int, list[float]]:
        """
        Embed texts in batches with retries.
        
        FOCUS: Batch embeddings (100-500 per call)
        """
        results: dict[int, list[float]] = {}
        # Sort by index for consistent batching
        texts_with_indices = sorted(texts_with_indices, key=lambda x: x[0])

        # Process batches
        for batch_start in range(0, len(texts_with_indices), self.batch_size):
            batch = texts_with_indices[batch_start:batch_start + self.batch_size]
            batch_indices = [idx for idx, _ in batch]
            batch_texts = [text for _, text in batch]
            
            # Embed with retries
            embeddings = await self._embed_with_retry(batch_texts)
            
            # Map back to original indices
            for idx, embedding in zip(batch_indices, embeddings):
                results[idx] = embedding
        
        return results
    
    async def _embed_with_retry(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed texts with exponential backoff retry."""
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                return await self._client.embed(texts, self.model)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embedding attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
        
        raise RuntimeError(
            f"Failed to generate embeddings after {self.max_retries} attempts: {last_error}"
        )
        
    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query text.
        
        Convenience method for search queries.
        """
        result = await self.embed_texts([query], use_cache=False)
        return result.embeddings[0]
    
    async def embed_documents(
        self,
        documents: list[str],
        document_ids: Optional[list[str]] = None,
    ) -> EmbeddingResult:
        """
        Embed documents with caching.
        
        MUST: Cache embeddings for unchanged docs
        """
        return await self.embed_texts(documents, use_cache=True)
    
    def validate_model_consistency(self, other_model_id: str) -> bool:
        """
        CRITICAL: Validate that models match.
        
        Index and query embeddings MUST use the same model.
        """
        is_consistent = self.model_id == other_model_id
        
        if not is_consistent:
            logger.error(
                f"Model mismatch! Current: {self.model_id}, "
                f"Other: {other_model_id}. This will produce garbage results."
            )
        
        return is_consistent
        
        
def create_embedding_generator(
    model_key: Optional[str] = None,
    **kwargs,
) -> EmbeddingGenerator:
    """
    Factory function to create an embedding generator.

    Args:
        model_key: Model identifier (defaults to config settings)
        **kwargs: Additional arguments for EmbeddingGenerator

    Returns:
        Configured EmbeddingGenerator instance

    Example:
        >>> generator = create_embedding_generator()  # Uses config
        >>> result = await generator.embed_texts(["Hello world"])
    """
    return EmbeddingGenerator(model_key=model_key, **kwargs)

        
            