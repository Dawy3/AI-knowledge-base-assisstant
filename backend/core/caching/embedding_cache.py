"""
Embedding Cache for RAG Pipeline.

Cache document embeddings to avoid re-embedding unchanged content.
Uses content hash as key - if content unchanged, reuse embedding.

EXPECTED: 10-20% cost savings on embedding API calls
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import redis

logger = logging.getLogger(__name__)


@dataclass
class CachedEmbedding:
    """Cached embedding with metadata."""
    embedding: list[float]
    model_id: str
    content_hash: str
    created_at: float


class EmbeddingCache:
    """
    Cache for document embeddings.
    
    Key: hash(content + model_id)
    Value: embedding vector
    
    Usage:
        cache = EmbeddingCache(redis_client)
        
        # Check cache first
        embedding = cache.get(content, model_id)
        if embedding is None:
            embedding = embed_func(content)
            cache.set(content, model_id, embedding)
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        ttl_seconds: int = 86400,  # 24 hours
        max_memory_size: int = 50000,
        prefix: str = "emb_cache",
    ):
        """
        Args:
            redis_client: Redis client (optional, uses in-memory if None)
            ttl_seconds: Cache TTL (default 24h)
            max_memory_size: Max entries in memory cache
            prefix: Redis key prefix
        """
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.max_size = max_memory_size
        self.prefix = prefix
        
        # In-memory fallback
        self._cache: dict[str, CachedEmbedding] = {}
        
        # Stats
        self._hits = 0
        self._misses = 0
        
        logger.info(f"EmbeddingCache initialized: ttl={ttl_seconds}s")
    
    def get(self, content: str, model_id: str) -> Optional[list[float]]:
        """
        Get cached embedding for content.
        
        Returns None if not cached or model mismatch.
        """
        key = self._make_key(content, model_id)
        
        # Try Redis first
        if self.redis:
            data = self.redis.get(f"{self.prefix}:{key}")
            if data:
                cached = json.loads(data)
                # Verify model matches
                if cached.get("model_id") == model_id:
                    self._hits += 1
                    return cached["embedding"]
        
        # Try memory cache
        if key in self._cache:
            cached = self._cache[key]
            if cached.model_id == model_id:
                self._hits += 1
                return cached.embedding
        
        self._misses += 1
        return None
    
    def set(
        self,
        content: str,
        model_id: str,
        embedding: list[float],
    ) -> None:
        """Cache embedding for content."""
        key = self._make_key(content, model_id)
        content_hash = self._hash_content(content)
        
        cached = CachedEmbedding(
            embedding=embedding,
            model_id=model_id,
            content_hash=content_hash,
            created_at=time.time(),
        )
        
        # Store in Redis
        if self.redis:
            data = {
                "embedding": embedding,
                "model_id": model_id,
                "content_hash": content_hash,
                "created_at": cached.created_at,
            }
            self.redis.setex(
                f"{self.prefix}:{key}",
                self.ttl,
                json.dumps(data),
            )
        
        # Store in memory
        self._memory_set(key, cached)
    
    def get_batch(
        self,
        contents: list[str],
        model_id: str,
    ) -> tuple[list[Optional[list[float]]], list[int]]:
        """
        Get cached embeddings for multiple contents.
        
        Returns:
            - List of embeddings (None for misses)
            - List of indices that need embedding
        """
        results = []
        missing_indices = []
        
        for i, content in enumerate(contents):
            embedding = self.get(content, model_id)
            results.append(embedding)
            if embedding is None:
                missing_indices.append(i)
        
        return results, missing_indices
    
    def set_batch(
        self,
        contents: list[str],
        model_id: str,
        embeddings: list[list[float]],
    ) -> None:
        """Cache multiple embeddings."""
        for content, embedding in zip(contents, embeddings):
            self.set(content, model_id, embedding)
    
    def _make_key(self, content: str, model_id: str) -> str:
        """Create cache key from content and model ID."""
        combined = f"{model_id}:{content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    def _hash_content(self, content: str) -> str:
        """Hash content for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _memory_set(self, key: str, cached: CachedEmbedding) -> None:
        """Store in memory with eviction."""
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        self._cache[key] = cached
    
    def _evict_oldest(self) -> None:
        """Remove oldest entry."""
        if not self._cache:
            return
        oldest = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest]
    
    def invalidate(self, content: str, model_id: str) -> None:
        """Remove specific entry from cache."""
        key = self._make_key(content, model_id)
        
        if self.redis:
            self.redis.delete(f"{self.prefix}:{key}")
        
        if key in self._cache:
            del self._cache[key]
    
    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        
        if self.redis:
            keys = self.redis.keys(f"{self.prefix}:*")
            if keys:
                self.redis.delete(*keys)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "size": len(self._cache),
        }