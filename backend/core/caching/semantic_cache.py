"""
Semantic Cache for RAG Pipeline.

3-Layer Architecture:
- Layer 1: Exact match (hash lookup - fast)
- Layer 2: Semantic similarity (embedding search, threshold > 0.9)
- Layer 3: Cross-encoder validation (filter false positives)

Start conservative (0.95 threshold), tune down based on false positive rate.
TARGET: 38%+ cache hit rate, 50-70% cost reduction
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import redis

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cached query-response pair."""
    query: str
    response: str
    embedding: list[float]
    metadata: dict
    created_at: float
    hit_count: int = 0


@dataclass
class CacheResult:
    """Result from cache lookup."""
    hit: bool
    response: Optional[str] = None
    layer: Optional[str] = None  # "exact", "semantic", None
    similarity: float = 0.0
    latency_ms: float = 0.0


class SemanticCache:
    """
    3-layer semantic cache.
    
    Layer 1: Exact match (hash lookup)
    Layer 2: Semantic similarity (threshold > 0.9)
    Layer 3: Cross-encoder validation (filter false positives)
    
    Usage:
        cache = SemanticCache(embed_func, redis_client)
        
        # Check cache
        result = await cache.get(query)
        if result.hit:
            return result.response
        
        # Generate and cache
        response = await generate(query)
        await cache.set(query, response)
    """
    
    def __init__(
        self,
        embed_func: Callable[[str], list[float]],
        redis_client: Optional[redis.Redis] = None,
        reranker: Optional[Any] = None,
        similarity_threshold: Optional[float] = None,
        rerank_threshold: float = 0.7,
        ttl_seconds: Optional[int] = None,
        max_cache_size: int = 10000,
        prefix: str = "sem_cache",
    ):
        """
        Args:
            embed_func: Function to embed queries
            redis_client: Redis client (optional, uses in-memory if None)
            reranker: Cross-encoder reranker for Layer 3 validation
            similarity_threshold: Threshold for semantic match (defaults to config)
            rerank_threshold: Threshold for cross-encoder validation
            ttl_seconds: Cache TTL (defaults to config)
            max_cache_size: Max entries in cache
            prefix: Redis key prefix
        """
        self.embed_func = embed_func
        self.redis = redis_client
        self.reranker = reranker
        # Use config defaults if not specified
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else settings.cache.semantic_cache_threshold
        self.rerank_threshold = rerank_threshold
        self.ttl = ttl_seconds if ttl_seconds is not None else settings.cache.semantic_cache_ttl
        self.max_size = max_cache_size
        self.prefix = prefix
        
        # In-memory fallback
        self._memory_cache: dict[str, CacheEntry] = {}
        self._embeddings: list[tuple[str, np.ndarray]] = []  # (key, embedding)
        
        # Stats
        self._stats = {"hits_exact": 0, "hits_semantic": 0, "misses": 0}
        
        logger.info(
            f"SemanticCache initialized: threshold={similarity_threshold}, "
            f"ttl={ttl_seconds}s, reranker={'yes' if reranker else 'no'}"
        )
    
    async def get(self, query: str) -> CacheResult:
        """
        Look up query in cache (3 layers).
        
        Layer 1: Exact hash match
        Layer 2: Semantic similarity search
        Layer 3: Cross-encoder validation
        """
        start = time.perf_counter()
        
        # Layer 1: Exact match
        exact_result = await self._exact_match(query)
        if exact_result:
            self._stats["hits_exact"] += 1
            return CacheResult(
                hit=True,
                response=exact_result,
                layer="exact",
                similarity=1.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        
        # Layer 2: Semantic similarity
        query_embedding = np.array(self.embed_func(query))
        semantic_result = await self._semantic_match(query, query_embedding)
        
        if semantic_result:
            self._stats["hits_semantic"] += 1
            return CacheResult(
                hit=True,
                response=semantic_result["response"],
                layer="semantic",
                similarity=semantic_result["similarity"],
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        
        self._stats["misses"] += 1
        return CacheResult(
            hit=False,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
    
    async def set(
        self,
        query: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Cache a query-response pair."""
        # Log what we're caching
        response_preview = response[:100] if response else "(empty)"
        logger.info(f"Caching response for query='{query[:50]}...': response_len={len(response) if response else 0}, preview='{response_preview}...'")

        # Skip empty responses to avoid caching failures
        if not response or not response.strip():
            logger.warning(f"Skipping cache for empty response: query='{query[:50]}...'")
            return

        query_hash = self._hash_query(query)

        try:
            embedding = self.embed_func(query)
        except Exception as e:
            logger.error(f"Failed to embed query for caching: {e}")
            return

        entry = CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            metadata=metadata or {},
            created_at=time.time(),
        )

        # Store in Redis or memory
        try:
            if self.redis:
                await self._redis_set(query_hash, entry)
                logger.info(f"Cached in Redis: query_hash={query_hash}")
            else:
                self._memory_set(query_hash, entry)
                logger.info(f"Cached in memory: query_hash={query_hash}")
        except Exception as e:
            logger.error(f"Failed to store cache entry: {e}")
    
    async def _exact_match(self, query: str) -> Optional[str]:
        """Layer 1: Exact hash match."""
        query_hash = self._hash_query(query)

        if self.redis:
            data = self.redis.get(f"{self.prefix}:exact:{query_hash}")
            if data:
                entry = json.loads(data)
                response = entry.get("response", "")
                # Skip empty cached responses
                if response and response.strip():
                    return response
                else:
                    # Delete corrupted cache entry
                    self.redis.delete(f"{self.prefix}:exact:{query_hash}")
                    logger.warning(f"Deleted corrupted cache entry (empty response)")
        else:
            if query_hash in self._memory_cache:
                response = self._memory_cache[query_hash].response
                # Skip empty cached responses
                if response and response.strip():
                    return response
                else:
                    # Delete corrupted cache entry
                    del self._memory_cache[query_hash]
                    self._embeddings = [(k, e) for k, e in self._embeddings if k != query_hash]
                    logger.warning(f"Deleted corrupted cache entry (empty response)")

        return None
    
    async def _semantic_match(
        self,
        query: str,
        query_embedding: np.ndarray,
    ) -> Optional[dict]:
        """Layer 2 + 3: Semantic match with cross-encoder validation."""
        
        # Find similar cached queries
        candidates = self._find_similar(query_embedding)
        
        if not candidates:
            return None
        
        best = candidates[0]
        
        # Check similarity threshold
        if best["similarity"] < self.similarity_threshold:
            return None
        
        # Layer 3: Cross-encoder validation (if available)
        if self.reranker:
            is_valid = await self._validate_with_reranker(query, best["query"])
            if not is_valid:
                logger.debug(f"Cross-encoder rejected cache hit: {best['similarity']:.3f}")
                return None
        
        return best
    
    def _find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """Find similar queries using cosine similarity."""
        if not self._embeddings:
            return []

        # Compute similarities
        similarities = []
        for key, emb in self._embeddings:
            sim = self._cosine_similarity(query_embedding, emb)
            similarities.append((key, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top candidates (skip empty responses)
        results = []
        for key, sim in similarities[:top_k * 2]:  # Check more to account for skipped
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                # Skip empty responses
                if entry.response and entry.response.strip():
                    results.append({
                        "query": entry.query,
                        "response": entry.response,
                        "similarity": sim,
                    })
                    if len(results) >= top_k:
                        break

        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity using numpy."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    async def _validate_with_reranker(self, query: str, cached_query: str) -> bool:
        """Layer 3: Cross-encoder validation to filter false positives."""
        try:
            # Use cross-encoder to score query pair
            score = self.reranker.predict([(query, cached_query)])[0]
            return score >= self.rerank_threshold
        except Exception as e:
            logger.warning(f"Reranker validation failed: {e}")
            return True  # Fail open
    
    def _hash_query(self, query: str) -> str:
        """Hash query for exact match lookup."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _memory_set(self, key: str, entry: CacheEntry) -> None:
        """Store in memory cache."""
        # Evict if at capacity
        if len(self._memory_cache) >= self.max_size:
            self._evict_oldest()
        
        self._memory_cache[key] = entry
        self._embeddings.append((key, np.array(entry.embedding)))
    
    async def _redis_set(self, key: str, entry: CacheEntry) -> None:
        """Store in Redis."""
        data = {
            "query": entry.query,
            "response": entry.response,
            "embedding": entry.embedding,
            "metadata": entry.metadata,
            "created_at": entry.created_at,
        }
        
        # Store exact match key
        self.redis.setex(
            f"{self.prefix}:exact:{key}",
            self.ttl,
            json.dumps(data),
        )
        
        # Also store in memory for semantic search
        self._memory_set(key, entry)
    
    def _evict_oldest(self) -> None:
        """Evict oldest entries when at capacity."""
        if not self._memory_cache:
            return
        
        # Find oldest
        oldest_key = min(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k].created_at,
        )
        
        del self._memory_cache[oldest_key]
        self._embeddings = [(k, e) for k, e in self._embeddings if k != oldest_key]
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = sum(self._stats.values())
        hit_rate = (self._stats["hits_exact"] + self._stats["hits_semantic"]) / total if total > 0 else 0
        
        return {
            "hits_exact": self._stats["hits_exact"],
            "hits_semantic": self._stats["hits_semantic"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "cache_size": len(self._memory_cache),
            "threshold": self.similarity_threshold,
        }
    
    def clear(self) -> None:
        """Clear the cache."""
        self._memory_cache.clear()
        self._embeddings.clear()
        self._stats = {"hits_exact": 0, "hits_semantic": 0, "misses": 0}
        
        if self.redis:
            # Clear Redis keys with prefix
            keys = self.redis.keys(f"{self.prefix}:*")
            if keys:
                self.redis.delete(*keys)
    
    def adjust_threshold(self, new_threshold: float) -> None:
        """
        Adjust similarity threshold.
        
        Start conservative (0.95), tune down if hit rate too low.
        Tune up if seeing false positives.
        """
        old = self.similarity_threshold
        self.similarity_threshold = max(0.8, min(0.99, new_threshold))
        logger.info(f"Adjusted threshold: {old:.2f} → {self.similarity_threshold:.2f}")