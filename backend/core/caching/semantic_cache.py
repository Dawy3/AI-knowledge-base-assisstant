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
        similarity_threshold: float = 0.95,     # Start conservative
        rerank_threshold: float= 0.7,
        ttl_seconds: int = 3600,
        max_cache_size : int = 10000,
        prefix: str = "sem_cache"
     ):
        """
        Args:
            embed_func: Function to embed queries
            redis_client: Redis client (optional, uses in-memory if None)
            reranker: Cross-encoder reranker for Layer 3 validation
            similarity_threshold: Threshold for semantic match (start at 0.95)
            rerank_threshold: Threshold for cross-encoder validation
            ttl_seconds: Cache TTL
            max_cache_size: Max entries in cache
            prefix: Redis key prefix
        """
        self.embed_func = embed_func
        self.redis = redis_client
        self.reranker = reranker
        self.similarity_threshold = similarity_threshold
        self.rerank_threshold = rerank_threshold
        self.ttl = ttl_seconds
        self.max_size = max_cache_size
        self.prefix = prefix
        
        # In-memory fallback
        self._memory_cache: dict[str, CacheEntry] = {}
        self._embeddings: list[tuple[str, np.ndarray]] = []  # (key, embedding)
        
        # Stats
        self._stats = {"hits_exact" : 0, "hits_semantic": 0, "misses": 0}
        
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
            self._stats["hits_exact"] +=1
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
                hit = True,
                response=semantic_result["resposne"],
                layer= "semantic",
                similarity=semantic_result["similarity"],
                latency_ms=(time.perf_counter() - start) * 1000,
            )
            
        self._stats["misses"] += 1
        return CacheResult(
            hit=False,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        
    