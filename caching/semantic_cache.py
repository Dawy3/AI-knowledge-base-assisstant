"""
Semantic Caching:
- Cache similar queries and their results
- Use embedding similarity for cache lookup
- Reduce LLM API calls and retrieval overhead
- TTL-based expiration
"""

import json
import hashlib
from typing import Optional, Dict, Any
import numpy as np
import redis
from datetime import datetime, timedelta
import logging

from core.embeddings import EmbeddingManager
from config.settings import settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Redis-based semantic cache with embedding similarity
    
    How it works:
    1. Query comes in -> generate embedding
    2. Search cache for similar queries (cosine similarity)
    3. If similarity > threshold, return cached result
    4. Otherwise, execute query and cache result
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        embedding_manager: EmbeddingManager,
        similarity_threshold: float = settings.CACHE_SIMILARITY_THRESHOLD,
        ttl: int = settings.CACHE_TTL
    ):
        self.redis = redis_client
        self.embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        
        # Redis key prefixes
        self.EMBEDDING_PREFIX = "cache:embedding:"
        self.RESULT_PREFIX = "cache:result:"
        self.INDEX_KEY = "cache:index"  # Sorted set for quick lookup
        
    def _generate_cache_key(self, query: str) -> str:
        """Generate unique cache key for query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result for query if similar query exists
        
        Args:
            query: User query
            
        Returns:
            Cached result dict or None
        """
        if not settings.ENABLE_SEMANTIC_CACHE:
            return None
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_query(query)
            
            # Get all cached query embeddings
            # In production, you might want to use a vector database for this
            # For now, we'll iterate through cached queries
            cached_queries = self.redis.smembers(self.INDEX_KEY)
            
            if not cached_queries:
                logger.debug("Cache empty")
                return None
            
            best_match = None
            best_similarity = 0.0
            
            for cached_query_bytes in cached_queries:
                cached_query = cached_query_bytes.decode('utf-8')
                cache_key = self._generate_cache_key(cached_query)
                
                # Get cached embedding
                embedding_key = f"{self.EMBEDDING_PREFIX}{cache_key}"
                cached_embedding_json = self.redis.get(embedding_key)
                
                if not cached_embedding_json:
                    # Clean up stale index entry
                    self.redis.srem(self.INDEX_KEY, cached_query)
                    continue
                
                cached_embedding = np.array(json.loads(cached_embedding_json))
                
                # Calculate similarity
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cache_key
            
            # Check if best match exceeds threshold
            if best_similarity >= self.similarity_threshold:
                result_key = f"{self.RESULT_PREFIX}{best_match}"
                cached_result_json = self.redis.get(result_key)
                
                if cached_result_json:
                    cached_result = json.loads(cached_result_json)
                    cached_result["cache_hit"] = True
                    cached_result["cache_similarity"] = best_similarity
                    
                    logger.info(
                        f"Cache HIT: similarity={best_similarity:.3f}, "
                        f"query='{query}'"
                    )
                    
                    return cached_result
            
            logger.debug(f"Cache MISS: best_similarity={best_similarity:.3f}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, query: str, result: Dict[str, Any]) -> bool:
        """
        Cache query result
        
        Args:
            query: User query
            result: Result to cache
            
        Returns:
            Success status
        """
        if not settings.ENABLE_SEMANTIC_CACHE:
            return False
        
        try:
            cache_key = self._generate_cache_key(query)
            
            # Generate and cache embedding
            query_embedding = self.embedding_manager.embed_query(query)
            embedding_key = f"{self.EMBEDDING_PREFIX}{cache_key}"
            
            # Serialize embedding
            embedding_json = json.dumps(query_embedding.tolist())
            self.redis.setex(embedding_key, self.ttl, embedding_json)
            
            # Cache result
            result_key = f"{self.RESULT_PREFIX}{cache_key}"
            
            # Add metadata
            cached_data = {
                "query": query,
                "result": result,
                "cached_at": datetime.now().isoformat(),
                "ttl": self.ttl
            }
            
            result_json = json.dumps(cached_data)
            self.redis.setex(result_key, self.ttl, result_json)
            
            # Add to index
            self.redis.sadd(self.INDEX_KEY, query)
            
            logger.info(f"Cached query: '{query}'")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def invalidate(self, query: str) -> bool:
        """
        Invalidate cached query
        
        Args:
            query: Query to invalidate
            
        Returns:
            Success status
        """
        try:
            cache_key = self._generate_cache_key(query)
            
            # Delete embedding
            embedding_key = f"{self.EMBEDDING_PREFIX}{cache_key}"
            self.redis.delete(embedding_key)
            
            # Delete result
            result_key = f"{self.RESULT_PREFIX}{cache_key}"
            self.redis.delete(result_key)
            
            # Remove from index
            self.redis.srem(self.INDEX_KEY, query)
            
            logger.info(f"Invalidated cache for: '{query}'")
            return True
            
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear entire cache
        
        Returns:
            Success status
        """
        try:
            # Get all cached queries
            cached_queries = self.redis.smembers(self.INDEX_KEY)
            
            for cached_query_bytes in cached_queries:
                cached_query = cached_query_bytes.decode('utf-8')
                self.invalidate(cached_query)
            
            logger.info("Cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Cache stats dict
        """
        try:
            cached_queries = self.redis.smembers(self.INDEX_KEY)
            
            return {
                "total_cached_queries": len(cached_queries),
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl,
                "enabled": settings.ENABLE_SEMANTIC_CACHE
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "error": str(e)
            }
    
    def warmup(self, common_queries: list[str], results: list[Dict]) -> int:
        """
        Pre-populate cache with common queries
        
        Args:
            common_queries: List of common queries
            results: Corresponding results
            
        Returns:
            Number of queries cached
        """
        count = 0
        
        for query, result in zip(common_queries, results):
            if self.set(query, result):
                count += 1
        
        logger.info(f"Cache warmed up with {count} queries")
        return count