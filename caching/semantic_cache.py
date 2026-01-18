"""
Semantic Caching:
- Cache similar queries and their results
- Use embedding similarity for cache lookup
- Reduce LLM API calls and retrieval overhead
- TTL- Based expiratin
"""

import json
import hashlib
import logging
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime

import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from core.embeddings import EmbeddingManager
from config.settings import settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Redis-based semantic cache using RediSearch Vector Similarity.
    Requires: Redis Stack (redis-stack-server)
    """
    
    
    def __init__(
        self,
        redis_client: redis.Redis,
        embedding_manager: EmbeddingManager,
        similarity_threshold: float = settings.CACHE_SIMILARITY_THRESHOL,
        ttl: int  = settings.CACHE_TTL,
        vector_dim: int = settings.DIMENSION
    ):
        self.redis = redis_client
        self.embedding_manger = embedding_manager
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.vector_dim = vector_dim
        
        # Configuration
        self.INDEX_NAME = "idx:semantic_cache"
        self.PREFIX = "cache:node:"
        
        # Initialize the vector index
        if settings.ENABLE_SEMANTIC_CACHE:
            self._ensure_index_exists()
        
    def _ensure_index_exists(self):
        """
        Creates the vector search Index in redis if it doesn't exist.
        """
        try:
            # check if index exists
            self.redis.ft(self.INDEX_NAME).info()
        except redis.exceptions.ResponseError:
            # Index does not exists, Create it
            logger.info(f"Creating vector index '{self.INDEX_NAME}'...")
            
            schema = (
                TextField("query"),             # Store original query text
                TextField("result_json"),       # Store the LLM response
                VectorField(
                    "embedding",                # The Vector column
                    "HNSW",                     # Algorithm (Hierarchical Navigable Small World)
                    {
                        "TYPE" : "FLOAT32",
                        "DIM" : self.vector_dim,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            )
            
            definition = IndexDefinition(
                prefix = [self.PREFIX],
                index_type= IndexType.HASH
            )
            
            try:
                self.redis.ft(self.INDEX_NAME).create_index(
                    fields=schema,
                    definition = definition
                )
                logger.info("Vector index created successfully.")
            except Exception as e:
                logger.error(f"Failed to create vector index: {e}")
                
        
    def _generate_cache_key(self, query: str) -> str:
        """Generate unique cache key for query"""
        # Create a deterministic key based on the query hash
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{self.PREFIX}{query_hash}"
        
    def get(self, query:str) -> Optional[Dict[str, Any]]:
        """
        Search for semantically similar queries using Redis Vector Search.
        """
        
        if not settings.ENABLE_SEMANTIC_CACHE:
            return None
        
        try:
            # 1. Generate query embedding 
            query_embedding = self.embedding_manger.embed_query(query)
            
            # 2. Prepare Vector Search Query
            # Redis 'COSINE' distance = 1 - cosine_similarity
            # If threshold is 0.9, we want distance < 0.1
            distance_threshold = 1 - self.similarity_threshold
            
            # Syntax: Return top 1 neighbor (KNN 1) where vector is $vec
            # We return 'score' which represents the distance
            q = Query(f"*=>[KNN 1 @embedding $vec AS score]").sort_by("score").return_field("result_json", "score", "query").dialect(2)
            params = {
                "vec" : np.array(query_embedding).astype(np.float32).tobytes()
            } 
            
            # 3. Execute Search
            results = self.redis.ft(self.INDEX_NAME).search(q, query_params= params)
            
            if not results.docs:
                logger.debug("Cache MISS (No results)")
                return None
            
            # 4. Cehck Threshold
            best_match = results.docs[0]
            score = float(best_match.score)  # This is the distance (0 to 1)
            
            if score <= distance_threshold:
                # Calculate similarity for logging (1 - distance)
                similarity = 1 - score
                logger.info(f"Cache HIT: similarity={similarity:.3f}, query='{query}'")
                
                result_data = json.loads(best_match.result_json)
                result_data["cache_hit"] = True
                result_data["cache_similarity"] = similarity
                return result_data
            
            logger.debug(f"Cache MISS: best match distance={score:.3f} (thresh: {distance_threshold:.3f})")
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
        
    def set(self, query:str, result: Dict[str, Any]) -> bool:
        """
        Cache query result using Redis Hash
        """
        if not settings.ENABLE_SEMANTIC_CACHE:
            return False
        
        try:
            # 1. Generate emedding 
            query_embedding = self.embedding_manger.embed_query(query)
            
            # 2. Prepare data
            key = self._generate_cache_key(query)
            
            # Redis vector search requires bytes for FLOAT32
            vector_bytes = np.array(query_embedding).astype(np.float32).tobytes()
            
            data = {
                "query" : query,
                "result_json": json.dumps(result),
                "embedding": vector_bytes,
                "created_at" : datetime.now().isoformat()
            }
            
            # 3. Store in Redis (HSET)
            # Pipeline ensures atomic execution of set + expire
            pipe = self.redis.pipeline()
            pipe.hset(key, mapping=data)
            pipe.expire(key, self.ttl)
            pipe.execute()
            
            logger.info(f"Cached query: '{query}'")
            return True
        
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
        
    def invalidate(self, query: str) -> bool:
        """
        Invalidate cached query
        """
        try:
            key = self._generate_cache_key(query)
            self.redis.delete(key)
            logger.info(f"Invalidated cache for: '{query}'")
            return True
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all cache entries
        """
        try:
            # We can't use FLUSHDB because we might share Redis with other services.
            # We search for keys matching our prefix.
            cursor = '0'
            pattern = f"{self.PREFIX}*"
            count = 0
            
            while cursor != 0:
                cursor, keys = self.redis.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    self.redis.delete(*keys)
                    count += len(keys)
            
            logger.info(f"Cache cleared ({count} entries removed)")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics from the Index
        """
        try:
            info = self.redis.ft(self.INDEX_NAME).info()
            return {
                "total_indexed_docs": info.get("num_docs"),
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl,
                "vector_dim": self.vector_dim,
                "enabled": settings.ENABLE_SEMANTIC_CACHE
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    def warmup(self, common_queries: list[str], results: list[Dict]) -> int:
        """
        Pre-populate cache
        """
        count = 0
        for query, result in zip(common_queries, results):
            if self.set(query, result):
                count += 1
        logger.info(f"Cache warmed up with {count} queries")
        return count
            
        
        
    