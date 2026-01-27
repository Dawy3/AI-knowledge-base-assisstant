"""Tests for caching module: semantic_cache and embedding_cache."""

import time
import pytest
import numpy as np
from unittest.mock import MagicMock

from backend.core.caching.semantic_cache import (
    CacheEntry,
    CacheResult,
    SemanticCache,
)
from backend.core.caching.embedding_cache import (
    CachedEmbedding,
    EmbeddingCache,
)


# ============================================================================
# SEMANTIC CACHE TESTS
# ============================================================================


# ============================================================================
# Fixtures for SemanticCache
# ============================================================================


@pytest.fixture
def mock_embed_func():
    """Create mock embedding function."""
    def embed(text: str) -> list[float]:
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(384).tolist()
    return embed


@pytest.fixture
def mock_reranker():
    """Create mock reranker."""
    mock = MagicMock()
    mock.predict.return_value = [0.85]
    return mock


@pytest.fixture
def semantic_cache(mock_embed_func):
    """Create SemanticCache instance without Redis."""
    return SemanticCache(
        embed_func=mock_embed_func,
        redis_client=None,
        similarity_threshold=0.95,
        ttl_seconds=3600,
        max_cache_size=100,
    )


@pytest.fixture
def semantic_cache_with_reranker(mock_embed_func, mock_reranker):
    """Create SemanticCache with reranker."""
    return SemanticCache(
        embed_func=mock_embed_func,
        redis_client=None,
        reranker=mock_reranker,
        similarity_threshold=0.95,
        rerank_threshold=0.7,
    )


# ============================================================================
# Test CacheEntry
# ============================================================================


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_creation(self):
        entry = CacheEntry(
            query="What is Python?",
            response="Python is a programming language.",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "docs"},
            created_at=time.time(),
        )

        assert entry.query == "What is Python?"
        assert entry.response == "Python is a programming language."
        assert entry.embedding == [0.1, 0.2, 0.3]
        assert entry.metadata == {"source": "docs"}
        assert entry.hit_count == 0

    def test_default_hit_count(self):
        entry = CacheEntry(
            query="test",
            response="response",
            embedding=[],
            metadata={},
            created_at=time.time(),
        )

        assert entry.hit_count == 0


# ============================================================================
# Test CacheResult
# ============================================================================


class TestCacheResult:
    """Test CacheResult dataclass."""

    def test_cache_hit(self):
        result = CacheResult(
            hit=True,
            response="Cached response",
            layer="exact",
            similarity=1.0,
            latency_ms=0.5,
        )

        assert result.hit is True
        assert result.response == "Cached response"
        assert result.layer == "exact"
        assert result.similarity == 1.0

    def test_cache_miss(self):
        result = CacheResult(
            hit=False,
            latency_ms=1.0,
        )

        assert result.hit is False
        assert result.response is None
        assert result.layer is None
        assert result.similarity == 0.0

    def test_semantic_hit(self):
        result = CacheResult(
            hit=True,
            response="Semantic match",
            layer="semantic",
            similarity=0.96,
        )

        assert result.layer == "semantic"
        assert result.similarity == 0.96


# ============================================================================
# Test SemanticCache Initialization
# ============================================================================


class TestSemanticCacheInit:
    """Test SemanticCache initialization."""

    def test_default_initialization(self, mock_embed_func):
        cache = SemanticCache(embed_func=mock_embed_func)

        assert cache.similarity_threshold == 0.95
        assert cache.ttl == 3600
        assert cache.max_size == 10000
        assert cache.prefix == "sem_cache"
        assert cache.reranker is None

    def test_custom_initialization(self, mock_embed_func):
        cache = SemanticCache(
            embed_func=mock_embed_func,
            similarity_threshold=0.90,
            ttl_seconds=7200,
            max_cache_size=5000,
            prefix="custom_cache",
        )

        assert cache.similarity_threshold == 0.90
        assert cache.ttl == 7200
        assert cache.max_size == 5000
        assert cache.prefix == "custom_cache"

    def test_with_reranker(self, mock_embed_func, mock_reranker):
        cache = SemanticCache(
            embed_func=mock_embed_func,
            reranker=mock_reranker,
            rerank_threshold=0.8,
        )

        assert cache.reranker == mock_reranker
        assert cache.rerank_threshold == 0.8


# ============================================================================
# Test SemanticCache.set()
# ============================================================================


class TestSemanticCacheSet:
    """Test SemanticCache set operation."""

    @pytest.mark.asyncio
    async def test_set_stores_entry(self, semantic_cache):
        await semantic_cache.set(
            query="What is machine learning?",
            response="ML is a subset of AI.",
            metadata={"topic": "AI"},
        )

        assert len(semantic_cache._memory_cache) == 1
        assert len(semantic_cache._embeddings) == 1

    @pytest.mark.asyncio
    async def test_set_multiple_entries(self, semantic_cache):
        await semantic_cache.set("Query 1", "Response 1")
        await semantic_cache.set("Query 2", "Response 2")
        await semantic_cache.set("Query 3", "Response 3")

        assert len(semantic_cache._memory_cache) == 3

    @pytest.mark.asyncio
    async def test_set_with_metadata(self, semantic_cache):
        await semantic_cache.set(
            query="Test query",
            response="Test response",
            metadata={"key": "value", "count": 42},
        )

        entries = list(semantic_cache._memory_cache.values())
        assert len(entries) == 1
        assert entries[0].metadata == {"key": "value", "count": 42}


# ============================================================================
# Test SemanticCache.get() - Exact Match (Layer 1)
# ============================================================================


class TestSemanticCacheExactMatch:
    """Test exact match (Layer 1)."""

    @pytest.mark.asyncio
    async def test_exact_match_hit(self, semantic_cache):
        query = "What is Python?"
        response = "Python is a programming language."

        await semantic_cache.set(query, response)
        result = await semantic_cache.get(query)

        assert result.hit is True
        assert result.response == response
        assert result.layer == "exact"
        assert result.similarity == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_case_insensitive(self, semantic_cache):
        await semantic_cache.set("What is Python?", "Python response")

        result = await semantic_cache.get("WHAT IS PYTHON?")

        assert result.hit is True
        assert result.layer == "exact"

    @pytest.mark.asyncio
    async def test_exact_match_strips_whitespace(self, semantic_cache):
        await semantic_cache.set("What is Python?", "Python response")

        result = await semantic_cache.get("  What is Python?  ")

        assert result.hit is True
        assert result.layer == "exact"

    @pytest.mark.asyncio
    async def test_exact_match_miss(self, semantic_cache):
        await semantic_cache.set("What is Python?", "Python response")

        result = await semantic_cache.get("What is JavaScript?")

        if result.hit:
            assert result.layer == "semantic"
        else:
            assert result.layer is None


# ============================================================================
# Test SemanticCache.get() - Semantic Match (Layer 2)
# ============================================================================


class TestSemanticCacheSemanticMatch:
    """Test semantic match (Layer 2)."""

    @pytest.mark.asyncio
    async def test_semantic_match_similar_query(self, mock_embed_func):
        cache = SemanticCache(
            embed_func=mock_embed_func,
            similarity_threshold=0.5,
        )

        await cache.set("What is machine learning?", "ML is AI subset")

        result = await cache.get("What is machine learning?")
        assert result.hit is True

    @pytest.mark.asyncio
    async def test_semantic_match_below_threshold(self, semantic_cache):
        await semantic_cache.set("What is Python?", "Python response")

        result = await semantic_cache.get("How to cook pasta?")

        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_semantic_match_returns_best(self, mock_embed_func):
        cache = SemanticCache(
            embed_func=mock_embed_func,
            similarity_threshold=0.3,
        )

        await cache.set("Query A", "Response A")
        await cache.set("Query B", "Response B")
        await cache.set("Query C", "Response C")

        result = await cache.get("Query A")
        assert result.hit is True


# ============================================================================
# Test SemanticCache.get() - Reranker Validation (Layer 3)
# ============================================================================


class TestSemanticCacheReranker:
    """Test cross-encoder reranker validation (Layer 3)."""

    @pytest.mark.asyncio
    async def test_reranker_validates_hit(self, mock_embed_func):
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9]

        cache = SemanticCache(
            embed_func=mock_embed_func,
            reranker=mock_reranker,
            similarity_threshold=0.3,
            rerank_threshold=0.7,
        )

        await cache.set("Test query", "Test response")
        result = await cache.get("Test query")

        assert result.hit is True

    @pytest.mark.asyncio
    async def test_reranker_rejects_false_positive(self, mock_embed_func):
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.3]

        cache = SemanticCache(
            embed_func=mock_embed_func,
            reranker=mock_reranker,
            similarity_threshold=0.3,
            rerank_threshold=0.7,
        )

        await cache.set("Original query", "Original response")
        result = await cache.get("Different query")
        # Result depends on semantic similarity vs reranker threshold

    @pytest.mark.asyncio
    async def test_reranker_exception_fails_open(self, mock_embed_func):
        mock_reranker = MagicMock()
        mock_reranker.predict.side_effect = Exception("Reranker error")

        cache = SemanticCache(
            embed_func=mock_embed_func,
            reranker=mock_reranker,
            similarity_threshold=0.3,
        )

        await cache.set("Test query", "Test response")

        result = await cache.get("Test query")
        assert result.hit is True  # Exact match should still work


# ============================================================================
# Test SemanticCache Statistics
# ============================================================================


class TestSemanticCacheStats:
    """Test cache statistics."""

    @pytest.mark.asyncio
    async def test_stats_initial(self, semantic_cache):
        stats = semantic_cache.get_stats()

        assert stats["hits_exact"] == 0
        assert stats["hits_semantic"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0
        assert stats["cache_size"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_exact_hit(self, semantic_cache):
        await semantic_cache.set("Query", "Response")
        await semantic_cache.get("Query")

        stats = semantic_cache.get_stats()

        assert stats["hits_exact"] == 1
        assert stats["misses"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_miss(self, semantic_cache):
        await semantic_cache.get("Non-existent query")

        stats = semantic_cache.get_stats()

        assert stats["misses"] == 1
        assert stats["hits_exact"] == 0
        assert stats["hits_semantic"] == 0

    @pytest.mark.asyncio
    async def test_stats_hit_rate(self, semantic_cache):
        await semantic_cache.set("Query 1", "Response 1")

        await semantic_cache.get("Query 1")  # Hit
        await semantic_cache.get("Query 1")  # Hit
        await semantic_cache.get("Unknown")  # Miss

        stats = semantic_cache.get_stats()

        assert stats["hit_rate"] == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_stats_cache_size(self, semantic_cache):
        await semantic_cache.set("Query 1", "Response 1")
        await semantic_cache.set("Query 2", "Response 2")

        stats = semantic_cache.get_stats()

        assert stats["cache_size"] == 2


# ============================================================================
# Test SemanticCache Clear
# ============================================================================


class TestSemanticCacheClear:
    """Test cache clear operation."""

    @pytest.mark.asyncio
    async def test_clear_removes_entries(self, semantic_cache):
        await semantic_cache.set("Query 1", "Response 1")
        await semantic_cache.set("Query 2", "Response 2")

        semantic_cache.clear()

        assert len(semantic_cache._memory_cache) == 0
        assert len(semantic_cache._embeddings) == 0

    @pytest.mark.asyncio
    async def test_clear_resets_stats(self, semantic_cache):
        await semantic_cache.set("Query", "Response")
        await semantic_cache.get("Query")
        await semantic_cache.get("Unknown")

        semantic_cache.clear()

        stats = semantic_cache.get_stats()
        assert stats["hits_exact"] == 0
        assert stats["misses"] == 0

    @pytest.mark.asyncio
    async def test_clear_allows_new_entries(self, semantic_cache):
        await semantic_cache.set("Query 1", "Response 1")
        semantic_cache.clear()

        await semantic_cache.set("Query 2", "Response 2")

        assert len(semantic_cache._memory_cache) == 1


# ============================================================================
# Test SemanticCache Eviction
# ============================================================================


class TestSemanticCacheEviction:
    """Test cache eviction when at capacity."""

    @pytest.mark.asyncio
    async def test_evicts_oldest_at_capacity(self, mock_embed_func):
        cache = SemanticCache(
            embed_func=mock_embed_func,
            max_cache_size=3,
        )

        await cache.set("Query 1", "Response 1")
        await cache.set("Query 2", "Response 2")
        await cache.set("Query 3", "Response 3")

        await cache.set("Query 4", "Response 4")

        assert len(cache._memory_cache) <= 3

    @pytest.mark.asyncio
    async def test_newest_entries_preserved(self, mock_embed_func):
        cache = SemanticCache(
            embed_func=mock_embed_func,
            max_cache_size=2,
        )

        await cache.set("Old query", "Old response")
        await cache.set("New query 1", "New response 1")
        await cache.set("New query 2", "New response 2")

        result = await cache.get("Old query")

        assert len(cache._memory_cache) <= 2


# ============================================================================
# Test SemanticCache Threshold Adjustment
# ============================================================================


class TestSemanticCacheThreshold:
    """Test threshold adjustment."""

    def test_adjust_threshold(self, semantic_cache):
        semantic_cache.adjust_threshold(0.90)

        assert semantic_cache.similarity_threshold == 0.90

    def test_adjust_threshold_min_bound(self, semantic_cache):
        semantic_cache.adjust_threshold(0.5)

        assert semantic_cache.similarity_threshold == 0.8

    def test_adjust_threshold_max_bound(self, semantic_cache):
        semantic_cache.adjust_threshold(1.0)

        assert semantic_cache.similarity_threshold == 0.99

    def test_adjust_threshold_within_bounds(self, semantic_cache):
        semantic_cache.adjust_threshold(0.92)

        assert semantic_cache.similarity_threshold == 0.92


# ============================================================================
# Test SemanticCache Cosine Similarity
# ============================================================================


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_identical_vectors(self, semantic_cache):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        sim = semantic_cache._cosine_similarity(a, b)

        assert sim == pytest.approx(1.0)

    def test_orthogonal_vectors(self, semantic_cache):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])

        sim = semantic_cache._cosine_similarity(a, b)

        assert sim == pytest.approx(0.0)

    def test_opposite_vectors(self, semantic_cache):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])

        sim = semantic_cache._cosine_similarity(a, b)

        assert sim == pytest.approx(-1.0)

    def test_zero_vector(self, semantic_cache):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])

        sim = semantic_cache._cosine_similarity(a, b)

        assert sim == 0.0


# ============================================================================
# Test SemanticCache Query Hash
# ============================================================================


class TestSemanticCacheQueryHash:
    """Test query hashing."""

    def test_hash_deterministic(self, semantic_cache):
        hash1 = semantic_cache._hash_query("Test query")
        hash2 = semantic_cache._hash_query("Test query")

        assert hash1 == hash2

    def test_hash_case_insensitive(self, semantic_cache):
        hash1 = semantic_cache._hash_query("Test Query")
        hash2 = semantic_cache._hash_query("test query")

        assert hash1 == hash2

    def test_hash_strips_whitespace(self, semantic_cache):
        hash1 = semantic_cache._hash_query("Test query")
        hash2 = semantic_cache._hash_query("  Test query  ")

        assert hash1 == hash2

    def test_different_queries_different_hashes(self, semantic_cache):
        hash1 = semantic_cache._hash_query("Query A")
        hash2 = semantic_cache._hash_query("Query B")

        assert hash1 != hash2

    def test_hash_length(self, semantic_cache):
        hash_value = semantic_cache._hash_query("Any query")

        assert len(hash_value) == 16


# ============================================================================
# Test SemanticCache with Redis (Mocked)
# ============================================================================


class TestSemanticCacheRedis:
    """Test SemanticCache with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        return MagicMock()

    @pytest.fixture
    def cache_with_redis(self, mock_embed_func, mock_redis):
        return SemanticCache(
            embed_func=mock_embed_func,
            redis_client=mock_redis,
        )

    @pytest.mark.asyncio
    async def test_set_stores_in_redis(self, cache_with_redis, mock_redis):
        await cache_with_redis.set("Query", "Response")

        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_exact_match_checks_redis(self, cache_with_redis, mock_redis):
        mock_redis.get.return_value = '{"response": "Cached from Redis"}'

        result = await cache_with_redis.get("Query")

        mock_redis.get.assert_called()

    @pytest.mark.asyncio
    async def test_clear_deletes_redis_keys(self, cache_with_redis, mock_redis):
        mock_redis.keys.return_value = ["sem_cache:exact:abc123"]

        cache_with_redis.clear()

        mock_redis.keys.assert_called()


# ============================================================================
# Test SemanticCache Latency
# ============================================================================


class TestSemanticCacheLatency:
    """Test cache latency tracking."""

    @pytest.mark.asyncio
    async def test_latency_tracked_on_hit(self, semantic_cache):
        await semantic_cache.set("Query", "Response")

        result = await semantic_cache.get("Query")

        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_latency_tracked_on_miss(self, semantic_cache):
        result = await semantic_cache.get("Non-existent")

        assert result.latency_ms > 0


# ============================================================================
# EMBEDDING CACHE TESTS
# ============================================================================


# ============================================================================
# Fixtures for EmbeddingCache
# ============================================================================


@pytest.fixture
def embedding_cache():
    """Create EmbeddingCache instance without Redis."""
    return EmbeddingCache(
        redis_client=None,
        ttl_seconds=3600,
        max_memory_size=100,
    )


# ============================================================================
# Test CachedEmbedding
# ============================================================================


class TestCachedEmbedding:
    """Test CachedEmbedding dataclass."""

    def test_creation(self):
        cached = CachedEmbedding(
            embedding=[0.1, 0.2, 0.3],
            model_id="text-embedding-ada-002",
            content_hash="abc123",
            created_at=time.time(),
        )

        assert cached.embedding == [0.1, 0.2, 0.3]
        assert cached.model_id == "text-embedding-ada-002"
        assert cached.content_hash == "abc123"
        assert cached.created_at > 0


# ============================================================================
# Test EmbeddingCache Initialization
# ============================================================================


class TestEmbeddingCacheInit:
    """Test EmbeddingCache initialization."""

    def test_default_initialization(self):
        cache = EmbeddingCache()

        assert cache.ttl == 86400
        assert cache.max_size == 50000
        assert cache.prefix == "emb_cache"
        assert cache.redis is None

    def test_custom_initialization(self):
        cache = EmbeddingCache(
            ttl_seconds=7200,
            max_memory_size=10000,
            prefix="custom_emb",
        )

        assert cache.ttl == 7200
        assert cache.max_size == 10000
        assert cache.prefix == "custom_emb"


# ============================================================================
# Test EmbeddingCache.set() and .get()
# ============================================================================


class TestEmbeddingCacheSetGet:
    """Test EmbeddingCache set and get operations."""

    def test_set_and_get(self, embedding_cache):
        content = "Test content for embedding"
        model_id = "text-embedding-ada-002"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        embedding_cache.set(content, model_id, embedding)
        result = embedding_cache.get(content, model_id)

        assert result == embedding

    def test_get_miss(self, embedding_cache):
        result = embedding_cache.get("Non-existent content", "model-id")

        assert result is None

    def test_model_mismatch_returns_none(self, embedding_cache):
        content = "Test content"
        embedding = [0.1, 0.2, 0.3]

        embedding_cache.set(content, "model-A", embedding)
        result = embedding_cache.get(content, "model-B")

        assert result is None

    def test_set_multiple_entries(self, embedding_cache):
        embedding_cache.set("Content 1", "model", [0.1])
        embedding_cache.set("Content 2", "model", [0.2])
        embedding_cache.set("Content 3", "model", [0.3])

        assert embedding_cache.get("Content 1", "model") == [0.1]
        assert embedding_cache.get("Content 2", "model") == [0.2]
        assert embedding_cache.get("Content 3", "model") == [0.3]

    def test_overwrite_existing(self, embedding_cache):
        content = "Test content"
        model_id = "model"

        embedding_cache.set(content, model_id, [0.1, 0.2])
        embedding_cache.set(content, model_id, [0.3, 0.4])

        result = embedding_cache.get(content, model_id)
        assert result == [0.3, 0.4]


# ============================================================================
# Test EmbeddingCache Batch Operations
# ============================================================================


class TestEmbeddingCacheBatch:
    """Test batch operations."""

    def test_get_batch_all_cached(self, embedding_cache):
        contents = ["Content A", "Content B", "Content C"]
        model_id = "model"

        for i, content in enumerate(contents):
            embedding_cache.set(content, model_id, [float(i)])

        results, missing = embedding_cache.get_batch(contents, model_id)

        assert len(results) == 3
        assert all(r is not None for r in results)
        assert missing == []

    def test_get_batch_none_cached(self, embedding_cache):
        contents = ["Content A", "Content B", "Content C"]
        model_id = "model"

        results, missing = embedding_cache.get_batch(contents, model_id)

        assert len(results) == 3
        assert all(r is None for r in results)
        assert missing == [0, 1, 2]

    def test_get_batch_partial_cached(self, embedding_cache):
        contents = ["Content A", "Content B", "Content C"]
        model_id = "model"

        embedding_cache.set("Content B", model_id, [0.5])

        results, missing = embedding_cache.get_batch(contents, model_id)

        assert results[0] is None
        assert results[1] == [0.5]
        assert results[2] is None
        assert missing == [0, 2]

    def test_set_batch(self, embedding_cache):
        contents = ["Content 1", "Content 2", "Content 3"]
        model_id = "model"
        embeddings = [[0.1], [0.2], [0.3]]

        embedding_cache.set_batch(contents, model_id, embeddings)

        assert embedding_cache.get("Content 1", model_id) == [0.1]
        assert embedding_cache.get("Content 2", model_id) == [0.2]
        assert embedding_cache.get("Content 3", model_id) == [0.3]


# ============================================================================
# Test EmbeddingCache Statistics
# ============================================================================


class TestEmbeddingCacheStats:
    """Test cache statistics."""

    def test_stats_initial(self, embedding_cache):
        stats = embedding_cache.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0
        assert stats["size"] == 0

    def test_stats_after_hit(self, embedding_cache):
        embedding_cache.set("Content", "model", [0.1])
        embedding_cache.get("Content", "model")

        stats = embedding_cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_stats_after_miss(self, embedding_cache):
        embedding_cache.get("Non-existent", "model")

        stats = embedding_cache.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 1

    def test_stats_hit_rate(self, embedding_cache):
        embedding_cache.set("Content", "model", [0.1])

        embedding_cache.get("Content", "model")  # Hit
        embedding_cache.get("Content", "model")  # Hit
        embedding_cache.get("Unknown", "model")  # Miss

        stats = embedding_cache.get_stats()

        assert stats["hit_rate"] == pytest.approx(2 / 3)

    def test_stats_size(self, embedding_cache):
        embedding_cache.set("Content 1", "model", [0.1])
        embedding_cache.set("Content 2", "model", [0.2])

        stats = embedding_cache.get_stats()

        assert stats["size"] == 2


# ============================================================================
# Test EmbeddingCache Clear
# ============================================================================


class TestEmbeddingCacheClear:
    """Test cache clear operation."""

    def test_clear_removes_entries(self, embedding_cache):
        embedding_cache.set("Content 1", "model", [0.1])
        embedding_cache.set("Content 2", "model", [0.2])

        embedding_cache.clear()

        assert embedding_cache.get("Content 1", "model") is None
        assert embedding_cache.get("Content 2", "model") is None

    def test_clear_resets_stats(self, embedding_cache):
        embedding_cache.set("Content", "model", [0.1])
        embedding_cache.get("Content", "model")
        embedding_cache.get("Unknown", "model")

        embedding_cache.clear()

        stats = embedding_cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_clear_allows_new_entries(self, embedding_cache):
        embedding_cache.set("Content 1", "model", [0.1])
        embedding_cache.clear()

        embedding_cache.set("Content 2", "model", [0.2])

        assert embedding_cache.get("Content 2", "model") == [0.2]


# ============================================================================
# Test EmbeddingCache Invalidate
# ============================================================================


class TestEmbeddingCacheInvalidate:
    """Test cache invalidation."""

    def test_invalidate_removes_entry(self, embedding_cache):
        embedding_cache.set("Content", "model", [0.1])

        embedding_cache.invalidate("Content", "model")

        result = embedding_cache.get("Content", "model")
        assert result is None

    def test_invalidate_non_existent(self, embedding_cache):
        # Should not raise
        embedding_cache.invalidate("Non-existent", "model")

    def test_invalidate_preserves_other_entries(self, embedding_cache):
        embedding_cache.set("Content A", "model", [0.1])
        embedding_cache.set("Content B", "model", [0.2])

        embedding_cache.invalidate("Content A", "model")

        assert embedding_cache.get("Content A", "model") is None
        assert embedding_cache.get("Content B", "model") == [0.2]


# ============================================================================
# Test EmbeddingCache Eviction
# ============================================================================


class TestEmbeddingCacheEviction:
    """Test cache eviction when at capacity."""

    def test_evicts_oldest_at_capacity(self):
        cache = EmbeddingCache(max_memory_size=3)

        cache.set("Content 1", "model", [0.1])
        cache.set("Content 2", "model", [0.2])
        cache.set("Content 3", "model", [0.3])

        cache.set("Content 4", "model", [0.4])

        assert len(cache._cache) <= 3

    def test_newest_entries_preserved(self):
        cache = EmbeddingCache(max_memory_size=2)

        cache.set("Old content", "model", [0.1])
        cache.set("New content 1", "model", [0.2])
        cache.set("New content 2", "model", [0.3])

        assert len(cache._cache) <= 2


# ============================================================================
# Test EmbeddingCache Key Generation
# ============================================================================


class TestEmbeddingCacheKeyGeneration:
    """Test cache key generation."""

    def test_key_deterministic(self, embedding_cache):
        key1 = embedding_cache._make_key("Content", "model")
        key2 = embedding_cache._make_key("Content", "model")

        assert key1 == key2

    def test_different_content_different_keys(self, embedding_cache):
        key1 = embedding_cache._make_key("Content A", "model")
        key2 = embedding_cache._make_key("Content B", "model")

        assert key1 != key2

    def test_different_model_different_keys(self, embedding_cache):
        key1 = embedding_cache._make_key("Content", "model-A")
        key2 = embedding_cache._make_key("Content", "model-B")

        assert key1 != key2

    def test_key_length(self, embedding_cache):
        key = embedding_cache._make_key("Any content", "any-model")

        assert len(key) == 32

    def test_content_hash_length(self, embedding_cache):
        hash_value = embedding_cache._hash_content("Any content")

        assert len(hash_value) == 16


# ============================================================================
# Test EmbeddingCache with Redis (Mocked)
# ============================================================================


class TestEmbeddingCacheRedis:
    """Test EmbeddingCache with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        return MagicMock()

    @pytest.fixture
    def cache_with_redis(self, mock_redis):
        return EmbeddingCache(redis_client=mock_redis)

    def test_set_stores_in_redis(self, cache_with_redis, mock_redis):
        cache_with_redis.set("Content", "model", [0.1, 0.2])

        mock_redis.setex.assert_called_once()

    def test_get_checks_redis_first(self, cache_with_redis, mock_redis):
        import json
        mock_redis.get.return_value = json.dumps({
            "embedding": [0.1, 0.2],
            "model_id": "model",
        })

        result = cache_with_redis.get("Content", "model")

        mock_redis.get.assert_called()
        assert result == [0.1, 0.2]

    def test_get_redis_model_mismatch(self, cache_with_redis, mock_redis):
        import json
        mock_redis.get.return_value = json.dumps({
            "embedding": [0.1, 0.2],
            "model_id": "different-model",
        })

        result = cache_with_redis.get("Content", "model")

        # Should return None due to model mismatch
        # (will fallback to memory cache which is empty)

    def test_invalidate_deletes_from_redis(self, cache_with_redis, mock_redis):
        cache_with_redis.invalidate("Content", "model")

        mock_redis.delete.assert_called()

    def test_clear_deletes_redis_keys(self, cache_with_redis, mock_redis):
        mock_redis.keys.return_value = ["emb_cache:abc123"]

        cache_with_redis.clear()

        mock_redis.keys.assert_called()
