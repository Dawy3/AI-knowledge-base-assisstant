"""
Integration tests for end-to-end retrieval pipeline.

Tests the full flow: Query -> Embedding -> Vector Search -> BM25 -> Hybrid Fusion -> Rerank
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.retrieval.vector_search import (
    VectorSearch,
    InMemoryVectorStore,
    VectorSearchResult,
    HNSWConfig,
)
from backend.core.retrieval.bm25_search import BM25Search, BM25SearchResult
from backend.core.retrieval.hybrid_search import (
    HybridSearch,
    HybridSearchConfig,
    FusionMethod,
)
from backend.core.retrieval.reranker import Reranker, RerankerConfig, NoOpReranker


# Test data
SAMPLE_CHUNKS = [
    {"chunk_id": "c1", "content": "Python is a programming language.", "document_id": "d1", "chunk_index": 0},
    {"chunk_id": "c2", "content": "Machine learning uses algorithms to learn from data.", "document_id": "d1", "chunk_index": 1},
    {"chunk_id": "c3", "content": "Error code ERR-404 means resource not found.", "document_id": "d2", "chunk_index": 0},
    {"chunk_id": "c4", "content": "Deep learning is a subset of machine learning.", "document_id": "d2", "chunk_index": 1},
    {"chunk_id": "c5", "content": "Natural language processing handles text data.", "document_id": "d3", "chunk_index": 0},
]

SAMPLE_EMBEDDINGS = [
    [0.1] * 384,
    [0.2] * 384,
    [0.3] * 384,
    [0.4] * 384,
    [0.5] * 384,
]


class TestVectorSearchIntegration:
    """Test vector search component."""

    @pytest.fixture
    def vector_store(self):
        return InMemoryVectorStore(dimensions=384)

    @pytest.fixture
    def vector_search(self, vector_store):
        return VectorSearch(store=vector_store, default_top_k=10)

    @pytest.mark.asyncio
    async def test_index_and_search(self, vector_search):
        """Test indexing chunks and searching."""
        # Index
        count = await vector_search.index(SAMPLE_CHUNKS, SAMPLE_EMBEDDINGS)
        assert count == len(SAMPLE_CHUNKS)

        # Search
        query_embedding = [0.15] * 384
        response = await vector_search.search(query_embedding, top_k=3)

        assert len(response.results) == 3
        assert response.latency_ms >= 0
        assert all(isinstance(r, VectorSearchResult) for r in response.results)

    @pytest.mark.asyncio
    async def test_search_returns_scored_results(self, vector_search):
        """Test that search returns results with scores."""
        await vector_search.index(SAMPLE_CHUNKS, SAMPLE_EMBEDDINGS)

        query_embedding = [0.1] * 384
        response = await vector_search.search(query_embedding, top_k=5)

        # Scores should be sorted descending
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_with_filter(self, vector_search):
        """Test search with metadata filter."""
        await vector_search.index(SAMPLE_CHUNKS, SAMPLE_EMBEDDINGS)

        query_embedding = [0.2] * 384
        response = await vector_search.search(
            query_embedding,
            top_k=10,
            filter={"document_id": "d1"},
        )

        # All results should be from d1
        for result in response.results:
            assert result.document_id == "d1"


class TestBM25SearchIntegration:
    """Test BM25 search component."""

    @pytest.fixture
    def bm25_search(self):
        bm25 = BM25Search()
        bm25.index(SAMPLE_CHUNKS)
        return bm25

    def test_exact_match_search(self, bm25_search):
        """Test BM25 finds exact term matches."""
        response = bm25_search.search("ERR-404", top_k=5)

        assert len(response.results) > 0
        # Top result should contain the exact term
        assert "ERR-404" in response.results[0].content

    def test_keyword_search(self, bm25_search):
        """Test BM25 keyword matching."""
        response = bm25_search.search("machine learning", top_k=5)

        assert len(response.results) > 0
        # Results should be relevant to machine learning
        top_contents = [r.content.lower() for r in response.results[:2]]
        assert any("machine learning" in c for c in top_contents)

    def test_search_returns_metadata(self, bm25_search):
        """Test that BM25 results include metadata."""
        response = bm25_search.search("Python programming", top_k=3)

        for result in response.results:
            assert result.chunk_id
            assert result.content


class TestHybridSearchIntegration:
    """Test hybrid search combining vector and BM25."""

    @pytest.fixture
    def hybrid_search(self):
        # Setup vector search
        vector_store = InMemoryVectorStore(dimensions=384)
        vector_search = VectorSearch(store=vector_store, default_top_k=100)

        # Setup BM25
        bm25_search = BM25Search()

        # Create hybrid
        config = HybridSearchConfig(
            vector_weight=5.0,
            bm25_weight=3.0,
            fusion_method=FusionMethod.WEIGHTED_SUM,
        )
        return HybridSearch(vector_search, bm25_search, config), vector_search, bm25_search

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_results(self, hybrid_search):
        """Test hybrid search combines vector and BM25 results."""
        hybrid, vector_search, bm25_search = hybrid_search

        # Index data
        await vector_search.index(SAMPLE_CHUNKS, SAMPLE_EMBEDDINGS)
        bm25_search.index(SAMPLE_CHUNKS)

        # Search
        query = "machine learning algorithms"
        query_embedding = [0.25] * 384

        response = await hybrid.search(query, query_embedding, top_k=5)

        assert len(response.results) > 0
        assert response.total_vector_results > 0
        assert response.total_bm25_results > 0

    @pytest.mark.asyncio
    async def test_hybrid_search_overlap_tracking(self, hybrid_search):
        """Test hybrid search tracks overlap between methods."""
        hybrid, vector_search, bm25_search = hybrid_search

        await vector_search.index(SAMPLE_CHUNKS, SAMPLE_EMBEDDINGS)
        bm25_search.index(SAMPLE_CHUNKS)

        response = await hybrid.search("deep learning", [0.4] * 384, top_k=5)

        # Should track overlap
        assert response.unique_results >= response.overlap_count

    @pytest.mark.asyncio
    async def test_hybrid_search_score_fusion(self, hybrid_search):
        """Test that hybrid search properly fuses scores."""
        hybrid, vector_search, bm25_search = hybrid_search

        await vector_search.index(SAMPLE_CHUNKS, SAMPLE_EMBEDDINGS)
        bm25_search.index(SAMPLE_CHUNKS)

        response = await hybrid.search("Python", [0.1] * 384, top_k=5)

        for result in response.results:
            # Combined score should reflect both sources
            assert result.combined_score >= 0
            assert result.from_vector or result.from_bm25


class TestRerankerIntegration:
    """Test reranking component."""

    @pytest.fixture
    def reranker(self):
        # Use NoOp for testing to avoid loading models
        config = RerankerConfig(top_k_input=10, top_k_output=5)
        return Reranker(config=config, reranker=NoOpReranker())

    @pytest.mark.asyncio
    async def test_rerank_reduces_results(self, reranker):
        """Test reranker reduces candidate count."""
        candidates = [
            {"chunk_id": f"c{i}", "content": f"Content {i}", "score": 1.0 - i * 0.1}
            for i in range(10)
        ]

        result = await reranker.rerank("test query", candidates, top_k=5)

        assert len(result.results) == 5
        assert result.input_count == 10
        assert result.output_count == 5

    @pytest.mark.asyncio
    async def test_rerank_tracks_rank_changes(self, reranker):
        """Test reranker tracks rank changes."""
        candidates = [
            {"chunk_id": f"c{i}", "content": f"Content {i}", "score": 0.9 - i * 0.1}
            for i in range(10)
        ]

        result = await reranker.rerank("query", candidates, top_k=5)

        for r in result.results:
            assert r.original_rank >= 0
            assert r.new_rank >= 0


class TestFullRetrievalPipeline:
    """End-to-end retrieval pipeline tests."""

    @pytest.fixture
    def full_pipeline(self):
        """Setup complete retrieval pipeline."""
        # Vector search
        vector_store = InMemoryVectorStore(dimensions=384)
        vector_search = VectorSearch(store=vector_store, default_top_k=100)

        # BM25
        bm25_search = BM25Search()

        # Hybrid
        hybrid_config = HybridSearchConfig(
            vector_weight=5.0,
            bm25_weight=3.0,
            vector_top_k=100,
            bm25_top_k=100,
            final_top_k=100,
        )
        hybrid_search = HybridSearch(vector_search, bm25_search, hybrid_config)

        # Reranker
        reranker_config = RerankerConfig(top_k_input=100, top_k_output=10)
        reranker = Reranker(config=reranker_config, reranker=NoOpReranker())

        return {
            "vector_search": vector_search,
            "bm25_search": bm25_search,
            "hybrid_search": hybrid_search,
            "reranker": reranker,
        }

    @pytest.mark.asyncio
    async def test_end_to_end_retrieval(self, full_pipeline):
        """Test full retrieval pipeline: index -> search -> rerank."""
        vs = full_pipeline["vector_search"]
        bm25 = full_pipeline["bm25_search"]
        hybrid = full_pipeline["hybrid_search"]
        reranker = full_pipeline["reranker"]

        # 1. Index documents
        await vs.index(SAMPLE_CHUNKS, SAMPLE_EMBEDDINGS)
        bm25.index(SAMPLE_CHUNKS)

        # 2. Hybrid search
        query = "What is machine learning?"
        query_embedding = [0.2] * 384

        hybrid_results = await hybrid.search(query, query_embedding, top_k=100)
        assert len(hybrid_results.results) > 0

        # 3. Rerank
        candidates = [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": r.combined_score,
                "metadata": r.metadata,
                "document_id": r.document_id,
            }
            for r in hybrid_results.results
        ]

        reranked = await reranker.rerank(query, candidates, top_k=10)

        # Final results
        assert len(reranked.results) <= 10
        assert reranked.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_pipeline_handles_empty_results(self, full_pipeline):
        """Test pipeline handles queries with no matches."""
        vs = full_pipeline["vector_search"]
        bm25 = full_pipeline["bm25_search"]
        hybrid = full_pipeline["hybrid_search"]

        # Index minimal data
        await vs.index(SAMPLE_CHUNKS[:1], SAMPLE_EMBEDDINGS[:1])
        bm25.index(SAMPLE_CHUNKS[:1])

        # Search for something not in index
        query = "quantum computing blockchain"
        query_embedding = [0.9] * 384

        results = await hybrid.search(query, query_embedding, top_k=10)

        # Should not crash, may return low-relevance results
        assert results is not None

    @pytest.mark.asyncio
    async def test_pipeline_latency(self, full_pipeline):
        """Test pipeline latency is reasonable."""
        vs = full_pipeline["vector_search"]
        bm25 = full_pipeline["bm25_search"]
        hybrid = full_pipeline["hybrid_search"]
        reranker = full_pipeline["reranker"]

        await vs.index(SAMPLE_CHUNKS, SAMPLE_EMBEDDINGS)
        bm25.index(SAMPLE_CHUNKS)

        query = "programming language"
        query_embedding = [0.1] * 384

        # Time the full pipeline
        import time
        start = time.perf_counter()

        hybrid_results = await hybrid.search(query, query_embedding, top_k=100)
        candidates = [{"chunk_id": r.chunk_id, "content": r.content, "score": r.combined_score} for r in hybrid_results.results]
        await reranker.rerank(query, candidates, top_k=10)

        total_ms = (time.perf_counter() - start) * 1000

        # Pipeline should complete in reasonable time (< 500ms for in-memory)
        assert total_ms < 500


class TestRetrievalQuality:
    """Test retrieval quality metrics."""

    @pytest.fixture
    def indexed_pipeline(self):
        """Pipeline with indexed documents."""
        vector_store = InMemoryVectorStore(dimensions=384)
        vector_search = VectorSearch(store=vector_store, default_top_k=10)
        bm25_search = BM25Search()

        # Index
        asyncio.get_event_loop().run_until_complete(
            vector_search.index(SAMPLE_CHUNKS, SAMPLE_EMBEDDINGS)
        )
        bm25_search.index(SAMPLE_CHUNKS)

        return vector_search, bm25_search

    @pytest.mark.asyncio
    async def test_relevant_results_ranked_higher(self, indexed_pipeline):
        """Test that more relevant results are ranked higher."""
        vector_search, _ = indexed_pipeline

        # Query for Python (first chunk)
        query_embedding = [0.1] * 384
        response = await vector_search.search(query_embedding, top_k=5)

        # First result should be most similar
        assert response.results[0].score >= response.results[-1].score

    def test_bm25_exact_match_priority(self, indexed_pipeline):
        """Test BM25 prioritizes exact matches."""
        _, bm25_search = indexed_pipeline

        response = bm25_search.search("ERR-404", top_k=5)

        # Exact match should be first
        assert "ERR-404" in response.results[0].content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
