"""Tests for hybrid search implementation."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.retrieval.hybrid_search import (
    FusionMethod,
    HybridSearchConfig,
    HybridSearchResult,
    HybridSearchResponse,
    ScoreFusion,
    HybridSearch,
    create_hybrid_search,
)
from backend.core.retrieval.vector_search import VectorSearchResult, VectorSearchResponse
from backend.core.retrieval.bm25_search import BM25SearchResult, BM25SearchResponse


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_vector_results():
    """Create mock vector search results."""
    return [
        VectorSearchResult(
            chunk_id="chunk_1",
            score=0.95,
            content="Vector result 1 content",
            metadata={"source": "doc1"},
            document_id="doc1",
        ),
        VectorSearchResult(
            chunk_id="chunk_2",
            score=0.85,
            content="Vector result 2 content",
            metadata={"source": "doc1"},
            document_id="doc1",
        ),
        VectorSearchResult(
            chunk_id="chunk_3",
            score=0.75,
            content="Vector result 3 content",
            metadata={"source": "doc2"},
            document_id="doc2",
        ),
    ]


@pytest.fixture
def mock_bm25_results():
    """Create mock BM25 search results."""
    return [
        BM25SearchResult(
            chunk_id="chunk_2",  # Overlaps with vector
            score=5.5,
            content="BM25 result 2 content",
            metadata={"source": "doc1"},
            document_id="doc1",
        ),
        BM25SearchResult(
            chunk_id="chunk_4",
            score=4.2,
            content="BM25 result 4 content",
            metadata={"source": "doc3"},
            document_id="doc3",
        ),
        BM25SearchResult(
            chunk_id="chunk_5",
            score=3.1,
            content="BM25 result 5 content",
            metadata={"source": "doc3"},
            document_id="doc3",
        ),
    ]


@pytest.fixture
def mock_vector_search(mock_vector_results):
    """Create mock VectorSearch instance."""
    mock = AsyncMock()
    mock.search.return_value = VectorSearchResponse(
        results=mock_vector_results,
        query_embedding=[0.1] * 1536,
        latency_ms=15.0,
        total_candidates=3,
    )
    return mock


@pytest.fixture
def mock_bm25_search(mock_bm25_results):
    """Create mock BM25Search instance."""
    mock = MagicMock()
    mock.search.return_value = BM25SearchResponse(
        results=mock_bm25_results,
        query_tokens=["test", "query"],
        latency_ms=5.0,
        total_documents=100,
    )
    return mock


@pytest.fixture
def hybrid_search(mock_vector_search, mock_bm25_search):
    """Create HybridSearch instance with mocks."""
    return HybridSearch(
        vector_search=mock_vector_search,
        bm25_search=mock_bm25_search,
    )


# ============================================================================
# Test FusionMethod Enum
# ============================================================================


class TestFusionMethod:
    """Test FusionMethod enum."""

    def test_weighted_sum_value(self):
        assert FusionMethod.WEIGHTED_SUM.value == "weighted_sum"

    def test_rrf_value(self):
        assert FusionMethod.RECIPROCAL_RANK.value == "rrf"

    def test_relative_score_value(self):
        assert FusionMethod.RELATIVE_SCORE.value == "relative_score"


# ============================================================================
# Test HybridSearchConfig
# ============================================================================


class TestHybridSearchConfig:
    """Test HybridSearchConfig dataclass."""

    def test_default_values(self):
        config = HybridSearchConfig()

        assert config.vector_weight == 5.0
        assert config.bm25_weight == 3.0
        assert config.recency_weight == 0.2
        assert config.fusion_method == FusionMethod.WEIGHTED_SUM
        assert config.rrf_k == 60
        assert config.vector_top_k == 100
        assert config.bm25_top_k == 100
        assert config.final_top_k == 100
        assert config.enable_recency_boost is False
        assert config.normalize_scores is True

    def test_custom_values(self):
        config = HybridSearchConfig(
            vector_weight=7.0,
            bm25_weight=4.0,
            recency_weight=0.5,
            fusion_method=FusionMethod.RECIPROCAL_RANK,
            rrf_k=30,
            vector_top_k=50,
            bm25_top_k=50,
            final_top_k=20,
            enable_recency_boost=True,
        )

        assert config.vector_weight == 7.0
        assert config.bm25_weight == 4.0
        assert config.recency_weight == 0.5
        assert config.fusion_method == FusionMethod.RECIPROCAL_RANK
        assert config.rrf_k == 30
        assert config.enable_recency_boost is True


# ============================================================================
# Test HybridSearchResult
# ============================================================================


class TestHybridSearchResult:
    """Test HybridSearchResult dataclass."""

    def test_creation(self):
        result = HybridSearchResult(
            chunk_id="chunk_1",
            combined_score=8.5,
            vector_score=0.9,
            bm25_score=0.7,
            content="Test content",
            document_id="doc1",
        )

        assert result.chunk_id == "chunk_1"
        assert result.combined_score == 8.5
        assert result.vector_score == 0.9
        assert result.bm25_score == 0.7
        assert result.content == "Test content"

    def test_default_values(self):
        result = HybridSearchResult(chunk_id="test", combined_score=1.0)

        assert result.vector_score == 0.0
        assert result.bm25_score == 0.0
        assert result.recency_score == 0.0
        assert result.content == ""
        assert result.metadata == {}
        assert result.from_vector is False
        assert result.from_bm25 is False

    def test_to_dict(self):
        result = HybridSearchResult(
            chunk_id="chunk_1",
            combined_score=8.5,
            vector_score=0.9,
            bm25_score=0.7,
            recency_score=0.3,
            content="Test content",
            metadata={"key": "value"},
            document_id="doc1",
            from_vector=True,
            from_bm25=True,
        )

        d = result.to_dict()

        assert d["chunk_id"] == "chunk_1"
        assert d["combined_score"] == 8.5
        assert d["vector_score"] == 0.9
        assert d["bm25_score"] == 0.7
        assert d["recency_score"] == 0.3
        assert d["content"] == "Test content"
        assert d["metadata"] == {"key": "value"}
        assert d["from_vector"] is True
        assert d["from_bm25"] is True


# ============================================================================
# Test HybridSearchResponse
# ============================================================================


class TestHybridSearchResponse:
    """Test HybridSearchResponse dataclass."""

    def test_creation(self):
        results = [
            HybridSearchResult(chunk_id="1", combined_score=0.9),
            HybridSearchResult(chunk_id="2", combined_score=0.8),
        ]
        response = HybridSearchResponse(
            results=results,
            query="test query",
            latency_ms=20.0,
        )

        assert len(response.results) == 2
        assert response.query == "test query"
        assert response.latency_ms == 20.0

    def test_top_score_property(self):
        results = [
            HybridSearchResult(chunk_id="1", combined_score=0.9),
            HybridSearchResult(chunk_id="2", combined_score=0.8),
        ]
        response = HybridSearchResponse(results=results)

        assert response.top_score == 0.9

    def test_top_score_empty_results(self):
        response = HybridSearchResponse(results=[])

        assert response.top_score == 0.0

    def test_overlap_ratio(self):
        response = HybridSearchResponse(
            results=[],
            unique_results=10,
            overlap_count=4,
        )

        assert response.overlap_ratio == 0.4

    def test_overlap_ratio_zero_unique(self):
        response = HybridSearchResponse(
            results=[],
            unique_results=0,
            overlap_count=0,
        )

        assert response.overlap_ratio == 0.0


# ============================================================================
# Test ScoreFusion
# ============================================================================


class TestScoreFusion:
    """Test ScoreFusion methods."""

    def test_weighted_sum(self):
        config = HybridSearchConfig(
            vector_weight=5.0,
            bm25_weight=3.0,
            recency_weight=0.2,
        )

        score = ScoreFusion.weighted_sum(
            vector_score=0.8,
            bm25_score=0.6,
            recency_score=0.5,
            config=config,
        )

        # (0.8 * 5) + (0.6 * 3) + (0.5 * 0.2) = 4 + 1.8 + 0.1 = 5.9
        assert score == pytest.approx(5.9)

    def test_weighted_sum_zero_scores(self):
        config = HybridSearchConfig()

        score = ScoreFusion.weighted_sum(0.0, 0.0, 0.0, config)

        assert score == 0.0

    def test_reciprocal_rank_fusion_both_present(self):
        score = ScoreFusion.reciprocal_rank_fusion(
            vector_rank=0,
            bm25_rank=2,
            k=60,
        )

        # 1/(60+0+1) + 1/(60+2+1) = 1/61 + 1/63 ≈ 0.0164 + 0.0159 ≈ 0.0323
        expected = 1 / 61 + 1 / 63
        assert score == pytest.approx(expected)

    def test_reciprocal_rank_fusion_vector_only(self):
        score = ScoreFusion.reciprocal_rank_fusion(
            vector_rank=0,
            bm25_rank=None,
            k=60,
        )

        expected = 1 / 61
        assert score == pytest.approx(expected)

    def test_reciprocal_rank_fusion_bm25_only(self):
        score = ScoreFusion.reciprocal_rank_fusion(
            vector_rank=None,
            bm25_rank=0,
            k=60,
        )

        expected = 1 / 61
        assert score == pytest.approx(expected)

    def test_normalize_scores(self):
        scores = [0.2, 0.5, 0.8, 1.0]

        normalized = ScoreFusion.normalize_scores(scores)

        assert normalized[0] == pytest.approx(0.0)  # min -> 0
        assert normalized[-1] == pytest.approx(1.0)  # max -> 1
        assert normalized[1] == pytest.approx(0.375)  # (0.5-0.2)/(1.0-0.2)
        assert normalized[2] == pytest.approx(0.75)  # (0.8-0.2)/(1.0-0.2)

    def test_normalize_scores_empty(self):
        assert ScoreFusion.normalize_scores([]) == []

    def test_normalize_scores_same_values(self):
        scores = [0.5, 0.5, 0.5]

        normalized = ScoreFusion.normalize_scores(scores)

        assert all(s == 1.0 for s in normalized)


# ============================================================================
# Test HybridSearch
# ============================================================================


class TestHybridSearch:
    """Test HybridSearch class."""

    def test_initialization(self, mock_vector_search, mock_bm25_search):
        hybrid = HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
        )

        assert hybrid.vector_search == mock_vector_search
        assert hybrid.bm25_search == mock_bm25_search
        assert hybrid.config is not None

    def test_initialization_with_config(self, mock_vector_search, mock_bm25_search):
        config = HybridSearchConfig(vector_weight=10.0)

        hybrid = HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            config=config,
        )

        assert hybrid.config.vector_weight == 10.0

    @pytest.mark.asyncio
    async def test_search_returns_response(self, hybrid_search):
        query_embedding = [0.1] * 1536

        response = await hybrid_search.search(
            query="test query",
            query_embedding=query_embedding,
        )

        assert isinstance(response, HybridSearchResponse)
        assert response.query == "test query"
        assert len(response.results) > 0

    @pytest.mark.asyncio
    async def test_search_calls_both_searches(
        self, hybrid_search, mock_vector_search, mock_bm25_search
    ):
        query_embedding = [0.1] * 1536

        await hybrid_search.search(
            query="test query",
            query_embedding=query_embedding,
        )

        mock_vector_search.search.assert_called_once()
        mock_bm25_search.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_fuses_results(self, hybrid_search):
        query_embedding = [0.1] * 1536

        response = await hybrid_search.search(
            query="test query",
            query_embedding=query_embedding,
        )

        # Should have results from both sources
        chunk_ids = {r.chunk_id for r in response.results}

        # chunk_2 appears in both, others are unique
        assert "chunk_1" in chunk_ids  # vector only
        assert "chunk_2" in chunk_ids  # both
        assert "chunk_4" in chunk_ids  # bm25 only

    @pytest.mark.asyncio
    async def test_search_tracks_overlap(self, hybrid_search):
        query_embedding = [0.1] * 1536

        response = await hybrid_search.search(
            query="test query",
            query_embedding=query_embedding,
        )

        # chunk_2 is in both vector and bm25 results
        assert response.overlap_count == 1
        assert response.total_vector_results == 3
        assert response.total_bm25_results == 3

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, hybrid_search):
        query_embedding = [0.1] * 1536

        response = await hybrid_search.search(
            query="test query",
            query_embedding=query_embedding,
            top_k=2,
        )

        assert len(response.results) <= 2

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_score(self, hybrid_search):
        query_embedding = [0.1] * 1536

        response = await hybrid_search.search(
            query="test query",
            query_embedding=query_embedding,
        )

        scores = [r.combined_score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_marks_source(self, hybrid_search):
        query_embedding = [0.1] * 1536

        response = await hybrid_search.search(
            query="test query",
            query_embedding=query_embedding,
        )

        results_by_id = {r.chunk_id: r for r in response.results}

        # chunk_1 is vector only
        assert results_by_id["chunk_1"].from_vector is True
        assert results_by_id["chunk_1"].from_bm25 is False

        # chunk_2 is in both
        assert results_by_id["chunk_2"].from_vector is True
        assert results_by_id["chunk_2"].from_bm25 is True

        # chunk_4 is bm25 only
        assert results_by_id["chunk_4"].from_vector is False
        assert results_by_id["chunk_4"].from_bm25 is True

    @pytest.mark.asyncio
    async def test_search_vector_only(self, hybrid_search):
        query_embedding = [0.1] * 1536

        response = await hybrid_search.search_vector_only(
            query_embedding=query_embedding,
            top_k=10,
        )

        assert isinstance(response, HybridSearchResponse)
        assert response.total_bm25_results == 0
        assert all(r.from_vector for r in response.results)

    def test_update_weights(self, hybrid_search):
        hybrid_search.update_weights(
            vector_weight=10.0,
            bm25_weight=5.0,
            recency_weight=0.5,
        )

        assert hybrid_search.config.vector_weight == 10.0
        assert hybrid_search.config.bm25_weight == 5.0
        assert hybrid_search.config.recency_weight == 0.5

    def test_update_weights_partial(self, hybrid_search):
        original_bm25 = hybrid_search.config.bm25_weight

        hybrid_search.update_weights(vector_weight=10.0)

        assert hybrid_search.config.vector_weight == 10.0
        assert hybrid_search.config.bm25_weight == original_bm25

    def test_get_stats(self, hybrid_search):
        stats = hybrid_search.get_stats()

        assert "vector_weight" in stats
        assert "bm25_weight" in stats
        assert "recency_weight" in stats
        assert "fusion_method" in stats
        assert stats["fusion_method"] == "weighted_sum"


class TestHybridSearchRRF:
    """Test HybridSearch with RRF fusion."""

    @pytest.fixture
    def hybrid_search_rrf(self, mock_vector_search, mock_bm25_search):
        config = HybridSearchConfig(fusion_method=FusionMethod.RECIPROCAL_RANK)
        return HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_rrf_fusion(self, hybrid_search_rrf):
        query_embedding = [0.1] * 1536

        response = await hybrid_search_rrf.search(
            query="test query",
            query_embedding=query_embedding,
        )

        assert len(response.results) > 0
        # RRF scores should be small (reciprocal of rank + k)
        for result in response.results:
            assert result.combined_score < 1.0


class TestHybridSearchRecency:
    """Test HybridSearch with recency boost."""

    @pytest.fixture
    def mock_vector_results_with_dates(self):
        now = datetime.now()
        return [
            VectorSearchResult(
                chunk_id="recent",
                score=0.8,
                content="Recent content",
                metadata={
                    "created_at": now.isoformat(),
                },
                document_id="doc1",
            ),
            VectorSearchResult(
                chunk_id="old",
                score=0.9,
                content="Old content",
                metadata={
                    "created_at": (now - timedelta(days=60)).isoformat(),
                },
                document_id="doc2",
            ),
        ]

    @pytest.fixture
    def hybrid_search_recency(self, mock_vector_results_with_dates, mock_bm25_search):
        mock_vector = AsyncMock()
        mock_vector.search.return_value = VectorSearchResponse(
            results=mock_vector_results_with_dates,
            latency_ms=10.0,
        )

        config = HybridSearchConfig(
            enable_recency_boost=True,
            recency_weight=1.0,
            recency_decay_days=30.0,
        )

        return HybridSearch(
            vector_search=mock_vector,
            bm25_search=mock_bm25_search,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_recency_boost_applied(self, hybrid_search_recency):
        query_embedding = [0.1] * 1536

        response = await hybrid_search_recency.search(
            query="test query",
            query_embedding=query_embedding,
        )

        results_by_id = {r.chunk_id: r for r in response.results}

        # Recent document should have higher recency score
        if "recent" in results_by_id and "old" in results_by_id:
            assert results_by_id["recent"].recency_score > results_by_id["old"].recency_score


# ============================================================================
# Test create_hybrid_search Factory
# ============================================================================


class TestCreateHybridSearch:
    """Test create_hybrid_search factory function."""

    def test_create_with_defaults(self, mock_vector_search, mock_bm25_search):
        hybrid = create_hybrid_search(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
        )

        assert isinstance(hybrid, HybridSearch)
        assert hybrid.config.vector_weight == 5.0
        assert hybrid.config.bm25_weight == 3.0
        assert hybrid.config.recency_weight == 0.2

    def test_create_with_custom_weights(self, mock_vector_search, mock_bm25_search):
        hybrid = create_hybrid_search(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            vector_weight=10.0,
            bm25_weight=5.0,
            recency_weight=0.5,
        )

        assert hybrid.config.vector_weight == 10.0
        assert hybrid.config.bm25_weight == 5.0
        assert hybrid.config.recency_weight == 0.5

    def test_create_with_rrf_fusion(self, mock_vector_search, mock_bm25_search):
        hybrid = create_hybrid_search(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            fusion_method="rrf",
        )

        assert hybrid.config.fusion_method == FusionMethod.RECIPROCAL_RANK

    def test_create_with_kwargs(self, mock_vector_search, mock_bm25_search):
        hybrid = create_hybrid_search(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            vector_top_k=50,
            bm25_top_k=50,
            final_top_k=20,
        )

        assert hybrid.config.vector_top_k == 50
        assert hybrid.config.bm25_top_k == 50
        assert hybrid.config.final_top_k == 20
