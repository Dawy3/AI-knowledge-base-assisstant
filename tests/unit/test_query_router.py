"""Tests for query router implementation."""

import pytest
from unittest.mock import MagicMock, patch

from backend.core.query.router import (
    ModelTier,
    ModelRoute,
    RoutingReason,
    RoutingDecision,
    RouterConfig,
    RoutingStats,
    QueryRouter,
    AdaptiveRouter,
    create_query_router,
    TIER_TO_MODEL,
    MODEL_TO_TIER,
)
from backend.core.query.classifier import (
    QueryCategory,
    QueryIntent,
    ClassificationResult,
    QueryClassifier,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_classifier():
    """Create mock query classifier."""
    classifier = MagicMock(spec=QueryClassifier)
    classifier.classify.return_value = ClassificationResult(
        query="test query",
        category=QueryCategory.SIMPLE,
        confidence=0.85,
        complexity_score=0.3,
    )
    return classifier


@pytest.fixture
def router(mock_classifier):
    """Create QueryRouter with mock classifier."""
    return QueryRouter(classifier=mock_classifier)


@pytest.fixture
def router_config():
    """Create default RouterConfig."""
    return RouterConfig()


# ============================================================================
# Test Enums
# ============================================================================


class TestModelTier:
    """Test ModelTier enum."""

    def test_tier_values(self):
        assert ModelTier.TIER_1.value == "tier_1"
        assert ModelTier.TIER_2.value == "tier_2"
        assert ModelTier.TIER_3.value == "tier_3"

    def test_tier_count(self):
        assert len(ModelTier) == 3


class TestModelRoute:
    """Test ModelRoute enum."""

    def test_route_values(self):
        assert ModelRoute.GPT35.value == "gpt-3.5-turbo"
        assert ModelRoute.GPT4_MINI.value == "gpt-4o-mini"
        assert ModelRoute.GPT4.value == "gpt-4"
        assert ModelRoute.LOCAL.value == "local"
        assert ModelRoute.CLAUDE.value == "claude"
        assert ModelRoute.FALLBACK.value == "fallback"


class TestRoutingReason:
    """Test RoutingReason enum."""

    def test_tier_based_reasons(self):
        assert RoutingReason.SIMPLE_QUERY.value == "simple_query"
        assert RoutingReason.MODERATE_QUERY.value == "moderate_query"
        assert RoutingReason.COMPLEX_QUERY.value == "complex_query"

    def test_special_case_reasons(self):
        assert RoutingReason.FAQ_QUERY.value == "faq_query"
        assert RoutingReason.UNCERTAIN.value == "uncertain"
        assert RoutingReason.OUT_OF_SCOPE.value == "out_of_scope"

    def test_override_reasons(self):
        assert RoutingReason.USER_PREFERENCE.value == "user_preference"
        assert RoutingReason.LATENCY_REQUIREMENT.value == "latency_requirement"
        assert RoutingReason.MODEL_UNAVAILABLE.value == "model_unavailable"


class TestTierModelMappings:
    """Test tier-to-model mappings."""

    def test_tier_to_model(self):
        assert TIER_TO_MODEL[ModelTier.TIER_1] == ModelRoute.GPT35
        assert TIER_TO_MODEL[ModelTier.TIER_2] == ModelRoute.GPT4_MINI
        assert TIER_TO_MODEL[ModelTier.TIER_3] == ModelRoute.GPT4

    def test_model_to_tier(self):
        assert MODEL_TO_TIER[ModelRoute.GPT35] == ModelTier.TIER_1
        assert MODEL_TO_TIER[ModelRoute.GPT4_MINI] == ModelTier.TIER_2
        assert MODEL_TO_TIER[ModelRoute.GPT4] == ModelTier.TIER_3
        assert MODEL_TO_TIER[ModelRoute.LOCAL] == ModelTier.TIER_1


# ============================================================================
# Test RoutingDecision
# ============================================================================


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_creation(self):
        decision = RoutingDecision(
            route=ModelRoute.GPT35,
            reason=RoutingReason.SIMPLE_QUERY,
            confidence=0.9,
            tier=ModelTier.TIER_1,
        )

        assert decision.route == ModelRoute.GPT35
        assert decision.reason == RoutingReason.SIMPLE_QUERY
        assert decision.confidence == 0.9
        assert decision.tier == ModelTier.TIER_1

    def test_default_values(self):
        decision = RoutingDecision(
            route=ModelRoute.GPT35,
            reason=RoutingReason.SIMPLE_QUERY,
            confidence=0.9,
        )

        assert decision.tier == ModelTier.TIER_1
        assert decision.query == ""
        assert decision.category == QueryCategory.SIMPLE
        assert decision.complexity_score == 0.5
        assert decision.estimated_cost == 0.0
        assert decision.is_fallback is False
        assert decision.metadata == {}

    def test_to_dict(self):
        decision = RoutingDecision(
            route=ModelRoute.GPT4,
            reason=RoutingReason.COMPLEX_QUERY,
            confidence=0.85,
            tier=ModelTier.TIER_3,
            query="complex query",
            category=QueryCategory.COMPLEX,
            complexity_score=0.8,
            estimated_cost=0.05,
        )

        d = decision.to_dict()

        assert d["route"] == "gpt-4"
        assert d["tier"] == "tier_3"
        assert d["reason"] == "complex_query"
        assert d["confidence"] == 0.85
        assert d["query"] == "complex query"
        assert d["category"] == "complex"
        assert d["complexity_score"] == 0.8


# ============================================================================
# Test RouterConfig
# ============================================================================


class TestRouterConfig:
    """Test RouterConfig dataclass."""

    def test_default_values(self):
        config = RouterConfig()

        # Target ratios (70/25/5)
        assert config.tier1_target_ratio == 0.70
        assert config.tier2_target_ratio == 0.25
        assert config.tier3_target_ratio == 0.05

        # Complexity thresholds
        assert config.tier1_max_complexity == 0.4
        assert config.tier2_max_complexity == 0.75

        # Confidence threshold
        assert config.uncertain_threshold == 0.6

        # Model availability
        assert config.tier1_available is True
        assert config.tier2_available is True
        assert config.tier3_available is True

    def test_custom_values(self):
        config = RouterConfig(
            tier1_target_ratio=0.60,
            tier2_target_ratio=0.30,
            tier3_target_ratio=0.10,
            tier1_max_complexity=0.5,
            tier2_max_complexity=0.8,
        )

        assert config.tier1_target_ratio == 0.60
        assert config.tier2_target_ratio == 0.30
        assert config.tier3_target_ratio == 0.10
        assert config.tier1_max_complexity == 0.5
        assert config.tier2_max_complexity == 0.8

    def test_legacy_aliases(self):
        config = RouterConfig()

        assert config.gpt35_target_ratio == config.tier1_target_ratio
        assert config.gpt4_target_ratio == config.tier3_target_ratio
        assert config.simple_max_complexity == config.tier1_max_complexity
        assert config.gpt4_min_complexity == config.tier2_max_complexity
        assert config.gpt35_available == config.tier1_available
        assert config.gpt4_available == config.tier3_available

    def test_cost_configuration(self):
        config = RouterConfig()

        assert config.tier1_cost_per_1k == 0.002
        assert config.tier2_cost_per_1k == 0.015
        assert config.tier3_cost_per_1k == 0.03
        assert config.local_cost_per_1k == 0.0


# ============================================================================
# Test RoutingStats
# ============================================================================


class TestRoutingStats:
    """Test RoutingStats dataclass."""

    def test_initial_values(self):
        stats = RoutingStats()

        assert stats.total_queries == 0
        assert stats.tier1_queries == 0
        assert stats.tier2_queries == 0
        assert stats.tier3_queries == 0
        assert stats.total_cost == 0.0

    def test_tier_ratios_empty(self):
        stats = RoutingStats()

        assert stats.tier1_ratio == 0.0
        assert stats.tier2_ratio == 0.0
        assert stats.tier3_ratio == 0.0

    def test_tier_ratios_with_data(self):
        stats = RoutingStats(
            total_queries=100,
            tier1_queries=70,
            tier2_queries=25,
            tier3_queries=5,
        )

        assert stats.tier1_ratio == 0.70
        assert stats.tier2_ratio == 0.25
        assert stats.tier3_ratio == 0.05

    def test_legacy_aliases(self):
        stats = RoutingStats(
            total_queries=100,
            tier1_queries=70,
            tier3_queries=5,
        )

        assert stats.gpt35_queries == stats.tier1_queries
        assert stats.gpt4_queries == stats.tier3_queries
        assert stats.gpt35_ratio == stats.tier1_ratio
        assert stats.gpt4_ratio == stats.tier3_ratio

    def test_to_dict(self):
        stats = RoutingStats(
            total_queries=100,
            tier1_queries=70,
            tier2_queries=25,
            tier3_queries=5,
            total_cost=1.5,
        )

        d = stats.to_dict()

        assert d["total_queries"] == 100
        assert d["tier1_queries"] == 70
        assert d["tier2_queries"] == 25
        assert d["tier3_queries"] == 5
        assert d["tier1_ratio"] == 0.70
        assert d["total_cost"] == 1.5


# ============================================================================
# Test QueryRouter
# ============================================================================


class TestQueryRouterInit:
    """Test QueryRouter initialization."""

    def test_default_initialization(self):
        router = QueryRouter()

        assert router.classifier is not None
        assert router.config is not None
        assert router._stats.total_queries == 0

    def test_custom_classifier(self, mock_classifier):
        router = QueryRouter(classifier=mock_classifier)

        assert router.classifier == mock_classifier

    def test_custom_config(self):
        config = RouterConfig(tier1_max_complexity=0.5)
        router = QueryRouter(config=config)

        assert router.config.tier1_max_complexity == 0.5


class TestQueryRouterRouting:
    """Test QueryRouter routing decisions."""

    def test_simple_query_routes_to_tier1(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="What is Python?",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.2,
        )
        router = QueryRouter(classifier=mock_classifier)

        decision = router.route("What is Python?")

        assert decision.tier == ModelTier.TIER_1
        assert decision.route == ModelRoute.GPT35
        assert decision.reason == RoutingReason.SIMPLE_QUERY

    def test_complex_query_routes_to_tier3(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Complex query",
            category=QueryCategory.COMPLEX,
            confidence=0.9,
            complexity_score=0.85,
        )
        router = QueryRouter(classifier=mock_classifier)

        decision = router.route("Complex query")

        assert decision.tier == ModelTier.TIER_3
        assert decision.route == ModelRoute.GPT4
        assert decision.reason == RoutingReason.COMPLEX_QUERY

    def test_moderate_query_routes_to_tier2(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Moderate query",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.55,
        )
        router = QueryRouter(classifier=mock_classifier)

        decision = router.route("Moderate query")

        assert decision.tier == ModelTier.TIER_2
        assert decision.route == ModelRoute.GPT4_MINI
        assert decision.reason == RoutingReason.MODERATE_QUERY

    def test_faq_query_routes_to_tier1(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="FAQ query",
            category=QueryCategory.FAQ,
            confidence=0.9,
            complexity_score=0.3,
        )
        router = QueryRouter(classifier=mock_classifier)

        decision = router.route("FAQ query")

        assert decision.tier == ModelTier.TIER_1
        assert decision.reason == RoutingReason.FAQ_QUERY

    def test_out_of_scope_routes_to_tier1(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Out of scope",
            category=QueryCategory.OUT_OF_SCOPE,
            confidence=0.9,
            complexity_score=0.5,
        )
        router = QueryRouter(classifier=mock_classifier)

        decision = router.route("Out of scope")

        assert decision.tier == ModelTier.TIER_1
        assert decision.reason == RoutingReason.OUT_OF_SCOPE

    def test_uncertain_routes_to_higher_tier(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Uncertain query",
            category=QueryCategory.SIMPLE,
            confidence=0.4,  # Below uncertain_threshold
            complexity_score=0.3,
        )
        router = QueryRouter(classifier=mock_classifier)

        decision = router.route("Uncertain query")

        # Should bump from TIER_1 to TIER_2
        assert decision.tier == ModelTier.TIER_2
        assert decision.reason == RoutingReason.UNCERTAIN

    def test_user_preference_tier(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Test",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.2,
        )
        router = QueryRouter(classifier=mock_classifier)

        decision = router.route("Test", context={"preferred_tier": "tier_3"})

        assert decision.tier == ModelTier.TIER_3
        assert decision.reason == RoutingReason.USER_PREFERENCE

    def test_user_preference_model(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Test",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.2,
        )
        router = QueryRouter(classifier=mock_classifier)

        decision = router.route("Test", context={"preferred_model": "gpt-4"})

        assert decision.route == ModelRoute.GPT4
        assert decision.tier == ModelTier.TIER_3
        assert decision.reason == RoutingReason.USER_PREFERENCE

    def test_latency_requirement_routes_to_tier1(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Fast query",
            category=QueryCategory.COMPLEX,
            confidence=0.9,
            complexity_score=0.8,
        )
        router = QueryRouter(classifier=mock_classifier)

        decision = router.route("Fast query", context={"max_latency_ms": 100})

        assert decision.tier == ModelTier.TIER_1
        assert decision.reason == RoutingReason.LATENCY_REQUIREMENT


class TestQueryRouterFallback:
    """Test QueryRouter fallback handling."""

    def test_tier1_unavailable_falls_back_to_tier2(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Test",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.2,
        )
        config = RouterConfig(tier1_available=False)
        router = QueryRouter(classifier=mock_classifier, config=config)

        decision = router.route("Test")

        assert decision.tier == ModelTier.TIER_2
        assert decision.is_fallback is True

    def test_tier3_unavailable_falls_back_to_tier2(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Complex",
            category=QueryCategory.COMPLEX,
            confidence=0.9,
            complexity_score=0.9,
        )
        config = RouterConfig(tier3_available=False)
        router = QueryRouter(classifier=mock_classifier, config=config)

        decision = router.route("Complex")

        assert decision.tier == ModelTier.TIER_2
        assert decision.is_fallback is True

    def test_fallback_route_assignment(self, router):
        decision = router.route("Test")

        # Should have fallback route assigned
        assert decision.fallback_route is not None


class TestQueryRouterStats:
    """Test QueryRouter statistics tracking."""

    def test_stats_increment_on_route(self, router):
        router.route("Query 1")
        router.route("Query 2")
        router.route("Query 3")

        stats = router.get_stats()

        assert stats.total_queries == 3

    def test_stats_track_tier1(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Simple",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.2,
        )
        router = QueryRouter(classifier=mock_classifier)

        router.route("Simple 1")
        router.route("Simple 2")

        stats = router.get_stats()

        assert stats.tier1_queries == 2

    def test_stats_track_tier3(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Complex",
            category=QueryCategory.COMPLEX,
            confidence=0.9,
            complexity_score=0.9,
        )
        router = QueryRouter(classifier=mock_classifier)

        router.route("Complex")

        stats = router.get_stats()

        assert stats.tier3_queries == 1

    def test_stats_reset(self, router):
        router.route("Query 1")
        router.route("Query 2")

        router.reset_stats()

        stats = router.get_stats()
        assert stats.total_queries == 0

    def test_routing_distribution(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Test",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.2,
        )
        router = QueryRouter(classifier=mock_classifier)

        for _ in range(10):
            router.route("Test")

        distribution = router.get_routing_distribution()

        assert "tier1" in distribution
        assert "tier2" in distribution
        assert "tier3" in distribution
        assert "target_tier1" in distribution


class TestQueryRouterHelpers:
    """Test QueryRouter helper methods."""

    def test_get_tier(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Test",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.2,
        )
        router = QueryRouter(classifier=mock_classifier)

        tier = router.get_tier("Test")

        assert tier == ModelTier.TIER_1

    def test_should_use_tier1(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Simple",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.2,
        )
        router = QueryRouter(classifier=mock_classifier)

        assert router.should_use_tier1("Simple") is True
        assert router.should_use_simple_model("Simple") is True

    def test_should_use_tier3(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Complex",
            category=QueryCategory.COMPLEX,
            confidence=0.9,
            complexity_score=0.9,
        )
        router = QueryRouter(classifier=mock_classifier)

        assert router.should_use_tier3("Complex") is True
        assert router.should_use_complex_model("Complex") is True

    def test_route_batch(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Test",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.3,
        )
        router = QueryRouter(classifier=mock_classifier)

        decisions = router.route_batch(["Query 1", "Query 2", "Query 3"])

        assert len(decisions) == 3
        assert all(isinstance(d, RoutingDecision) for d in decisions)

    def test_update_config(self, router):
        router.update_config(tier1_max_complexity=0.5, tier2_max_complexity=0.85)

        assert router.config.tier1_max_complexity == 0.5
        assert router.config.tier2_max_complexity == 0.85

    def test_update_config_legacy_params(self, router):
        router.update_config(simple_max_complexity=0.45)

        assert router.config.tier1_max_complexity == 0.45


class TestQueryRouterAsync:
    """Test QueryRouter async methods."""

    @pytest.mark.asyncio
    async def test_route_async(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Test",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.3,
        )
        router = QueryRouter(classifier=mock_classifier)

        decision = await router.route_async("Test")

        assert isinstance(decision, RoutingDecision)


# ============================================================================
# Test AdaptiveRouter
# ============================================================================


class TestAdaptiveRouter:
    """Test AdaptiveRouter class."""

    def test_initialization(self, mock_classifier):
        router = AdaptiveRouter(classifier=mock_classifier, adjustment_interval=50)

        assert router.adjustment_interval == 50
        assert router._queries_since_adjustment == 0

    def test_routing_increments_counter(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Test",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.3,
        )
        router = AdaptiveRouter(classifier=mock_classifier, adjustment_interval=100)

        router.route("Test 1")
        router.route("Test 2")

        assert router._queries_since_adjustment == 2

    def test_adjustment_resets_counter(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Test",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.3,
        )
        router = AdaptiveRouter(classifier=mock_classifier, adjustment_interval=3)

        router.route("Test 1")
        router.route("Test 2")
        router.route("Test 3")  # Should trigger adjustment

        assert router._queries_since_adjustment == 0


# ============================================================================
# Test create_query_router Factory
# ============================================================================


class TestCreateQueryRouter:
    """Test create_query_router factory function."""

    def test_create_default(self):
        router = create_query_router()

        assert isinstance(router, QueryRouter)
        assert router.config.tier1_target_ratio == 0.70
        assert router.config.tier2_target_ratio == 0.25
        assert router.config.tier3_target_ratio == 0.05

    def test_create_with_custom_ratios(self):
        router = create_query_router(
            tier1_target_ratio=0.60,
            tier2_target_ratio=0.30,
            tier3_target_ratio=0.10,
        )

        assert router.config.tier1_target_ratio == 0.60
        assert router.config.tier2_target_ratio == 0.30
        assert router.config.tier3_target_ratio == 0.10

    def test_create_adaptive(self):
        router = create_query_router(adaptive=True)

        assert isinstance(router, AdaptiveRouter)

    def test_create_with_classifier(self, mock_classifier):
        router = create_query_router(classifier=mock_classifier)

        assert router.classifier == mock_classifier

    def test_create_with_legacy_param(self):
        router = create_query_router(gpt35_target_ratio=0.80)

        assert router.config.tier1_target_ratio == 0.80

    def test_create_with_kwargs(self):
        router = create_query_router(
            tier1_max_complexity=0.5,
            tier2_max_complexity=0.8,
        )

        assert router.config.tier1_max_complexity == 0.5
        assert router.config.tier2_max_complexity == 0.8


# ============================================================================
# Test Cost Estimation
# ============================================================================


class TestCostEstimation:
    """Test cost estimation in routing."""

    def test_tier1_cost_estimate(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Simple query",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.2,
        )
        router = QueryRouter(classifier=mock_classifier)

        decision = router.route("Simple query")

        assert decision.estimated_cost > 0
        assert decision.estimated_cost < 0.01  # Tier 1 is cheap

    def test_tier3_costs_more(self, mock_classifier):
        router = QueryRouter(classifier=mock_classifier)

        # Route simple query
        mock_classifier.classify.return_value = ClassificationResult(
            query="Simple",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.2,
        )
        simple_decision = router.route("Simple query here")

        # Route complex query
        mock_classifier.classify.return_value = ClassificationResult(
            query="Complex",
            category=QueryCategory.COMPLEX,
            confidence=0.9,
            complexity_score=0.9,
        )
        complex_decision = router.route("Complex query here")

        # Tier 3 should cost more than Tier 1
        assert complex_decision.estimated_cost > simple_decision.estimated_cost

    def test_estimated_savings_tracked(self, mock_classifier):
        mock_classifier.classify.return_value = ClassificationResult(
            query="Simple",
            category=QueryCategory.SIMPLE,
            confidence=0.9,
            complexity_score=0.2,
        )
        router = QueryRouter(classifier=mock_classifier)

        for _ in range(10):
            router.route("Simple query")

        stats = router.get_stats()

        # Should have savings (compared to routing all to Tier 3)
        assert stats.estimated_savings > 0
