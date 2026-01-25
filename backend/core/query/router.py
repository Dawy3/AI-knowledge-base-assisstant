"""
Model Routing Logic for RAG Pipeline.

MODEL ROUTING PATTERN - 3-Tier Architecture:
- Tier 1 (70% queries): Simple model for straightforward Q&A
- Tier 2 (25% queries): Medium model for moderate complexity
- Tier 3 (5% queries): Best model for complex reasoning

FOCUS: Route queries to appropriate model tier based on complexity
MUST: When uncertain, route to higher tier (prefer quality over cost)
EXPECTED: Cost optimization with quality balance

Intelligent routing optimizes:
1. Cost - Route simple queries to cheapest model (Tier 1)
2. Quality - Route complex queries to best model (Tier 3)
3. Balance - Route moderate queries to medium model (Tier 2)
4. Latency - Route time-sensitive queries to faster models
5. Reliability - Fallback routing on model failures

Routing strategy:
- Simple/FAQ queries → Tier 1: GPT-3.5 (fast, cheap)
- Moderate complexity → Tier 2: GPT-4o-mini (balanced)
- Complex reasoning → Tier 3: GPT-4 (expensive but highest quality)
- Uncertain → Higher tier (prefer quality over cost when uncertain)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .classifier import ClassificationResult, QueryCategory, QueryClassifier

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tiers for routing pattern."""

    TIER_1 = "tier_1"  # Simple model (70% of queries) - straightforward Q&A
    TIER_2 = "tier_2"  # Medium model (25% of queries) - moderate complexity
    TIER_3 = "tier_3"  # Best model (5% of queries) - complex reasoning


class ModelRoute(str, Enum):
    """Available model routes mapped to tiers."""

    # Tier 1: Simple model (70%) - Fast, cheap, straightforward Q&A
    GPT35 = "gpt-3.5-turbo"

    # Tier 2: Medium model (25%) - Balanced cost/quality for moderate complexity
    GPT4_MINI = "gpt-4o-mini"

    # Tier 3: Best model (5%) - Expensive but highest quality for complex reasoning
    GPT4 = "gpt-4"

    # Alternatives and fallbacks
    LOCAL = "local"              # Local fine-tuned model
    CLAUDE = "claude"            # Claude as alternative
    FALLBACK = "fallback"        # Fallback model


# Mapping from tiers to default models
TIER_TO_MODEL: dict[ModelTier, ModelRoute] = {
    ModelTier.TIER_1: ModelRoute.GPT35,
    ModelTier.TIER_2: ModelRoute.GPT4_MINI,
    ModelTier.TIER_3: ModelRoute.GPT4,
}

# Mapping from models to tiers
MODEL_TO_TIER: dict[ModelRoute, ModelTier] = {
    ModelRoute.GPT35: ModelTier.TIER_1,
    ModelRoute.GPT4_MINI: ModelTier.TIER_2,
    ModelRoute.GPT4: ModelTier.TIER_3,
    ModelRoute.LOCAL: ModelTier.TIER_1,  # Local treated as Tier 1
}


class RoutingReason(str, Enum):
    """Reason for routing decision."""

    # Tier-based routing reasons
    SIMPLE_QUERY = "simple_query"        # Tier 1: Straightforward Q&A
    MODERATE_QUERY = "moderate_query"    # Tier 2: Moderate complexity
    COMPLEX_QUERY = "complex_query"      # Tier 3: Complex reasoning

    # Special case routing
    FAQ_QUERY = "faq_query"              # Tier 1: Known FAQ pattern
    UNCERTAIN = "uncertain"              # Route to higher tier when uncertain
    OUT_OF_SCOPE = "out_of_scope"        # Tier 1: Just explain it's out of scope

    # User/system overrides
    USER_PREFERENCE = "user_preference"
    COST_OPTIMIZATION = "cost_optimization"
    LATENCY_REQUIREMENT = "latency_requirement"
    MODEL_UNAVAILABLE = "model_unavailable"
    FALLBACK = "fallback"


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    route: ModelRoute
    reason: RoutingReason
    confidence: float  # 0-1 confidence in routing decision

    # Tier information
    tier: ModelTier = ModelTier.TIER_1

    # Original classification
    query: str = ""
    category: QueryCategory = QueryCategory.SIMPLE
    complexity_score: float = 0.5

    # Cost estimates
    estimated_cost: float = 0.0  # Estimated cost for this query

    # Performance
    latency_ms: float = 0.0

    # Fallback info
    fallback_route: Optional[ModelRoute] = None
    is_fallback: bool = False

    # Metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "route": self.route.value,
            "tier": self.tier.value,
            "reason": self.reason.value,
            "confidence": self.confidence,
            "query": self.query,
            "category": self.category.value,
            "complexity_score": self.complexity_score,
            "estimated_cost": self.estimated_cost,
            "latency_ms": self.latency_ms,
            "is_fallback": self.is_fallback,
        }


@dataclass
class RouterConfig:
    """
    Configuration for 3-tier query router.

    Model Routing Pattern:
    - Tier 1 (70%): Simple model for straightforward Q&A
    - Tier 2 (25%): Medium model for moderate complexity
    - Tier 3 (5%): Best model for complex reasoning
    """

    # Target routing distribution (must sum to 1.0)
    tier1_target_ratio: float = 0.70  # 70% to Tier 1 (GPT-3.5) - simple queries
    tier2_target_ratio: float = 0.25  # 25% to Tier 2 (GPT-4o-mini) - moderate
    tier3_target_ratio: float = 0.05  # 5% to Tier 3 (GPT-4) - complex reasoning

    # Complexity thresholds for tier routing
    # Tier 1: complexity <= tier1_max_complexity
    # Tier 2: tier1_max_complexity < complexity <= tier2_max_complexity
    # Tier 3: complexity > tier2_max_complexity
    tier1_max_complexity: float = 0.4   # Route to Tier 1 if complexity <= this
    tier2_max_complexity: float = 0.75  # Route to Tier 2 if complexity <= this

    # Confidence thresholds
    uncertain_threshold: float = 0.6    # Below this = uncertain

    # MUST: When uncertain, route to higher tier (prefer quality over cost)
    route_uncertain_to_higher_tier: bool = True
    uncertain_tier_bump: int = 1  # How many tiers to bump up when uncertain

    # Cost configuration (per 1K tokens)
    tier1_cost_per_1k: float = 0.002   # Tier 1: GPT-3.5 (cheapest)
    tier2_cost_per_1k: float = 0.015   # Tier 2: GPT-4o-mini (moderate)
    tier3_cost_per_1k: float = 0.03    # Tier 3: GPT-4 (expensive)
    local_cost_per_1k: float = 0.0     # Local fallback (free)

    # Model availability
    tier1_available: bool = True   # GPT-3.5
    tier2_available: bool = True   # GPT-4o-mini
    tier3_available: bool = True   # GPT-4
    local_model_available: bool = True  # Fallback option

    # Latency thresholds (ms)
    low_latency_threshold: float = 500    # Need response in <500ms

    # A/B testing
    ab_test_enabled: bool = False
    ab_test_tier3_percentage: float = 0.05  # 5% to Tier 3 for comparison

    # Legacy aliases for backward compatibility
    @property
    def gpt35_target_ratio(self) -> float:
        return self.tier1_target_ratio

    @property
    def gpt4_target_ratio(self) -> float:
        return self.tier3_target_ratio

    @property
    def simple_max_complexity(self) -> float:
        return self.tier1_max_complexity

    @simple_max_complexity.setter
    def simple_max_complexity(self, value: float) -> None:
        self.tier1_max_complexity = value

    @property
    def gpt4_min_complexity(self) -> float:
        return self.tier2_max_complexity

    @gpt4_min_complexity.setter
    def gpt4_min_complexity(self, value: float) -> None:
        self.tier2_max_complexity = value

    @property
    def route_uncertain_to_powerful(self) -> bool:
        return self.route_uncertain_to_higher_tier

    @property
    def gpt35_available(self) -> bool:
        return self.tier1_available

    @property
    def gpt4_available(self) -> bool:
        return self.tier3_available

    @property
    def gpt35_cost_per_1k(self) -> float:
        return self.tier1_cost_per_1k

    @property
    def gpt4_cost_per_1k(self) -> float:
        return self.tier3_cost_per_1k


@dataclass
class RoutingStats:
    """
    Statistics for 3-tier routing decisions.

    Tracks distribution across tiers to monitor adherence to 70/25/5 target.
    """

    total_queries: int = 0

    # Per-tier counts
    tier1_queries: int = 0   # Tier 1: Simple model (target: 70%)
    tier2_queries: int = 0   # Tier 2: Medium model (target: 25%)
    tier3_queries: int = 0   # Tier 3: Best model (target: 5%)
    local_queries: int = 0   # Local/fallback queries
    fallback_queries: int = 0

    total_cost: float = 0.0
    estimated_savings: float = 0.0  # Savings vs routing all to Tier 3

    # Legacy aliases
    @property
    def gpt35_queries(self) -> int:
        return self.tier1_queries

    @property
    def gpt4_queries(self) -> int:
        return self.tier3_queries

    @property
    def tier1_ratio(self) -> float:
        """Ratio of queries routed to Tier 1 (target: 70%)."""
        if self.total_queries == 0:
            return 0.0
        return self.tier1_queries / self.total_queries

    @property
    def tier2_ratio(self) -> float:
        """Ratio of queries routed to Tier 2 (target: 25%)."""
        if self.total_queries == 0:
            return 0.0
        return self.tier2_queries / self.total_queries

    @property
    def tier3_ratio(self) -> float:
        """Ratio of queries routed to Tier 3 (target: 5%)."""
        if self.total_queries == 0:
            return 0.0
        return self.tier3_queries / self.total_queries

    # Legacy property aliases
    @property
    def gpt35_ratio(self) -> float:
        return self.tier1_ratio

    @property
    def gpt4_ratio(self) -> float:
        return self.tier3_ratio

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            # Tier-based stats
            "tier1_queries": self.tier1_queries,
            "tier2_queries": self.tier2_queries,
            "tier3_queries": self.tier3_queries,
            "tier1_ratio": self.tier1_ratio,
            "tier2_ratio": self.tier2_ratio,
            "tier3_ratio": self.tier3_ratio,
            # Cost stats
            "total_cost": self.total_cost,
            "estimated_savings": self.estimated_savings,
            # Legacy
            "gpt35_queries": self.gpt35_queries,
            "gpt4_queries": self.gpt4_queries,
        }


class QueryRouter:
    """
    3-Tier Query Router for model selection.

    MODEL ROUTING PATTERN:
    - Tier 1 (70%): Simple model (GPT-3.5) for straightforward Q&A
    - Tier 2 (25%): Medium model (GPT-4o-mini) for moderate complexity
    - Tier 3 (5%): Best model (GPT-4) for complex reasoning

    FOCUS: Route queries to appropriate model tier based on complexity
    MUST: When uncertain, route to higher tier (prefer quality over cost)
    EXPECTED: Cost optimization with quality balance

    Usage:
        router = QueryRouter(classifier)
        decision = router.route(query)

        if decision.tier == ModelTier.TIER_1:
            response = await gpt35.generate(query)      # Simple Q&A
        elif decision.tier == ModelTier.TIER_2:
            response = await gpt4_mini.generate(query)  # Moderate complexity
        else:
            response = await gpt4.generate(query)       # Complex reasoning
    """

    def __init__(
        self,
        classifier: Optional[QueryClassifier] = None,
        config: Optional[RouterConfig] = None,
    ):
        """
        Initialize 3-tier query router.

        Args:
            classifier: Query classifier instance
            config: Router configuration
        """
        self.classifier = classifier or QueryClassifier()
        self.config = config or RouterConfig()

        # Statistics tracking
        self._stats = RoutingStats()

        logger.info(
            f"Initialized 3-Tier QueryRouter: "
            f"Tier1={self.config.tier1_target_ratio:.0%}, "
            f"Tier2={self.config.tier2_target_ratio:.0%}, "
            f"Tier3={self.config.tier3_target_ratio:.0%}"
        )
    
    def route(
        self,
        query: str,
        classification: Optional[ClassificationResult] = None,
        context: Optional[dict] = None,
    ) -> RoutingDecision:
        """
        Route a query to appropriate model tier.

        3-Tier Model Routing Pattern:
        - Tier 1 (70%): Simple model for straightforward Q&A
        - Tier 2 (25%): Medium model for moderate complexity
        - Tier 3 (5%): Best model for complex reasoning

        Args:
            query: Query to route
            classification: Optional pre-computed classification
            context: Optional context (user preferences, etc.)

        Returns:
            RoutingDecision with selected model, tier, and reasoning
        """
        start_time = time.time()

        # Get classification if not provided
        if classification is None:
            classification = self.classifier.classify(query)

        # Determine tier and route based on classification
        tier, route, reason, confidence = self._determine_route(classification, context)

        # Check model availability and apply fallback if needed
        route, tier, is_fallback = self._apply_availability_check(route, tier)

        # Estimate cost
        estimated_cost = self._estimate_cost(query, route)

        latency_ms = (time.time() - start_time) * 1000

        decision = RoutingDecision(
            route=route,
            tier=tier,
            reason=reason,
            confidence=confidence,
            query=query,
            category=classification.category,
            complexity_score=classification.complexity_score,
            estimated_cost=estimated_cost,
            latency_ms=latency_ms,
            fallback_route=self._get_fallback_route(route),
            is_fallback=is_fallback,
        )

        # Update statistics
        self._update_stats(decision)

        logger.debug(
            f"Routed query to {tier.value}/{route.value}: "
            f"category={classification.category.value}, "
            f"complexity={classification.complexity_score:.2f}, "
            f"reason={reason.value}"
        )

        return decision
    
    def _determine_route(
        self,
        classification: ClassificationResult,
        context: Optional[dict] = None,
    ) -> tuple[ModelTier, ModelRoute, RoutingReason, float]:
        """
        Determine tier and route based on classification and context.

        3-Tier Model Routing Pattern:
        - Tier 1 (70%): complexity <= 0.4 → Simple Q&A
        - Tier 2 (25%): 0.4 < complexity <= 0.75 → Moderate complexity
        - Tier 3 (5%): complexity > 0.75 → Complex reasoning
        """
        category = classification.category
        complexity = classification.complexity_score
        confidence = classification.confidence

        # Check for user preference in context (explicit tier or model)
        if context:
            if context.get("preferred_tier"):
                tier = ModelTier(context["preferred_tier"])
                return tier, TIER_TO_MODEL[tier], RoutingReason.USER_PREFERENCE, 0.95

            if context.get("preferred_model"):
                preferred = context["preferred_model"]
                if preferred == "gpt-4":
                    return ModelTier.TIER_3, ModelRoute.GPT4, RoutingReason.USER_PREFERENCE, 0.95
                elif preferred == "gpt-4o-mini":
                    return ModelTier.TIER_2, ModelRoute.GPT4_MINI, RoutingReason.USER_PREFERENCE, 0.95
                elif preferred == "gpt-3.5-turbo":
                    return ModelTier.TIER_1, ModelRoute.GPT35, RoutingReason.USER_PREFERENCE, 0.95

        # Check for latency requirements - use Tier 1 for low latency
        if context and context.get("max_latency_ms"):
            if context["max_latency_ms"] < self.config.low_latency_threshold:
                return ModelTier.TIER_1, ModelRoute.GPT35, RoutingReason.LATENCY_REQUIREMENT, 0.9

        # Route out-of-scope queries to Tier 1 (just explain it's out of scope)
        if category == QueryCategory.OUT_OF_SCOPE:
            return ModelTier.TIER_1, ModelRoute.GPT35, RoutingReason.OUT_OF_SCOPE, 0.9

        # Route FAQ queries to Tier 1 (straightforward cached answers)
        if category == QueryCategory.FAQ:
            return ModelTier.TIER_1, ModelRoute.GPT35, RoutingReason.FAQ_QUERY, 0.9

        # Determine base tier from complexity score
        base_tier, reason = self._complexity_to_tier(complexity, category)

        # MUST: When uncertain, route to higher tier (prefer quality over cost)
        if confidence < self.config.uncertain_threshold:
            if self.config.route_uncertain_to_higher_tier:
                bumped_tier = self._bump_tier(base_tier, self.config.uncertain_tier_bump)
                return bumped_tier, TIER_TO_MODEL[bumped_tier], RoutingReason.UNCERTAIN, 0.7

        return base_tier, TIER_TO_MODEL[base_tier], reason, 0.85

    def _complexity_to_tier(
        self,
        complexity: float,
        category: QueryCategory,
    ) -> tuple[ModelTier, RoutingReason]:
        """
        Map complexity score to model tier.

        Thresholds configured to achieve target distribution:
        - Tier 1 (70%): complexity <= tier1_max_complexity (0.4)
        - Tier 2 (25%): tier1_max < complexity <= tier2_max (0.75)
        - Tier 3 (5%): complexity > tier2_max_complexity
        """
        if complexity <= self.config.tier1_max_complexity:
            # Tier 1: Simple queries (70%)
            return ModelTier.TIER_1, RoutingReason.SIMPLE_QUERY

        elif complexity <= self.config.tier2_max_complexity:
            # Tier 2: Moderate complexity (25%)
            # Use category as secondary signal for edge cases
            if category == QueryCategory.COMPLEX:
                # If classified as complex but moderate score, bump to Tier 2
                return ModelTier.TIER_2, RoutingReason.MODERATE_QUERY
            elif category == QueryCategory.SIMPLE:
                # If classified as simple but moderate score, stay in Tier 2
                return ModelTier.TIER_2, RoutingReason.MODERATE_QUERY
            else:
                return ModelTier.TIER_2, RoutingReason.MODERATE_QUERY

        else:
            # Tier 3: Complex reasoning (5%)
            return ModelTier.TIER_3, RoutingReason.COMPLEX_QUERY

    def _bump_tier(self, tier: ModelTier, levels: int = 1) -> ModelTier:
        """Bump tier up by specified levels (for uncertain queries)."""
        tier_order = [ModelTier.TIER_1, ModelTier.TIER_2, ModelTier.TIER_3]
        current_idx = tier_order.index(tier)
        new_idx = min(current_idx + levels, len(tier_order) - 1)
        return tier_order[new_idx]
    
    def _apply_availability_check(
        self,
        route: ModelRoute,
        tier: ModelTier,
    ) -> tuple[ModelRoute, ModelTier, bool]:
        """Check model availability and apply fallback if needed."""

        # Tier 1 (GPT-3.5) unavailable
        if route == ModelRoute.GPT35 and not self.config.tier1_available:
            logger.warning("Tier 1 (GPT-3.5) unavailable, falling back to Tier 2")
            return ModelRoute.GPT4_MINI, ModelTier.TIER_2, True

        # Tier 2 (GPT-4o-mini) unavailable
        if route == ModelRoute.GPT4_MINI and not self.config.tier2_available:
            # Fall back to Tier 1 first (cost optimization)
            if self.config.tier1_available:
                logger.warning("Tier 2 (GPT-4o-mini) unavailable, falling back to Tier 1")
                return ModelRoute.GPT35, ModelTier.TIER_1, True
            # If Tier 1 also unavailable, go to Tier 3
            elif self.config.tier3_available:
                logger.warning("Tier 2 unavailable, Tier 1 unavailable, falling back to Tier 3")
                return ModelRoute.GPT4, ModelTier.TIER_3, True

        # Tier 3 (GPT-4) unavailable
        if route == ModelRoute.GPT4 and not self.config.tier3_available:
            # Fall back to Tier 2
            if self.config.tier2_available:
                logger.warning("Tier 3 (GPT-4) unavailable, falling back to Tier 2")
                return ModelRoute.GPT4_MINI, ModelTier.TIER_2, True
            # If Tier 2 also unavailable, go to Tier 1
            elif self.config.tier1_available:
                logger.warning("Tier 3 unavailable, Tier 2 unavailable, falling back to Tier 1")
                return ModelRoute.GPT35, ModelTier.TIER_1, True

        # Local model unavailable
        if route == ModelRoute.LOCAL and not self.config.local_model_available:
            logger.warning("Local model unavailable, falling back to Tier 1")
            return ModelRoute.GPT35, ModelTier.TIER_1, True

        return route, tier, False
    
    def _get_fallback_route(self, primary_route: ModelRoute) -> ModelRoute:
        """Get fallback route for a primary route (prefer adjacent tier)."""
        fallbacks = {
            # Tier 1 → Tier 2 (bump up if simple model fails)
            ModelRoute.GPT35: ModelRoute.GPT4_MINI,
            # Tier 2 → Tier 1 (fall back to cheaper if medium fails)
            ModelRoute.GPT4_MINI: ModelRoute.GPT35,
            # Tier 3 → Tier 2 (fall back to medium if best fails)
            ModelRoute.GPT4: ModelRoute.GPT4_MINI,
            # Local → Tier 1
            ModelRoute.LOCAL: ModelRoute.GPT35,
        }
        return fallbacks.get(primary_route, ModelRoute.GPT35)
    
    def _estimate_cost(self, query: str, route: ModelRoute) -> float:
        """Estimate cost for processing this query."""
        # Rough token estimate (words * 1.3)
        estimated_tokens = len(query.split()) * 1.3
        estimated_tokens += 500  # Add for response

        # Cost per 1K tokens by tier
        cost_per_1k = {
            ModelRoute.LOCAL: self.config.local_cost_per_1k,
            ModelRoute.GPT35: self.config.tier1_cost_per_1k,      # Tier 1
            ModelRoute.GPT4_MINI: self.config.tier2_cost_per_1k,  # Tier 2
            ModelRoute.GPT4: self.config.tier3_cost_per_1k,       # Tier 3
        }

        rate = cost_per_1k.get(route, self.config.tier1_cost_per_1k)
        return (estimated_tokens / 1000) * rate
    
    def _update_stats(self, decision: RoutingDecision) -> None:
        """Update routing statistics for 3-tier tracking."""
        self._stats.total_queries += 1

        # Track by tier
        if decision.tier == ModelTier.TIER_1:
            self._stats.tier1_queries += 1
        elif decision.tier == ModelTier.TIER_2:
            self._stats.tier2_queries += 1
        elif decision.tier == ModelTier.TIER_3:
            self._stats.tier3_queries += 1

        # Track local separately
        if decision.route == ModelRoute.LOCAL:
            self._stats.local_queries += 1

        if decision.is_fallback:
            self._stats.fallback_queries += 1

        self._stats.total_cost += decision.estimated_cost

        # Calculate savings (compared to routing all to Tier 3)
        tier3_cost = self._estimate_cost(decision.query, ModelRoute.GPT4)
        self._stats.estimated_savings += tier3_cost - decision.estimated_cost
    
    async def route_async(
        self,
        query: str,
        classification: Optional[ClassificationResult] = None,
        context: Optional[dict] = None,
    ) -> RoutingDecision:
        """Async version of route."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.route(query, classification, context)
        )
    
    def route_batch(
        self,
        queries: list[str],
        context: Optional[dict] = None,
    ) -> list[RoutingDecision]:
        """Route multiple queries."""
        return [self.route(q, context=context) for q in queries]
    
    def get_tier(self, query: str) -> ModelTier:
        """Quick check to get the tier for a query."""
        decision = self.route(query)
        return decision.tier

    def should_use_tier1(self, query: str) -> bool:
        """Quick check if query should use Tier 1 (simple model)."""
        return self.get_tier(query) == ModelTier.TIER_1

    def should_use_tier2(self, query: str) -> bool:
        """Quick check if query should use Tier 2 (medium model)."""
        return self.get_tier(query) == ModelTier.TIER_2

    def should_use_tier3(self, query: str) -> bool:
        """Quick check if query should use Tier 3 (best model)."""
        return self.get_tier(query) == ModelTier.TIER_3

    # Legacy aliases
    def should_use_simple_model(self, query: str) -> bool:
        """Quick check if query should use simple model (Tier 1)."""
        return self.should_use_tier1(query)

    def should_use_complex_model(self, query: str) -> bool:
        """Quick check if query should use complex model (Tier 3)."""
        return self.should_use_tier3(query)

    def get_stats(self) -> RoutingStats:
        """Get routing statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._stats = RoutingStats()

    def get_routing_distribution(self) -> dict[str, float]:
        """Get current 3-tier routing distribution vs targets."""
        return {
            # Current distribution
            "tier1": self._stats.tier1_ratio,
            "tier2": self._stats.tier2_ratio,
            "tier3": self._stats.tier3_ratio,
            # Target distribution
            "target_tier1": self.config.tier1_target_ratio,
            "target_tier2": self.config.tier2_target_ratio,
            "target_tier3": self.config.tier3_target_ratio,
            # Legacy aliases
            "gpt35": self._stats.tier1_ratio,
            "gpt4": self._stats.tier3_ratio,
            "target_gpt35": self.config.tier1_target_ratio,
            "target_gpt4": self.config.tier3_target_ratio,
        }
    
    def update_config(
        self,
        tier1_max_complexity: Optional[float] = None,
        tier2_max_complexity: Optional[float] = None,
        uncertain_threshold: Optional[float] = None,
        # Legacy aliases
        simple_max_complexity: Optional[float] = None,
        gpt4_min_complexity: Optional[float] = None,
    ) -> None:
        """
        Update 3-tier routing configuration dynamically.

        Useful for A/B testing or adaptive routing.

        Args:
            tier1_max_complexity: Max complexity for Tier 1 routing
            tier2_max_complexity: Max complexity for Tier 2 routing
            uncertain_threshold: Confidence threshold for uncertain queries
        """
        # Handle legacy parameter names
        if simple_max_complexity is not None:
            tier1_max_complexity = simple_max_complexity
        if gpt4_min_complexity is not None:
            tier2_max_complexity = gpt4_min_complexity

        if tier1_max_complexity is not None:
            self.config.tier1_max_complexity = tier1_max_complexity
        if tier2_max_complexity is not None:
            self.config.tier2_max_complexity = tier2_max_complexity
        if uncertain_threshold is not None:
            self.config.uncertain_threshold = uncertain_threshold

        logger.info(
            f"Updated 3-tier router config: "
            f"tier1_max={self.config.tier1_max_complexity}, "
            f"tier2_max={self.config.tier2_max_complexity}, "
            f"uncertain={self.config.uncertain_threshold}"
        )


class AdaptiveRouter(QueryRouter):
    """
    Adaptive 3-tier router that adjusts thresholds based on performance.

    Monitors routing outcomes and adjusts to meet target distribution:
    - Tier 1 (70%): Simple model for straightforward Q&A
    - Tier 2 (25%): Medium model for moderate complexity
    - Tier 3 (5%): Best model for complex reasoning
    """

    def __init__(
        self,
        classifier: Optional[QueryClassifier] = None,
        config: Optional[RouterConfig] = None,
        adjustment_interval: int = 100,  # Adjust every N queries
    ):
        super().__init__(classifier, config)
        self.adjustment_interval = adjustment_interval
        self._queries_since_adjustment = 0

    def route(
        self,
        query: str,
        classification: Optional[ClassificationResult] = None,
        context: Optional[dict] = None,
    ) -> RoutingDecision:
        """Route with adaptive adjustment."""
        decision = super().route(query, classification, context)

        self._queries_since_adjustment += 1

        # Check if we need to adjust
        if self._queries_since_adjustment >= self.adjustment_interval:
            self._adjust_thresholds()
            self._queries_since_adjustment = 0

        return decision

    def _adjust_thresholds(self) -> None:
        """Adjust thresholds to meet 3-tier target distribution (70/25/5)."""
        current_tier1 = self._stats.tier1_ratio
        current_tier2 = self._stats.tier2_ratio
        current_tier3 = self._stats.tier3_ratio

        target_tier1 = self.config.tier1_target_ratio  # 0.70
        target_tier2 = self.config.tier2_target_ratio  # 0.25
        target_tier3 = self.config.tier3_target_ratio  # 0.05

        adjustment_step = 0.03

        # Adjust Tier 1 threshold (tier1_max_complexity)
        # If Tier 1 is below target, increase threshold to capture more queries
        if current_tier1 < target_tier1 - 0.05:
            self.config.tier1_max_complexity = min(
                0.6,  # Max threshold for Tier 1
                self.config.tier1_max_complexity + adjustment_step
            )
            logger.info(
                f"Increased tier1_max to {self.config.tier1_max_complexity:.2f} "
                f"(current={current_tier1:.1%}, target={target_tier1:.1%})"
            )
        # If Tier 1 is above target, decrease threshold
        elif current_tier1 > target_tier1 + 0.05:
            self.config.tier1_max_complexity = max(
                0.2,  # Min threshold for Tier 1
                self.config.tier1_max_complexity - adjustment_step
            )
            logger.info(
                f"Decreased tier1_max to {self.config.tier1_max_complexity:.2f} "
                f"(current={current_tier1:.1%}, target={target_tier1:.1%})"
            )

        # Adjust Tier 2 threshold (tier2_max_complexity)
        # If Tier 3 is above target (5%), lower Tier 2 max to push more to Tier 2
        if current_tier3 > target_tier3 + 0.02:
            self.config.tier2_max_complexity = min(
                0.9,  # Max threshold for Tier 2
                self.config.tier2_max_complexity + adjustment_step
            )
            logger.info(
                f"Increased tier2_max to {self.config.tier2_max_complexity:.2f} "
                f"(tier3: current={current_tier3:.1%}, target={target_tier3:.1%})"
            )
        # If Tier 3 is below target, raise Tier 2 max to allow more to Tier 3
        elif current_tier3 < target_tier3 - 0.02:
            self.config.tier2_max_complexity = max(
                self.config.tier1_max_complexity + 0.1,  # Must be above Tier 1 max
                self.config.tier2_max_complexity - adjustment_step
            )
            logger.info(
                f"Decreased tier2_max to {self.config.tier2_max_complexity:.2f} "
                f"(tier3: current={current_tier3:.1%}, target={target_tier3:.1%})"
            )


def create_query_router(
    classifier: Optional[QueryClassifier] = None,
    tier1_target_ratio: float = 0.70,
    tier2_target_ratio: float = 0.25,
    tier3_target_ratio: float = 0.05,
    adaptive: bool = False,
    # Legacy parameter
    gpt35_target_ratio: Optional[float] = None,
    **kwargs,
) -> QueryRouter:
    """
    Factory function to create 3-tier query router.

    Model Routing Pattern (default 70/25/5):
    - Tier 1: Simple model for straightforward Q&A
    - Tier 2: Medium model for moderate complexity
    - Tier 3: Best model for complex reasoning

    Args:
        classifier: Query classifier instance
        tier1_target_ratio: Target ratio for Tier 1 (default: 0.70)
        tier2_target_ratio: Target ratio for Tier 2 (default: 0.25)
        tier3_target_ratio: Target ratio for Tier 3 (default: 0.05)
        adaptive: Whether to use adaptive routing
        gpt35_target_ratio: Legacy alias for tier1_target_ratio
        **kwargs: Additional config options

    Returns:
        Configured QueryRouter instance
    """
    # Handle legacy parameter
    if gpt35_target_ratio is not None:
        tier1_target_ratio = gpt35_target_ratio
        # Distribute remaining between Tier 2 and 3 (5:1 ratio)
        remaining = 1.0 - tier1_target_ratio
        tier3_target_ratio = remaining / 6  # ~5% of total
        tier2_target_ratio = remaining - tier3_target_ratio  # ~25% of total

    config = RouterConfig(
        tier1_target_ratio=tier1_target_ratio,
        tier2_target_ratio=tier2_target_ratio,
        tier3_target_ratio=tier3_target_ratio,
        **kwargs,
    )

    if adaptive:
        return AdaptiveRouter(
            classifier=classifier,
            config=config,
        )

    return QueryRouter(
        classifier=classifier,
        config=config,
    )