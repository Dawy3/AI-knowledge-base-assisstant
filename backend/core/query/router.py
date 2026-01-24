"""
Model Routing Logic for RAG Pipeline.

FOCUS: Simple model for easy, hard model for complex
MUST: When uncertain, route to better model
EXPECTED: Cost optimization with quality balance

Intelligent routing optimizes:
1. Cost - Route simple queries to cheaper models (GPT-3.5)
2. Quality - Route complex queries to powerful models (GPT-4)
3. Latency - Route time-sensitive queries to faster models
4. Reliability - Fallback routing on model failures

Routing strategy:
- Simple/FAQ queries → GPT-3.5 (fast, cheap)
- Complex queries → GPT-4 (expensive but high quality)
- Uncertain → GPT-4 (prefer quality over cost when uncertain)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .classifier import ClassificationResult, QueryCategory, QueryClassifier

logger = logging.getLogger(__name__)


class ModelRoute(str, Enum):
    """Available model routes."""
    
    LOCAL = "local"              # Local fine-tuned model
    GPT4 = "gpt-4"               # GPT-4 for complex queries
    GPT35 = "gpt-3.5-turbo"      # GPT-3.5 for medium complexity
    CLAUDE = "claude"            # Claude as alternative
    FALLBACK = "fallback"        # Fallback model


class RoutingReason(str, Enum):
    """Reason for routing decision."""
    
    SIMPLE_QUERY = "simple_query"
    COMPLEX_QUERY = "complex_query"
    FAQ_QUERY = "faq_query"
    UNCERTAIN = "uncertain"
    OUT_OF_SCOPE = "out_of_scope"
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
    
    # Original classification
    query: str
    category: QueryCategory
    complexity_score: float
    
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
    """Configuration for query router."""

    # Target routing distribution
    gpt35_target_ratio: float = 0.70  # TARGET: 70% to GPT-3.5 (simple queries)
    gpt4_target_ratio: float = 0.30   # TARGET: 30% to GPT-4 (complex queries)

    # Complexity thresholds
    simple_max_complexity: float = 0.6   # Route to GPT-3.5 if complexity <= this
    gpt4_min_complexity: float = 0.7     # Route to GPT-4 if complexity >= this
    
    # Confidence thresholds
    uncertain_threshold: float = 0.6    # Below this = uncertain
    
    # MUST: When uncertain, route to better model
    route_uncertain_to_powerful: bool = True

    # Cost configuration (per 1K tokens)
    gpt35_cost_per_1k: float = 0.002    # Simple model (cheap)
    gpt4_cost_per_1k: float = 0.03      # Complex model (expensive)
    local_cost_per_1k: float = 0.0      # Local fallback

    # Model availability
    gpt35_available: bool = True
    gpt4_available: bool = True
    local_model_available: bool = True  # Fallback option
    
    # Latency thresholds (ms)
    low_latency_threshold: float = 500    # Need response in <500ms
    
    # A/B testing
    ab_test_enabled: bool = False
    ab_test_gpt4_percentage: float = 0.1  # 10% of local queries to GPT-4 for comparison


@dataclass
class RoutingStats:
    """Statistics for routing decisions."""

    total_queries: int = 0
    gpt35_queries: int = 0   # Simple model queries
    gpt4_queries: int = 0    # Complex model queries
    local_queries: int = 0   # Fallback queries
    fallback_queries: int = 0

    total_cost: float = 0.0
    estimated_savings: float = 0.0

    @property
    def gpt35_ratio(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.gpt35_queries / self.total_queries

    @property
    def gpt4_ratio(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.gpt4_queries / self.total_queries

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "gpt35_queries": self.gpt35_queries,
            "gpt4_queries": self.gpt4_queries,
            "gpt35_ratio": self.gpt35_ratio,
            "gpt4_ratio": self.gpt4_ratio,
            "total_cost": self.total_cost,
            "estimated_savings": self.estimated_savings,
        }


class QueryRouter:
    """
    Query router for model selection.

    FOCUS: Simple model for easy, hard model for complex
    MUST: When uncertain, route to better model
    EXPECTED: Cost optimization with quality balance

    Usage:
        router = QueryRouter(classifier)
        decision = router.route(query)

        if decision.route == ModelRoute.GPT35:
            response = await gpt35.generate(query)  # Simple queries
        else:
            response = await gpt4.generate(query)   # Complex queries
    """
    
    def __init__(
        self,
        classifier: Optional[QueryClassifier] = None,
        config: Optional[RouterConfig] = None,
    ):
        """
        Initialize query router.
        
        Args:
            classifier: Query classifier instance
            config: Router configuration
        """
        self.classifier = classifier or QueryClassifier()
        self.config = config or RouterConfig()
        
        # Statistics tracking
        self._stats = RoutingStats()

        logger.info(
            f"Initialized QueryRouter: "
            f"gpt35_target={self.config.gpt35_target_ratio:.0%}, "
            f"gpt4_target={self.config.gpt4_target_ratio:.0%}"
        )
    
    def route(
        self,
        query: str,
        classification: Optional[ClassificationResult] = None,
        context: Optional[dict] = None,
    ) -> RoutingDecision:
        """
        Route a query to appropriate model.

        FOCUS: Simple model (GPT-3.5) for easy, hard model (GPT-4) for complex

        Args:
            query: Query to route
            classification: Optional pre-computed classification
            context: Optional context (user preferences, etc.)

        Returns:
            RoutingDecision with selected model and reasoning
        """
        start_time = time.time()
        
        # Get classification if not provided
        if classification is None:
            classification = self.classifier.classify(query)
        
        # Determine route based on classification
        route, reason, confidence = self._determine_route(classification, context)
        
        # Check model availability and apply fallback if needed
        route, is_fallback = self._apply_availability_check(route)
        
        # Estimate cost
        estimated_cost = self._estimate_cost(query, route)
        
        latency_ms = (time.time() - start_time) * 1000
        
        decision = RoutingDecision(
            route=route,
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
            f"Routed query to {route.value}: "
            f"category={classification.category.value}, "
            f"complexity={classification.complexity_score:.2f}, "
            f"reason={reason.value}"
        )
        
        return decision
    
    def _determine_route(
        self,
        classification: ClassificationResult,
        context: Optional[dict] = None,
    ) -> tuple[ModelRoute, RoutingReason, float]:
        """
        Determine route based on classification and context.

        Simple queries → GPT-3.5 (easy model)
        Complex queries → GPT-4 (hard model)
        """
        category = classification.category
        complexity = classification.complexity_score
        confidence = classification.confidence

        # Check for user preference in context
        if context and context.get("preferred_model"):
            preferred = context["preferred_model"]
            if preferred == "gpt-4":
                return ModelRoute.GPT4, RoutingReason.USER_PREFERENCE, 0.95
            elif preferred == "gpt-3.5-turbo":
                return ModelRoute.GPT35, RoutingReason.USER_PREFERENCE, 0.95

        # Check for latency requirements - use simple model for low latency
        if context and context.get("max_latency_ms"):
            if context["max_latency_ms"] < self.config.low_latency_threshold:
                return ModelRoute.GPT35, RoutingReason.LATENCY_REQUIREMENT, 0.9

        # Route out-of-scope queries to simple model (just explain it's out of scope)
        if category == QueryCategory.OUT_OF_SCOPE:
            return ModelRoute.GPT35, RoutingReason.OUT_OF_SCOPE, 0.9

        # Route FAQ queries to simple model (straightforward answers)
        if category == QueryCategory.FAQ:
            return ModelRoute.GPT35, RoutingReason.FAQ_QUERY, 0.9

        # MUST: When uncertain, route to better model
        if confidence < self.config.uncertain_threshold:
            if self.config.route_uncertain_to_powerful:
                return ModelRoute.GPT4, RoutingReason.UNCERTAIN, 0.7

        # Route by complexity: simple → GPT-3.5, complex → GPT-4
        if complexity <= self.config.simple_max_complexity:
            # Simple queries → GPT-3.5 (easy model)
            return ModelRoute.GPT35, RoutingReason.SIMPLE_QUERY, 0.85

        elif complexity >= self.config.gpt4_min_complexity:
            # Complex queries → GPT-4 (hard model)
            return ModelRoute.GPT4, RoutingReason.COMPLEX_QUERY, 0.85

        else:
            # Medium complexity - route based on category
            if category == QueryCategory.SIMPLE:
                return ModelRoute.GPT35, RoutingReason.SIMPLE_QUERY, 0.75
            elif category == QueryCategory.COMPLEX:
                return ModelRoute.GPT4, RoutingReason.COMPLEX_QUERY, 0.75
            else:
                # Default to cost optimization (GPT-3.5)
                return ModelRoute.GPT35, RoutingReason.COST_OPTIMIZATION, 0.7
    
    def _apply_availability_check(
        self,
        route: ModelRoute,
    ) -> tuple[ModelRoute, bool]:
        """Check model availability and apply fallback if needed."""

        if route == ModelRoute.GPT35 and not self.config.gpt35_available:
            logger.warning("GPT-3.5 unavailable, falling back to GPT-4")
            return ModelRoute.GPT4, True

        if route == ModelRoute.GPT4 and not self.config.gpt4_available:
            logger.warning("GPT-4 unavailable, falling back to GPT-3.5")
            return ModelRoute.GPT35, True

        if route == ModelRoute.LOCAL and not self.config.local_model_available:
            logger.warning("Local model unavailable, falling back to GPT-3.5")
            return ModelRoute.GPT35, True

        return route, False
    
    def _get_fallback_route(self, primary_route: ModelRoute) -> ModelRoute:
        """Get fallback route for a primary route."""
        fallbacks = {
            ModelRoute.GPT35: ModelRoute.GPT4,   # Simple → Hard as fallback
            ModelRoute.GPT4: ModelRoute.GPT35,   # Hard → Simple as fallback
            ModelRoute.LOCAL: ModelRoute.GPT35,  # Local → Simple as fallback
        }
        return fallbacks.get(primary_route, ModelRoute.GPT35)
    
    def _estimate_cost(self, query: str, route: ModelRoute) -> float:
        """Estimate cost for processing this query."""
        # Rough token estimate (words * 1.3)
        estimated_tokens = len(query.split()) * 1.3
        estimated_tokens += 500  # Add for response
        
        # Cost per 1K tokens
        cost_per_1k = {
            ModelRoute.LOCAL: self.config.local_cost_per_1k,
            ModelRoute.GPT4: self.config.gpt4_cost_per_1k,
            ModelRoute.GPT35: self.config.gpt35_cost_per_1k,
        }
        
        rate = cost_per_1k.get(route, self.config.gpt35_cost_per_1k)
        return (estimated_tokens / 1000) * rate
    
    def _update_stats(self, decision: RoutingDecision) -> None:
        """Update routing statistics."""
        self._stats.total_queries += 1

        if decision.route == ModelRoute.GPT35:
            self._stats.gpt35_queries += 1
        elif decision.route == ModelRoute.GPT4:
            self._stats.gpt4_queries += 1
        elif decision.route == ModelRoute.LOCAL:
            self._stats.local_queries += 1

        if decision.is_fallback:
            self._stats.fallback_queries += 1

        self._stats.total_cost += decision.estimated_cost

        # Calculate savings (compared to routing all to GPT-4)
        gpt4_cost = self._estimate_cost(decision.query, ModelRoute.GPT4)
        self._stats.estimated_savings += gpt4_cost - decision.estimated_cost
    
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
    
    def should_use_simple_model(self, query: str) -> bool:
        """Quick check if query should use simple model (GPT-3.5)."""
        decision = self.route(query)
        return decision.route == ModelRoute.GPT35

    def should_use_complex_model(self, query: str) -> bool:
        """Quick check if query should use complex model (GPT-4)."""
        decision = self.route(query)
        return decision.route == ModelRoute.GPT4
    
    def get_stats(self) -> RoutingStats:
        """Get routing statistics."""
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._stats = RoutingStats()
    
    def get_routing_distribution(self) -> dict[str, float]:
        """Get current routing distribution."""
        return {
            "gpt35": self._stats.gpt35_ratio,
            "gpt4": self._stats.gpt4_ratio,
            "target_gpt35": self.config.gpt35_target_ratio,
            "target_gpt4": self.config.gpt4_target_ratio,
        }
    
    def update_config(
        self,
        simple_max_complexity: Optional[float] = None,
        gpt4_min_complexity: Optional[float] = None,
        uncertain_threshold: Optional[float] = None,
    ) -> None:
        """
        Update routing configuration dynamically.

        Useful for A/B testing or adaptive routing.
        """
        if simple_max_complexity is not None:
            self.config.simple_max_complexity = simple_max_complexity
        if gpt4_min_complexity is not None:
            self.config.gpt4_min_complexity = gpt4_min_complexity
        if uncertain_threshold is not None:
            self.config.uncertain_threshold = uncertain_threshold

        logger.info(
            f"Updated router config: "
            f"simple_max={self.config.simple_max_complexity}, "
            f"gpt4_min={self.config.gpt4_min_complexity}, "
            f"uncertain={self.config.uncertain_threshold}"
        )


class AdaptiveRouter(QueryRouter):
    """
    Adaptive router that adjusts thresholds based on performance.

    Monitors routing outcomes and adjusts to meet target distribution.
    Balances between simple model (GPT-3.5) and complex model (GPT-4).
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
        """Adjust thresholds to meet target distribution."""
        current_gpt35 = self._stats.gpt35_ratio
        target_gpt35 = self.config.gpt35_target_ratio

        # If routing too many to GPT-4, increase simple model threshold
        if current_gpt35 < target_gpt35 - 0.05:
            self.config.simple_max_complexity = min(
                0.9,
                self.config.simple_max_complexity + 0.05
            )
            logger.info(
                f"Adjusted simple_max_complexity to {self.config.simple_max_complexity} "
                f"(current_gpt35={current_gpt35:.2%}, target={target_gpt35:.2%})"
            )

        # If routing too many to GPT-3.5, decrease simple model threshold
        elif current_gpt35 > target_gpt35 + 0.05:
            self.config.simple_max_complexity = max(
                0.3,
                self.config.simple_max_complexity - 0.05
            )
            logger.info(
                f"Adjusted simple_max_complexity to {self.config.simple_max_complexity} "
                f"(current_gpt35={current_gpt35:.2%}, target={target_gpt35:.2%})"
            )


def create_query_router(
    classifier: Optional[QueryClassifier] = None,
    gpt35_target_ratio: float = 0.70,
    adaptive: bool = False,
    **kwargs,
) -> QueryRouter:
    """
    Factory function to create query router.

    Args:
        classifier: Query classifier instance
        gpt35_target_ratio: Target ratio for simple model (GPT-3.5) routing
        adaptive: Whether to use adaptive routing
        **kwargs: Additional config options

    Returns:
        Configured QueryRouter instance
    """
    config = RouterConfig(
        gpt35_target_ratio=gpt35_target_ratio,
        gpt4_target_ratio=1.0 - gpt35_target_ratio,
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