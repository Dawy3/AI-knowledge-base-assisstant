"""
Query Classification for RAG Pipeline.

FOCUS: Route to appropriate handler
MUST: Simple/FAQ/Complex/Out-of-scope
TARGET: <12ms classification time
TARGET: >94% accuracy

Query classification enables:
1. Routing simple queries to faster/cheaper models
2. Detecting FAQ-like queries for caching
3. Identifying complex queries needing advanced retrieval
4. Filtering out-of-scope queries to avoid hallucination
"""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class QueryCategory(str, Enum):
    """Query categories for routing."""
    
    SIMPLE = "simple"            # Direct factual questions
    FAQ = "faq"                  # Common questions, likely cached
    COMPLEX = "complex"          # Multi-step reasoning needed
    CONVERSATIONAL = "conversational"  # Follow-up or chat-like
    OUT_OF_SCOPE = "out_of_scope"  # Not in knowledge domain
    AMBIGUOUS = "ambiguous"      # Needs clarification
    

class QueryIntent(str, Enum):
    """Detected query intent."""
    
    FACTUAL = "factual"          # Looking for facts
    PROCEDURAL = "procedural"    # How-to questions
    COMPARATIVE = "comparative"  # Comparing options
    EXPLORATORY = "exploratory"  # Open-ended exploration
    NAVIGATIONAL = "navigational"  # Looking for specific resource
    TROUBLESHOOTING = "troubleshooting"  # Problem solving
    DEFINITION = "definition"    # What is X?
    OPINION = "opinion"          # Seeking opinions/recommendations


@dataclass
class ClassificationResult:
    """Result of query classification."""
    
    query: str
    category: QueryCategory
    confidence: float  # 0-1 confidence score
    
    # Additional classification
    intent: Optional[QueryIntent] = None
    complexity_score: float = 0.5  # 0=simple, 1=complex
    
    # Domain detection
    is_in_domain: bool = True
    domain_confidence: float = 1.0
    detected_domain: Optional[str] = None
    
    # Metadata
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    
    # Performance
    latency_ms: float = 0.0
    
    # Routing hints
    suggested_model: Optional[str] = None
    use_cache: bool = False
    needs_clarification: bool = False
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "category": self.category.value,
            "confidence": self.confidence,
            "intent": self.intent.value if self.intent else None,
            "complexity_score": self.complexity_score,
            "is_in_domain": self.is_in_domain,
            "domain_confidence": self.domain_confidence,
            "detected_domain": self.detected_domain,
            "suggested_model": self.suggested_model,
            "use_cache": self.use_cache,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ClassifierConfig:
    """Configuration for query classifier."""
    
    # Complexity thresholds
    simple_max_words: int = 10
    complex_min_words: int = 25
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.85
    low_confidence_threshold: float = 0.5
    
    # Domain configuration
    domain_keywords: dict[str, list[str]] = field(default_factory=dict)
    out_of_scope_keywords: list[str] = field(default_factory=list)
    
    # FAQ detection
    faq_patterns: list[str] = field(default_factory=list)
    faq_similarity_threshold: float = 0.9
    
    # Performance targets
    target_latency_ms: float = 12.0  # TARGET: <12ms


class RuleBasedClassifier:
    """
    Rule-based query classifier.
    
    Fast classification using patterns and heuristics.
    TARGET: <12ms classification time
    """
    
    # Patterns for intent detection
    INTENT_PATTERNS = {
        QueryIntent.DEFINITION: [
            r"^what (?:is|are) ",
            r"^define ",
            r"^meaning of ",
            r"^explain what ",
        ],
        QueryIntent.PROCEDURAL: [
            r"^how (?:do|can|to|should) ",
            r"^steps to ",
            r"^way to ",
            r"^process (?:for|of|to) ",
            r"^guide (?:for|to) ",
        ],
        QueryIntent.COMPARATIVE: [
            r"(?:difference|compare|versus|vs\.?|better)",
            r"(?:which|what) (?:is|are) (?:better|best)",
            r"pros and cons",
        ],
        QueryIntent.TROUBLESHOOTING: [
            r"(?:error|issue|problem|bug|fail|broken|not working)",
            r"(?:fix|solve|resolve|debug|troubleshoot)",
            r"why (?:is|does|doesn't|won't)",
        ],
        QueryIntent.NAVIGATIONAL: [
            r"(?:where|find|locate|link|url|page)",
            r"(?:documentation|docs|guide) for",
        ],
    }
    
    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        "high": [
            r"\b(?:and|or|but)\b.*\b(?:and|or|but)\b",  # Multiple conjunctions
            r"\?.*\?",  # Multiple questions
            r"(?:first|then|after|before|finally)",  # Sequential
            r"(?:if|when|unless|assuming)",  # Conditional
            r"(?:compare|contrast|analyze|evaluate)",  # Analytical
        ],
        "low": [
            r"^(?:what|who|when|where) is ",  # Simple factual
            r"^(?:yes|no|true|false)\?",  # Boolean
            r"^define ",  # Definition
        ],
    }
    
    # Common FAQ patterns
    DEFAULT_FAQ_PATTERNS = [
        r"^how (?:do i|can i|to) (?:get started|begin|start)",
        r"^what (?:is|are) (?:the|your) (?:pricing|price|cost)",
        r"^how (?:much|many)",
        r"^(?:do you|can you|is there) (?:support|offer|have)",
        r"^what (?:are|is) (?:the|your) (?:features|capabilities)",
        r"^how (?:do i|can i) (?:contact|reach)",
        r"^(?:forgot|reset|change) (?:my )?password",
    ]
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()
        
        # Compile regex patterns
        self._intent_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        
        self._complexity_high = [
            re.compile(p, re.IGNORECASE)
            for p in self.COMPLEXITY_INDICATORS["high"]
        ]
        self._complexity_low = [
            re.compile(p, re.IGNORECASE)
            for p in self.COMPLEXITY_INDICATORS["low"]
        ]
        
        faq_patterns = self.config.faq_patterns or self.DEFAULT_FAQ_PATTERNS
        self._faq_patterns = [
            re.compile(p, re.IGNORECASE) for p in faq_patterns
        ]
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify query using rules and heuristics.
        
        TARGET: <12ms classification time
        """
        start_time = time.time()
        
        query = query.strip()
        query_lower = query.lower()
        
        # Extract basic features
        word_count = len(query.split())
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Calculate complexity
        complexity = self._calculate_complexity(query_lower, word_count)
        
        # Determine category
        category, confidence = self._determine_category(
            query_lower, word_count, complexity, intent
        )
        
        # Check domain
        is_in_domain, domain_confidence, detected_domain = self._check_domain(
            query_lower
        )
        
        # Override if out of domain
        if not is_in_domain and domain_confidence > 0.7:
            category = QueryCategory.OUT_OF_SCOPE
            confidence = domain_confidence
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Warn if exceeding target
        if latency_ms > self.config.target_latency_ms:
            logger.warning(
                f"Classification exceeded target latency: "
                f"{latency_ms:.2f}ms > {self.config.target_latency_ms}ms"
            )
        
        return ClassificationResult(
            query=query,
            category=category,
            confidence=confidence,
            intent=intent,
            complexity_score=complexity,
            is_in_domain=is_in_domain,
            domain_confidence=domain_confidence,
            detected_domain=detected_domain,
            use_cache=category == QueryCategory.FAQ,
            needs_clarification=category == QueryCategory.AMBIGUOUS,
            latency_ms=latency_ms,
        )
        
    def _detect_intent(self, query: str) -> Optional[QueryIntent]:
        """Detect query intent using patterns."""
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    return intent
        return QueryIntent.FACTUAL  # Default

    def _calculate_complexity(self, query: str, word_count: int) -> float:
        """Calculate query complexity score (0-1)."""
        complexity = 0.5  # Base complexity
        
        # Adjust by word count
        if word_count <= self.config.simple_max_words:
            complexity -= 0.2
        elif word_count >= self.config.complex_min_words:
            complexity += 0.2
        
        # Check complexity indicators
        for pattern in self._complexity_high:
            if pattern.search(query):
                complexity += 0.15
        
        for pattern in self._complexity_low:
            if pattern.search(query):
                complexity -= 0.15
        
        # Clamp to 0-1
        return max(0.0, min(1.0, complexity))
    
    def _determine_category(
        self,
        query: str,
        word_count: int,
        complexity: float,
        intent: Optional[QueryIntent],
    ) -> tuple[QueryCategory, float]:
        """Determine query category and confidence."""
        
        # Check FAQ patterns first
        for pattern in self._faq_patterns:
            if pattern.search(query):
                return QueryCategory.FAQ, 0.9
        
        # Categorize by complexity
        if complexity < 0.3:
            category = QueryCategory.SIMPLE
            confidence = 0.9 - complexity
        elif complexity > 0.7:
            category = QueryCategory.COMPLEX
            confidence = complexity
        else:
            # Medium complexity - look at other signals
            if intent in [QueryIntent.DEFINITION, QueryIntent.FACTUAL]:
                category = QueryCategory.SIMPLE
                confidence = 0.7
            elif intent in [QueryIntent.COMPARATIVE, QueryIntent.EXPLORATORY]:
                category = QueryCategory.COMPLEX
                confidence = 0.7
            else:
                category = QueryCategory.SIMPLE
                confidence = 0.6
        
        # Check for conversational markers
        conversational_markers = ["thanks", "thank you", "please", "could you", "can you help"]
        if any(marker in query for marker in conversational_markers):
            # Might be conversational but still classify by content
            pass
        
        # Very short queries might be ambiguous
        if word_count <= 2:
            return QueryCategory.AMBIGUOUS, 0.6
        
        return category, confidence

    def _check_domain(
        self,
        query: str,
    ) -> tuple[bool, float, Optional[str]]:
        """Check if query is in the configured domain."""
        
        # Check out-of-scope keywords
        for keyword in self.config.out_of_scope_keywords:
            if keyword.lower() in query:
                return False, 0.8, None
        
        # Check domain keywords
        for domain, keywords in self.config.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query:
                    return True, 0.9, domain
        
        # Default: assume in domain with lower confidence
        return True, 0.6, None


class EmbeddingBasedClassifier:
    """
    Embedding-based query classifier.
    
    Uses embeddings to classify queries by similarity to examples.
    More accurate but slightly higher latency than rule-based.
    """
    
    def __init__(
        self,
        embedding_func: Callable[[str], list[float]],
        category_examples: dict[QueryCategory, list[str]],
        config: Optional[ClassifierConfig] = None,
    ):
        self.embedding_func = embedding_func
        self.config = config or ClassifierConfig()
        self._category_embeddings: dict[QueryCategory, list[list[float]]] = {}
        
        # Pre-compute category embeddings
        self._initialize_embeddings(category_examples)
        
    def _initialize_embeddings(
        self,
        category_examples: dict[QueryCategory, list[str]],
    ) -> None:
        """Pre-compute embeddings for category examples."""
        for category, examples in category_examples.items():
            self._category_embeddings[category] = [
                self.embedding_func(example) for example in examples
            ]
            
            
    def classify(self, query: str) -> ClassificationResult:
        """Classify query using embedding similarity."""
        start_time = time.time()
        
        query_embedding = self.embedding_func(query)
        
        # Find best matching category
        best_category = QueryCategory.SIMPLE
        best_score = 0.0
        
        for category, embeddings in self._category_embeddings.items():
            for emb in embeddings:
                score = self._cosine_similarity(query_embedding, emb)
                if score > best_score:
                    best_score = score
                    best_category = category
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ClassificationResult(
            query=query,
            category=best_category,
            confidence=best_score,
            latency_ms=latency_ms,
        )
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)


class QueryClassifier:
    """
    Main query classifier interface.
    
    FOCUS: Route to appropriate handler
    MUST: Simple/FAQ/Complex/Out-of-scope
    TARGET: <12ms classification time
    TARGET: >94% accuracy
    
    Usage:
        classifier = QueryClassifier()
        result = classifier.classify(query)
        
        if result.category == QueryCategory.SIMPLE:
            # Use fast model
        elif result.category == QueryCategory.COMPLEX:
            # Use advanced retrieval + powerful model
    """
    def __init__(
        self,
        config: Optional[ClassifierConfig] = None,
        embedding_func: Optional[Callable[[str], list[float]]] = None,
        category_examples: Optional[dict[QueryCategory, list[str]]] = None,
    ):
        """
        Initialize query classifier.
        
        Args:
            config: Classifier configuration
            embedding_func: Optional embedding function for similarity-based classification
            category_examples: Optional examples for each category
        """
        self.config = config or ClassifierConfig()
        
        # Primary: rule-based (fast)
        self._rule_classifier = RuleBasedClassifier(self.config)
        
        # Optional: embedding-based (more accurate)
        self._embedding_classifier = None
        if embedding_func and category_examples:
            self._embedding_classifier = EmbeddingBasedClassifier(
                embedding_func, category_examples, self.config
            )
        
        logger.info(
            f"Initialized QueryClassifier: "
            f"target_latency={self.config.target_latency_ms}ms"
        )
    
    def classify(
        self,
        query: str,
        use_embeddings: bool = False,
    ) -> ClassificationResult:
        """
        Classify a query.
        
        TARGET: <12ms classification time
        TARGET: >94% accuracy
        
        Args:
            query: Query to classify
            use_embeddings: Whether to use embedding-based classification
            
        Returns:
            ClassificationResult with category and metadata
        """
        if use_embeddings and self._embedding_classifier:
            result = self._embedding_classifier.classify(query)
        else:
            result = self._rule_classifier.classify(query)
        
        # Add routing suggestions
        result = self._add_routing_suggestions(result)
        
        return result
    
    def _add_routing_suggestions(
        self,
        result: ClassificationResult,
    ) -> ClassificationResult:
        """Add model and routing suggestions based on classification."""
        
        if result.category == QueryCategory.SIMPLE:
            result.suggested_model = "local"  # Fast local model
            result.use_cache = True
        elif result.category == QueryCategory.FAQ:
            result.suggested_model = "local"
            result.use_cache = True
        elif result.category == QueryCategory.COMPLEX:
            result.suggested_model = "gpt-4"  # Powerful model
            result.use_cache = False
        elif result.category == QueryCategory.OUT_OF_SCOPE:
            result.suggested_model = "local"  # Just explain it's out of scope
            result.use_cache = False
        elif result.category == QueryCategory.AMBIGUOUS:
            result.suggested_model = "local"
            result.needs_clarification = True
        else:
            result.suggested_model = "local"
        
        return result
    
    async def classify_async(
        self,
        query: str,
        use_embeddings: bool = False,
    ) -> ClassificationResult:
        """Async version of classify."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.classify(query, use_embeddings)
        )
    
    def classify_batch(
        self,
        queries: list[str],
    ) -> list[ClassificationResult]:
        """Classify multiple queries."""
        return [self.classify(q) for q in queries]
    
    def is_simple(self, query: str) -> bool:
        """Quick check if query is simple."""
        result = self.classify(query)
        return result.category == QueryCategory.SIMPLE
    
    def is_complex(self, query: str) -> bool:
        """Quick check if query is complex."""
        result = self.classify(query)
        return result.category == QueryCategory.COMPLEX
    
    def is_in_scope(self, query: str) -> bool:
        """Quick check if query is in scope."""
        result = self.classify(query)
        return result.category != QueryCategory.OUT_OF_SCOPE


def create_query_classifier(
    domain_keywords: Optional[dict[str, list[str]]] = None,
    out_of_scope_keywords: Optional[list[str]] = None,
    faq_patterns: Optional[list[str]] = None,
    **kwargs,
) -> QueryClassifier:
    """
    Factory function to create query classifier.
    
    Args:
        domain_keywords: Keywords by domain for domain detection
        out_of_scope_keywords: Keywords indicating out-of-scope queries
        faq_patterns: Regex patterns for FAQ detection
        **kwargs: Additional config options
        
    Returns:
        Configured QueryClassifier instance
    """
    config = ClassifierConfig(
        domain_keywords=domain_keywords or {},
        out_of_scope_keywords=out_of_scope_keywords or [],
        faq_patterns=faq_patterns or [],
        **kwargs,
    )
    
    return QueryClassifier(config=config)
