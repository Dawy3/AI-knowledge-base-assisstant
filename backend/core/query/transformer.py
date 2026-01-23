"""
Query Transformation for RAG Pipeline.

FOCUS: Multi-query generation (3-5 variations)
HIGHEST ROI: Semantic + keyword variations
OPTIONS: HyDE, query expansion, step-back

Query transformation improves retrieval by:
1. Generating multiple query variations to capture different phrasings
2. Creating hypothetical documents (HyDE) for better semantic matching
3. Expanding queries with related terms
4. Step-back prompting for complex queries

Expected improvement: +15-25% recall with multi-query approach
"""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TransformationType(str, Enum):
    """Query transformation types."""
    
    MULTI_QUERY = "multi_query"       # Generate multiple query variations => Highest ROI 'default'
    HYDE = "hyde"                      # Hypothetical Document Embedding
    EXPANSION = "expansion"            # Query expansion with related terms
    STEP_BACK = "step_back"           # Step-back prompting for complex queries
    DECOMPOSITION = "decomposition"    # Break complex query into sub-queries
    KEYWORD = "keyword"               # Extract keyword-focused query
    SEMANTIC = "semantic"             # Generate semantic variation


@dataclass
class TransformedQuery:
    """Result of query transformation."""

    original_query: str
    transformed_queries: list[str]
    transformation_type: TransformationType
    
    # Metadata
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    
    # HyDE specific
    hypothetical_document: Optional[str] = None
    
    # Step-back specific
    step_back_query: Optional[str] = None
    
    # Performance
    latency_ms: float = 0.0
    
    @property
    def all_queries(self) -> list[str]:
        """Get all queries including original."""
        queries = [self.original_query] + self.transformed_queries
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)
        return unique
    
    @property
    def query_count(self) -> int:
        """Total number of unique queries."""
        return len(self.all_queries)
    

@dataclass
class TransformerConfig:
    """Configuration for query transformer."""
    
    # Multi-query settings
    num_variations: int = 4  # 3-5 recommended
    include_original: bool = True
    include_keyword_query: bool = True
    include_semantic_query: bool = True
    
    # HyDE settings
    hyde_document_length: int = 100  # Target length in words
    
    # Expansion settings
    max_expansion_terms: int = 5
    
    # LLM settings (for LLM-based transformations)
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_timeout: float = 10.0


class BaseQueryTransformer(ABC):
    """Abstract base class for query transformers."""
    
    @abstractmethod
    async def transform(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> TransformedQuery:
        """Transform a Query."""
        pass
    
    
class RuleBasedTransformer(BaseQueryTransformer):
    """
    Rule-based query transformer.
    
    Generates variations without LLM calls for low latency.
    Good for simple transformations and keyword extraction.
    """
    
    # Common question words for variation
    QUESTION_WORDS = ["what", "how", "why", "when", "where", "who", "which"]
    
    # Synonym mappings for common terms
    SYNONYMS = {
        "how to": ["steps to", "way to", "method to", "process for"],
        "what is": ["define", "explain", "describe", "meaning of"],
        "best": ["top", "recommended", "optimal", "ideal"],
        "problem": ["issue", "error", "bug", "difficulty"],
        "fix": ["solve", "resolve", "repair", "correct"],
        "create": ["make", "build", "generate", "set up"],
        "delete": ["remove", "clear", "erase", "drop"],
        "get": ["retrieve", "fetch", "obtain", "access"],
    }
    
    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig()
    
    async def transform(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> TransformedQuery:
        """Generate query variations using rules."""
        start_time = time.time()
        
        variations = []
        keywords = self._extract_keywords(query)
        
        # Generate semantic variations (rephrasings)
        if self.config.include_semantic_query:
            semantic_vars = self._generate_semantic_variations(query)
            variations.extend(semantic_vars)
        
        # Generate keyword-focused query
        if self.config.include_keyword_query and keywords:
            keyword_query = " ".join(keywords)
            if keyword_query.lower() != query.lower():
                variations.append(keyword_query)
        
        # Generate synonym-based variations
        synonym_vars = self._generate_synonym_variations(query)
        variations.extend(synonym_vars)
        
        # Limit to configured number
        variations = variations[:self.config.num_variations]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return TransformedQuery(
            original_query=query,
            transformed_queries=variations,
            transformation_type=TransformationType.MULTI_QUERY,
            keywords=keywords,
            latency_ms=latency_ms,
        )
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords from query."""
        # Simple keyword extraction (stopword removal)
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "i", "me", "my", "myself", "we",
            "our", "you", "your", "he", "him", "she", "her", "it",
            "they", "them", "what", "which", "who", "whom", "this",
            "that", "these", "those", "am", "and", "but", "if", "or",
            "because", "until", "while", "please", "help", "want",
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z0-9_-]+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    def _generate_semantic_variations(self, query: str) -> list[str]:
        """Generate semantic variations of the query."""
        variations = []
        query_lower = query.lower()
        
        # Variation 1: Question form transformation
        if not any(query_lower.startswith(qw) for qw in self.QUESTION_WORDS):
            # Add question form
            variations.append(f"What is {query}?")
        else:
            # Remove question form
            for qw in self.QUESTION_WORDS:
                if query_lower.startswith(qw):
                    remainder = query[len(qw):].strip()
                    if remainder:
                        variations.append(remainder.rstrip("?"))
                    break
        
        # Variation 2: Add context request
        if "explain" not in query_lower:
            variations.append(f"Explain {query}")
        
        # Variation 3: Detailed form
        if "detail" not in query_lower and "specific" not in query_lower:
            variations.append(f"{query} in detail")
        
        return variations[:2]  # Limit variations
    
    def _generate_synonym_variations(self, query: str) -> list[str]:
        """Generate variations using synonyms."""
        variations = []
        query_lower = query.lower()
        
        for phrase, synonyms in self.SYNONYMS.items():
            if phrase in query_lower:
                # Replace with first synonym
                variation = query_lower.replace(phrase, synonyms[0])
                if variation != query_lower:
                    # Capitalize first letter
                    variations.append(variation.capitalize())
                    break  # Only one synonym replacement per query
        
        return variations
                

class LLMQueryTransformer(BaseQueryTransformer):
    """
    LLM-based query transformer.
    
    Uses LLM to generate high-quality query variations.
    Higher latency but better quality transformations.
    """
    
    MULTI_QUERY_PROMPT = """Generate {num} different versions of the following search query.
Each version should capture the same intent but use different words or phrasings.
Include both semantic variations and keyword-focused versions.

Original query: {query}

Return only the queries, one per line, without numbering or explanations."""

    HYDE_PROMPT = """Write a short passage (about {length} words) that would be a perfect answer to the following question.
The passage should be informative and directly answer the question.

Question: {query}

Passage:"""

    STEP_BACK_PROMPT = """Given the following specific question, generate a more general "step-back" question that would help understand the broader context needed to answer the original question.

Original question: {query}

Step-back question:"""

    DECOMPOSITION_PROMPT = """Break down the following complex question into 2-3 simpler sub-questions that, when answered together, would help answer the original question.

Complex question: {query}

Sub-questions (one per line):"""

    def __init__(
        self,
        llm_client: Any = None,  # LLM client instance
        config: Optional[TransformerConfig] = None,
    ):
        self.llm_client = llm_client
        self.config = config or TransformerConfig()
        
    async def transform(
        self,
        query: str,
        context: Optional[dict] = None,
        transformation_type: TransformationType = TransformationType.MULTI_QUERY,
    ) -> TransformedQuery:
        """Transform query using LLM."""
        start_time = time.time()
        
        if transformation_type == TransformationType.MULTI_QUERY:
            result = await self._multi_query_transform(query)
        elif transformation_type == TransformationType.HYDE:
            result = await self._hyde_transform(query)
        elif transformation_type == TransformationType.STEP_BACK:
            result = await self._step_back_transform(query)
        elif transformation_type == TransformationType.DECOMPOSITION:
            result = await self._decomposition_transform(query)
        else:
            # Fallback to rule-based
            rule_transformer = RuleBasedTransformer(self.config)
            result = await rule_transformer.transform(query, context)
        
        result.latency_ms = (time.time() - start_time) * 1000
        return result
    
    async def _multi_query_transform(self, query: str) -> TransformedQuery:
        """Generate multiple query variations using LLM."""
        prompt = self.MULTI_QUERY_PROMPT.format(
            num=self.config.num_variations,
            query=query,
        )
        
        response = await self._call_llm(prompt)
        
        # Parse response into queries
        variations = [
            line.strip()
            for line in response.split("\n")
            if line.strip() and line.strip() != query
        ]
        
        
        return TransformedQuery(
            original_query=query,
            transformed_queries=variations[:self.config.num_variations],
            transformation_type=TransformationType.MULTI_QUERY,
        )
    
    async def _hyde_transform(self, query: str) -> TransformedQuery:
        """Generate hypothetical document for HyDE."""
        prompt = self.HYDE_PROMPT.format(
            length=self.config.hyde_document_length,
            query=query,
        )
        
        hypothetical_doc = await self._call_llm(prompt)
        
        return TransformedQuery(
            original_query=query,
            transformed_queries=[hypothetical_doc],
            transformation_type=TransformationType.HYDE,
            hypothetical_document=hypothetical_doc,
        )
    
    async def _step_back_transform(self, query: str) -> TransformedQuery:
        """Generate step-back query for complex questions."""
        prompt = self.STEP_BACK_PROMPT.format(query=query)
        
        step_back_query = await self._call_llm(prompt)
        step_back_query = step_back_query.strip()
        
        return TransformedQuery(
            original_query=query,
            transformed_queries=[step_back_query],
            transformation_type=TransformationType.STEP_BACK,
            step_back_query=step_back_query,
        )
    
    async def _decomposition_transform(self, query: str) -> TransformedQuery:
        """Decompose complex query into sub-queries."""
        prompt = self.DECOMPOSITION_PROMPT.format(query=query)
        
        response = await self._call_llm(prompt)
        
        sub_queries = [
            line.strip().lstrip("0123456789.-) ")
            for line in response.split("\n")
            if line.strip()
        ]
        
        return TransformedQuery(
            original_query=query,
            transformed_queries=sub_queries,
            transformation_type=TransformationType.DECOMPOSITION,
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        if self.llm_client is None:
            logger.warning("No LLM client configured, returning empty response")
            return ""
        
        try:
            # Assuming OpenAI-compatible client
            response = await asyncio.wait_for(
                self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.llm_temperature,
                    max_tokens=500,
                ),
                timeout=self.config.llm_timeout,
            )
            return response.choices[0].message.content.strip()
        except asyncio.TimeoutError:
            logger.warning(f"LLM call timed out after {self.config.llm_timeout}s")
            return ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""


class QueryTransformer:
    """
    Main query transformer interface.
    
    FOCUS: Multi-query generation (3-5 variations)
    HIGHEST ROI: Semantic + keyword variations
    
    Combines rule-based and LLM-based transformations for
    optimal balance of quality and latency.
    
    Usage:
        transformer = QueryTransformer()
        result = await transformer.transform(query)
        all_queries = result.all_queries
    """
    
    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        llm_client: Any = None,
        use_llm: bool = False,
    ):
        """
        Initialize query transformer.
        
        Args:
            config: Transformer configuration
            llm_client: Optional LLM client for LLM-based transformations
            use_llm: Whether to use LLM for transformations
        """
        self.config = config or TransformerConfig()
        self.use_llm = use_llm and llm_client is not None
        
        # Initialize transformers
        self._rule_transformer = RuleBasedTransformer(self.config)
        self._llm_transformer = None
        
        if llm_client:
            self._llm_transformer = LLMQueryTransformer(llm_client, self.config)
            
        logger.info(
            f"Initialized QueryTransformer: "
            f"num_variations={self.config.num_variations}, "
            f"use_llm={self.use_llm}"
        )
                
    async def transform(
        self,
        query: str,
        transformation_type: TransformationType = TransformationType.MULTI_QUERY,
        context: Optional[dict] = None,
    ) -> TransformedQuery:
        """
        Transform a query into multiple variations.
        
        FOCUS: Generate 3-5 variations for better recall
        
        Args:
            query: Original search query
            transformation_type: Type of transformation to apply
            context: Optional context for transformation
            
        Returns:
            TransformedQuery with variations
        """
        start_time = time.time()
        
        # Validate query
        query = query.strip()
        if not query:
            return TransformedQuery(
                original_query="",
                transformed_queries=[],
                transformation_type=transformation_type,    
            )
        
        # Use LLM for complex transformations if available
        if self.use_llm and transformation_type in [
            TransformationType.HYDE,
            TransformationType.STEP_BACK,
            TransformationType.DECOMPOSITION,
        ]:
            result = await self._llm_transformer.transform(
                query, context, transformation_type
            )
        elif self.use_llm and transformation_type == TransformationType.MULTI_QUERY:
            # For multi-query, combine rule-based and LLM
            rule_result = await self._rule_transformer.transform(query, context)
            llm_result = await self._llm_transformer.transform(
                query, context, TransformationType.MULTI_QUERY
            )
            
            # Combine variations
            all_variations = rule_result.transformed_queries + llm_result.transformed_queries
            
            # Deduplicate
            seen = {query.lower()}
            unique_variations = []
            for v in all_variations:
                if v.lower() not in seen:
                    seen.add(v.lower())
                    unique_variations.append(v)
            
            result = TransformedQuery(
                original_query=query,
                transformed_queries=unique_variations[:self.config.num_variations],
                transformation_type=TransformationType.MULTI_QUERY,
                keywords=rule_result.keywords,
            )
        else:
            # Use rule-based transformer
            result = await self._rule_transformer.transform(query, context)
        
        result.latency_ms = (time.time() - start_time) * 1000
        
        logger.debug(
            f"Transformed query '{query}' into {result.query_count} variations "
            f"in {result.latency_ms:.2f}ms"
        )
        
        return result
    
    async def multi_query(self, query: str) -> TransformedQuery:
        """Generate multiple query variations (convenience method)."""
        return await self.transform(query, TransformationType.MULTI_QUERY)
    
    async def hyde(self, query: str) -> TransformedQuery:
        """Generate hypothetical document (HyDE)."""
        return await self.transform(query, TransformationType.HYDE)
    
    async def step_back(self, query: str) -> TransformedQuery:
        """Generate step-back query for complex questions."""
        return await self.transform(query, TransformationType.STEP_BACK)
    
    async def decompose(self, query: str) -> TransformedQuery:
        """Decompose complex query into sub-queries."""
        return await self.transform(query, TransformationType.DECOMPOSITION)
    
    def extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query (synchronous)."""
        return self._rule_transformer._extract_keywords(query)


def create_query_transformer(
    num_variations: int = 4,
    use_llm: bool = False,
    llm_client: Any = None,
    **kwargs,
) -> QueryTransformer:
    """
    Factory function to create query transformer.
    
    Args:
        num_variations: Number of query variations to generate
        use_llm: Whether to use LLM for transformations
        llm_client: LLM client instance
        **kwargs: Additional config options
        
    Returns:
        Configured QueryTransformer instance
    """
    config = TransformerConfig(
        num_variations=num_variations,
        **kwargs,
    )
    
    return QueryTransformer(
        config=config,
        llm_client=llm_client,
        use_llm=use_llm,
    )

            
