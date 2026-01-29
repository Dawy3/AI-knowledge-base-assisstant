"""
Query Transformation for RAG Pipeline.

FOCUS: Multi-query generation (3-5 variations)
HIGHEST ROI: Semantic + keyword variations
OPTIONS: HyDE, query expansion, step-back

Uses LangChain for LLM-based transformations.
Install: pip install langchain langchain-openai

Expected improvement: +15-25% recall with multi-query approach
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class TransformationType(str, Enum):
    """Query transformation types."""
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"
    STEP_BACK = "step_back"
    DECOMPOSITION = "decomposition"


@dataclass
class TransformedQuery:
    """Result of query transformation."""

    original_query: str
    transformed_queries: list[str]
    transformation_type: TransformationType
    keywords: list[str] = field(default_factory=list)
    hypothetical_document: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def all_queries(self) -> list[str]:
        """Get all queries including original."""
        queries = [self.original_query] + self.transformed_queries
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
        return len(self.all_queries)


@dataclass
class TransformerConfig:
    """Configuration for query transformer."""
    num_variations: int = 4
    include_original: bool = True
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7


class RuleBasedTransformer:
    """
    Fast rule-based query transformer (no LLM calls).

    Good for low latency transformations.
    """

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "to", "of", "in", "for", "on",
        "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "i", "me", "my", "we", "our", "you", "your",
        "he", "him", "she", "her", "it", "they", "them", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "and",
        "but", "if", "or", "because", "until", "while", "please", "help",
    }

    SYNONYMS = {
        "how to": ["steps to", "way to", "method to"],
        "what is": ["define", "explain", "meaning of"],
        "best": ["top", "recommended", "optimal"],
        "problem": ["issue", "error", "bug"],
        "fix": ["solve", "resolve", "repair"],
        "create": ["make", "build", "generate"],
    }

    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig()

    def transform(self, query: str) -> TransformedQuery:
        """Generate query variations using rules."""
        start_time = time.time()

        variations = []
        keywords = self._extract_keywords(query)
        query_lower = query.lower()

        # Keyword-only query
        if keywords:
            keyword_query = " ".join(keywords)
            if keyword_query.lower() != query_lower:
                variations.append(keyword_query)

        # Question form variations
        if not query_lower.startswith(("what", "how", "why", "when", "where")):
            variations.append(f"What is {query}?")
        else:
            # Remove question prefix
            for prefix in ["what is ", "how to ", "why ", "when ", "where "]:
                if query_lower.startswith(prefix):
                    remainder = query[len(prefix):].rstrip("?")
                    if remainder:
                        variations.append(remainder)
                    break

        # Synonym replacement
        for phrase, synonyms in self.SYNONYMS.items():
            if phrase in query_lower:
                variation = query_lower.replace(phrase, synonyms[0])
                variations.append(variation.capitalize())
                break

        # Limit variations
        variations = variations[: self.config.num_variations]

        return TransformedQuery(
            original_query=query,
            transformed_queries=variations,
            transformation_type=TransformationType.MULTI_QUERY,
            keywords=keywords,
            latency_ms=(time.time() - start_time) * 1000,
        )

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query."""
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower())
        return [w for w in words if w not in self.STOPWORDS and len(w) > 2]


class LLMQueryTransformer:
    """
    LLM-based query transformer using LangChain.

    Higher latency but better quality transformations.
    """

    def __init__(
        self,
        llm: Any = None,
        config: Optional[TransformerConfig] = None,
    ):
        """
        Initialize with LangChain LLM.

        Args:
            llm: LangChain LLM instance (e.g., ChatOpenAI)
            config: Transformer configuration
        """
        self.llm = llm
        self.config = config or TransformerConfig()
        self._multi_query_chain = None
        self._hyde_chain = None

    def _get_multi_query_chain(self):
        """Lazy load multi-query chain."""
        if self._multi_query_chain is None and self.llm:
            try:
                

                template = """Generate {num} different versions of this search query.
Each version should capture the same intent but use different words.
Return only the queries, one per line.

Query: {query}

Variations:"""

                prompt = ChatPromptTemplate.from_template(template)
                self._multi_query_chain = prompt | self.llm | StrOutputParser()
            except ImportError:
                logger.warning("LangChain not installed for multi-query")
        return self._multi_query_chain

    def _get_hyde_chain(self):
        """Lazy load HyDE chain."""
        if self._hyde_chain is None and self.llm:
            try:
                template = """Write a short passage (about 100 words) that would perfectly answer this question:

Question: {query}

Passage:"""

                prompt = ChatPromptTemplate.from_template(template)
                self._hyde_chain = prompt | self.llm | StrOutputParser()
            except ImportError:
                logger.warning("LangChain not installed for HyDE")
        return self._hyde_chain

    async def multi_query(self, query: str) -> TransformedQuery:
        """Generate multiple query variations."""
        start_time = time.time()

        chain = self._get_multi_query_chain()
        if not chain:
            return TransformedQuery(
                original_query=query,
                transformed_queries=[],
                transformation_type=TransformationType.MULTI_QUERY,
            )

        try:
            response = await chain.ainvoke({
                "query": query,
                "num": self.config.num_variations,
            })

            variations = [
                line.strip()
                for line in response.split("\n")
                if line.strip() and line.strip().lower() != query.lower()
            ]

            return TransformedQuery(
                original_query=query,
                transformed_queries=variations[: self.config.num_variations],
                transformation_type=TransformationType.MULTI_QUERY,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Multi-query transformation failed: {e}")
            return TransformedQuery(
                original_query=query,
                transformed_queries=[],
                transformation_type=TransformationType.MULTI_QUERY,
            )

    async def hyde(self, query: str) -> TransformedQuery:
        """Generate hypothetical document (HyDE)."""
        start_time = time.time()

        chain = self._get_hyde_chain()
        if not chain:
            return TransformedQuery(
                original_query=query,
                transformed_queries=[],
                transformation_type=TransformationType.HYDE,
            )

        try:
            hypothetical_doc = await chain.ainvoke({"query": query})

            return TransformedQuery(
                original_query=query,
                transformed_queries=[hypothetical_doc],
                transformation_type=TransformationType.HYDE,
                hypothetical_document=hypothetical_doc,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"HyDE transformation failed: {e}")
            return TransformedQuery(
                original_query=query,
                transformed_queries=[],
                transformation_type=TransformationType.HYDE,
            )


class QueryTransformer:
    """
    Main query transformer interface.

    FOCUS: Multi-query generation (3-5 variations)
    HIGHEST ROI: Semantic + keyword variations

    Usage:
        transformer = QueryTransformer()
        result = await transformer.transform(query)
        all_queries = result.all_queries
    """

    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        llm: Any = None,
    ):
        """
        Initialize query transformer.

        Args:
            config: Transformer configuration
            llm: Optional LangChain LLM for LLM-based transformations
        """
        self.config = config or TransformerConfig()
        self._rule_transformer = RuleBasedTransformer(self.config)
        self._llm_transformer = LLMQueryTransformer(llm, self.config) if llm else None

        logger.info(
            f"Initialized QueryTransformer: "
            f"num_variations={self.config.num_variations}, "
            f"use_llm={llm is not None}"
        )

    async def transform(
        self,
        query: str,
        transformation_type: TransformationType = TransformationType.MULTI_QUERY,
        use_llm: bool = False,
    ) -> TransformedQuery:
        """
        Transform a query into multiple variations.

        Args:
            query: Original search query
            transformation_type: Type of transformation
            use_llm: Whether to use LLM (if available)

        Returns:
            TransformedQuery with variations
        """
        query = query.strip()
        if not query:
            return TransformedQuery(
                original_query="",
                transformed_queries=[],
                transformation_type=transformation_type,
            )

        # Use LLM for complex transformations if available
        if use_llm and self._llm_transformer:
            if transformation_type == TransformationType.MULTI_QUERY:
                llm_result = await self._llm_transformer.multi_query(query)
                # Combine with rule-based for better coverage
                rule_result = self._rule_transformer.transform(query)

                all_variations = llm_result.transformed_queries + rule_result.transformed_queries
                seen = {query.lower()}
                unique = []
                for v in all_variations:
                    if v.lower() not in seen:
                        seen.add(v.lower())
                        unique.append(v)

                return TransformedQuery(
                    original_query=query,
                    transformed_queries=unique[: self.config.num_variations],
                    transformation_type=TransformationType.MULTI_QUERY,
                    keywords=rule_result.keywords,
                    latency_ms=llm_result.latency_ms,
                )

            elif transformation_type == TransformationType.HYDE:
                return await self._llm_transformer.hyde(query)

        # Fall back to rule-based
        return self._rule_transformer.transform(query)

    async def multi_query(self, query: str, use_llm: bool = False) -> TransformedQuery:
        """Generate multiple query variations."""
        return await self.transform(query, TransformationType.MULTI_QUERY, use_llm)

    async def hyde(self, query: str) -> TransformedQuery:
        """Generate hypothetical document (HyDE). Requires LLM."""
        return await self.transform(query, TransformationType.HYDE, use_llm=True)

    def extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query (synchronous)."""
        return self._rule_transformer._extract_keywords(query)


def create_query_transformer(
    num_variations: int = 4,
    llm: Any = None,
    **kwargs,
) -> QueryTransformer:
    """
    Factory function to create query transformer.

    Args:
        num_variations: Number of query variations to generate
        llm: Optional LangChain LLM instance
        **kwargs: Additional config options

    Returns:
        Configured QueryTransformer instance

    Example:
        # Rule-based only (fast)
        transformer = create_query_transformer()

        # With LLM (higher quality)
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        transformer = create_query_transformer(llm=llm)
    """
    config = TransformerConfig(num_variations=num_variations, **kwargs)
    return QueryTransformer(config=config, llm=llm)
