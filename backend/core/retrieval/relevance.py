"""
Relevance Filtering for RAG Pipeline.

FOCUS: Score chunks 0-1, drop below 0.6
EXPECTED: Reduce chunks 10→4, eliminate 30% noise
RESULT: Better accuracy + lower hallucination

Relevance filtering ensures only high-quality chunks reach the LLM,
reducing noise and improving response quality.
"""

import logging 
import time
import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)

class FilterStrategy(str, Enum):
    """Filtering strategies."""
    
    THRESHOLD = "threshold"      # Fixed score threshold
    TOP_K = "top_k"              # Keep top K results
    DYNAMIC = "dynamic"          # Adaptive threshold based on score distribution
    PERCENTILE = "percentile"    # Keep top percentile


@dataclass
class FilterConfig:
    """
    Configuration for relevance filtering.
    
    FOCUS: Score chunks 0-1, drop below 0.6
    """
    
    # Primary threshold
    min_relevance_score: float = 0.6  # Drop chunks below this
    
    # Strategy
    strategy: FilterStrategy = FilterStrategy.THRESHOLD
    
    # Top-K settings
    max_chunks: int = 10  # Maximum chunks to keep
    min_chunks: int = 1   # Minimum chunks to return (even if below threshold)
    
    # Dynamic threshold settings
    dynamic_std_multiplier: float = 1.0  # threshold = mean - (std * multiplier)
    
    # Percentile settings
    percentile: float = 70.0  # Keep top N percentile
    
    # Score normalization
    normalize_scores: bool = True  # Normalize to 0-1 range
    

@dataclass
class FilteredChunk:
    """A chunk that passed relevance filtering."""
    
    chunk_id: str
    content: str
    relevance_score: float  # Normalized 0-1
    original_score: float   # Original score before normalization
    
    # Metadata
    metadata: dict = field(default_factory=dict)
    document_id: str = ""
    rank: int = 0
    
    # Filter info
    passed_threshold: bool = True
    filter_reason: str = ""
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "original_score": self.original_score,
            "metadata": self.metadata,
            "document_id": self.document_id,
            "rank": self.rank,
        }


@dataclass
class FilteredResult:
    """Result of relevance filtering."""
    
    chunks: list[FilteredChunk]
    
    # Statistics
    input_count: int = 0
    output_count: int = 0
    filtered_count: int = 0
    
    # Threshold used
    threshold_used: float = 0.0
    strategy_used: FilterStrategy = FilterStrategy.THRESHOLD
    
    # Performance
    latency_ms: float = 0.0
    
    # Quality metrics
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    
    @property
    def filter_rate(self) -> float:
        """Percentage of chunks filtered out."""
        if self.input_count == 0:
            return 0.0
        return self.filtered_count / self.input_count
    
    @property
    def retention_rate(self) -> float:
        """Percentage of chunks retained."""
        return 1.0 - self.filter_rate
    
class ScoreNormalizer:
    """
    Normalize scores to 0-1 range.
    
    Different search methods produce different score ranges:
    - Cosine similarity: [-1, 1] or [0, 1]
    - BM25: [0, ∞)
    - Cross-encoder: varies by model
    """
    
    @staticmethod
    def normalize_cosine(scores: list[float]) -> list[float]:
        """Normalize cosine similarity scores (already 0-1 or -1 to 1)."""
        return [(s + 1) / 2 if s < 0 else s for s in scores]
    
    @staticmethod
    def normalize_minmax(scores: list[float]) -> list[float]:
        """Min-max normalization to 0-1 range."""
        if not scores:
            return []
        
        min_s = min(scores)
        max_s = max(scores)
        
        if max_s == min_s:
            return [1.0] * len(scores)
        
        return [(s - min_s) / (max_s - min_s) for s in scores]
    
    @staticmethod
    def normalize_sigmoid(scores: list[float], midpoint: float = 0.0) -> list[float]:
        """Sigmoid normalization for unbounded scores."""
        if not scores:
            return []
        return [1 / (1 + math.exp(-(s - midpoint))) for s in scores]
    
    @staticmethod
    def normalize_softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
        """Softmax normalization (scores sum to 1)."""
        if not scores:
            return []

        # Subtract max for numerical stability
        max_s = max(scores)
        exp_scores = [math.exp((s - max_s) / temperature) for s in scores]
        sum_exp = sum(exp_scores)

        return [e / sum_exp for e in exp_scores]
    
class RelevanceFilter:
    """
    Relevance filtering for RAG retrieval results.
    
    FOCUS: Score chunks 0-1, drop below 0.6
    EXPECTED: Reduce chunks 10→4, eliminate 30% noise
    RESULT: Better accuracy + lower hallucination
    
    Usage:
        filter = RelevanceFilter(min_score=0.6)
        result = filter.filter(chunks)
    """
    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        min_score: Optional[float] = None,
        max_chunks: Optional[int] = None,
    ):
        """
        Initialize relevance filter.
        
        Args:
            config: Full configuration object
            min_score: Shortcut for min_relevance_score
            max_chunks: Shortcut for max_chunks
        """
        self.config = config or FilterConfig()
        
        # Apply shortcuts
        if min_score is not None:
            self.config.min_relevance_score = min_score
        if max_chunks is not None:
            self.config.max_chunks = max_chunks
        
        self._normalizer = ScoreNormalizer()
        
        logger.info(
            f"Initialized RelevanceFilter: "
            f"threshold={self.config.min_relevance_score}, "
            f"strategy={self.config.strategy.value}, "
            f"max_chunks={self.config.max_chunks}"
        )
        
    def filter(
        self,
        chunks: list[dict],
        scores: Optional[list[float]] = None,        
    ) -> FilteredResult:
        """
        Filter chunks by relevance score.
        
        Args:
            chunks: List of chunks with 'chunk_id', 'content', and optionally 'score'
            scores: Optional separate list of scores (overrides chunk scores)
            
        Returns:
            FilteredResult with filtered chunks and statistics
        """
        start_time = time.time()
        
        if not chunks:
            return FilteredResult(
                chunks=[],
                input_count=0,
                output_count=0,
            )
        
        # Extract score
        if scores is None:
            scores = [c.get("score", c.get("relevance_score", 0.0)) for c in chunks]

        original_scores = scores.copy()
        
        # Normalize scores if configured
        if self.config.normalize_scores:
            scores = self._normalizer.normalize_minmax(scores)
            
        # Calculate threshold based on strategy
        threshold = self._calculate_threshold(scores)
        
        # Apply filtering
        filtered_chunks = []
        for i , (chunk, norm_score, orig_score) in enumerate(
            zip(chunks, scores, original_scores)
        ):
            passed = self._passes_filter(norm_score, threshold, len(filtered_chunks)) 
            
            filtered_chunk = FilteredChunk(
                chunk_id=chunk.get("chunk_id", f"chunk_{i}"),
                content=chunk.get("content", ""),
                relevance_score=norm_score,
                original_score=orig_score,
                metadata=chunk.get("metadata", {}),
                document_id=chunk.get("document_id", ""),
                rank=i,
                passed_threshold=passed,
                filter_reason="" if passed else f"score {norm_score:.3f} < {threshold:.3f}",
            )
            
            if passed:
                filtered_chunks.append(filtered_chunk)
        
        # Ensure minimum chunks
        if len(filtered_chunks) < self.config.min_chunks and chunks:
            # Add top chunks even if below threshold
            all_indexed = sorted(
                enumerate(zip(chunks, scores, original_scores)),
                key=lambda x: x[1][1],
                reverse=True
            )
            
            seen_ids = {c.chunk_id for c in filtered_chunks}
            
            for i, (chunk, norm_score, orig_score) in all_indexed:
                chunk_id = chunk.get("chunk_id", f"chunk_{i}")
                if chunk_id not in seen_ids:
                    filtered_chunks.append(FilteredChunk(
                        chunk_id=chunk_id,
                        content=chunk.get("content", ""),
                        relevance_score=norm_score,
                        original_score=orig_score,
                        metadata=chunk.get("metadata", {}),
                        document_id=chunk.get("document_id", ""),
                        rank=len(filtered_chunks),
                        passed_threshold=False,
                        filter_reason="added to meet minimum",
                    ))
                    seen_ids.add(chunk_id)
                
                if len(filtered_chunks) >= self.config.min_chunks:
                    break
                
        # Limit to max chunks
        filtered_chunks = filtered_chunks[:self.config.max_chunks]
        
        # Update ranks
        for i, chunk in enumerate(filtered_chunks):
            chunk.rank = i
        
        # Calculate statistics
        latency_ms = (time.time() - start_time) * 1000
        
        result_scores = [c.relevance_score for c in filtered_chunks]
        
        result = FilteredResult(
            chunks=filtered_chunks,
            input_count=len(chunks),
            output_count=len(filtered_chunks),
            filtered_count=len(chunks) - len(filtered_chunks),
            threshold_used=threshold,
            strategy_used=self.config.strategy,
            latency_ms=latency_ms,
            avg_score=sum(result_scores) / len(result_scores) if result_scores else 0.0,
            min_score=min(result_scores) if result_scores else 0.0,
            max_score=max(result_scores) if result_scores else 0.0,
        )
        
        logger.debug(
            f"Filtered {result.input_count} → {result.output_count} chunks "
            f"({result.filter_rate:.1%} filtered), threshold={threshold:.3f}"
        )
        
        return result
            


    def _calculate_threshold(self, scores: list[float]) -> float:
        """Calculate threshold based on strategy."""
        if self.config.strategy == FilterStrategy.THRESHOLD:
            return self.config.min_relevance_score
        
        elif self.config.strategy == FilterStrategy.DYNAMIC:
            if not scores:
                return self.config.min_relevance_score
            
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0
            
            dynamic_threshold = mean - (std * self.config.dynamic_std_multiplier)
            # Don't go below the minimum configured threshold
            return max(dynamic_threshold, self.config.min_relevance_score * 0.5)
        
        elif self.config.strategy == FilterStrategy.PERCENTILE:
            if not scores:
                return self.config.min_relevance_score
            
            sorted_scores = sorted(scores)
            percentile_idx = int(len(sorted_scores) * (1 - self.config.percentile / 100))
            percentile_idx = max(0, min(percentile_idx, len(sorted_scores) - 1))
            
            return sorted_scores[percentile_idx]
        
        elif self.config.strategy == FilterStrategy.TOP_K:
            # TOP_K doesn't use threshold, handled in _passes_filter
            return 0.0
        
        return self.config.min_relevance_score


       
    def _passes_filter(
        self,
        score: float,
        threshold: float,
        current_count: int,
    ) -> bool:
        """Check if a score passes the filter."""
        if self.config.strategy == FilterStrategy.TOP_K:
            return current_count < self.config.max_chunks
        
        return score >= threshold
    
    def filter_reranked(
        self,
        reranked_results: list,  # RerankedResult objects
    ) -> FilteredResult:
        """
        Filter reranked results.
        
        Convenience method for filtering after reranking.
        """
        chunks = []
        scores = []
        
        for result in reranked_results:
            chunks.append({
                "chunk_id": result.chunk_id,
                "content": result.content,
                "metadata": result.metadata,
                "document_id": result.document_id,
            })
            scores.append(result.rerank_score)
        
        return self.filter(chunks, scores)
    
    def get_stats(self) -> dict:
        """Get filter configuration stats."""
        return {
            "min_relevance_score": self.config.min_relevance_score,
            "strategy": self.config.strategy.value,
            "max_chunks": self.config.max_chunks,
            "min_chunks": self.config.min_chunks,
        }


class AdaptiveRelevanceFilter(RelevanceFilter):
    """
    Adaptive relevance filter that adjusts threshold based on query characteristics.
    
    Useful when query complexity varies significantly.
    """
    
    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        high_confidence_threshold: float = 0.8,
        low_confidence_threshold: float = 0.5,
    ):
        super().__init__(config)
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
    
    def filter_adaptive(
        self,
        chunks: list[dict],
        scores: Optional[list[float]] = None,
        query_confidence: float = 0.5,
    ) -> FilteredResult:
        """
        Filter with adaptive threshold based on query confidence.
        
        Args:
            chunks: Chunks to filter
            scores: Optional scores
            query_confidence: Confidence in query quality (0-1)
                High confidence → stricter filtering
                Low confidence → looser filtering
        """
        # Interpolate threshold based on confidence
        threshold = (
            self.low_confidence_threshold +
            (self.high_confidence_threshold - self.low_confidence_threshold) * query_confidence
        )
        
        # Temporarily override config
        original_threshold = self.config.min_relevance_score
        self.config.min_relevance_score = threshold
        
        result = self.filter(chunks, scores)
        
        # Restore
        self.config.min_relevance_score = original_threshold
        
        return result


def create_relevance_filter(
    min_score: float = 0.6,
    max_chunks: int = 10,
    strategy: str = "threshold",
    **kwargs,
) -> RelevanceFilter:
    """
    Factory function to create relevance filter.
    
    Args:
        min_score: Minimum relevance score (default 0.6)
        max_chunks: Maximum chunks to return
        strategy: Filtering strategy
        **kwargs: Additional config options
        
    Returns:
        Configured RelevanceFilter instance
    """
    config = FilterConfig(
        min_relevance_score=min_score,
        max_chunks=max_chunks,
        strategy=FilterStrategy(strategy),
        **kwargs,
    )
    
    return RelevanceFilter(config=config)