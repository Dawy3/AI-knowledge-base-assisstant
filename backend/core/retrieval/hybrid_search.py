"""
Hybrid Search Implementation for RAG Pipeline.

CRITICAL FILE - This is the core of the retrieval system.

FOCUS: Score = (Vector * 5) + (BM25 * 3) + (Recency * 0.2)
CRITICAL: Vector-only search WILL FAIL in production
MUST: BM25 for exact matches, Vector for semantic
EXPECTED: +40% quality improvement over vector-only

Hybrid search combines:
1. Vector search - Semantic similarity (handles synonyms, paraphrases)
2. BM25 search - Exact term matching (handles IDs, codes, specific terms)
3. Recency boost - Favor recent documents (optional)

Why hybrid beats vector-only:
- Vector search misses exact matches (error codes, IDs, technical terms)
- BM25 misses semantic similarity (synonyms, paraphrases)
- Combining both captures more relevant results
"""
import asyncio
import logging
import time
import math
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..config import settings
from .vector_search import VectorSearch, VectorSearchResult
from .bm25_search import BM25Search, BM25SearchResult

logger = logging.getLogger(__name__)

class FusionMethod(str, Enum):
    """Score fusion methods."""
    
    WEIGHTED_SUM = "weighted_sum"       # Simple weighted addition
    RECIPROCAL_RANK = "rrf"             # Reciprocal Rank Fusion
    RELATIVE_SCORE = "relative_score"   # Relative score Fusion
    

@dataclass
class HybridSearchConfig:
    """
    Configuration for hybrid search.

    FOCUS: Score = (Vector * 5) + (BM25 * 3) + (Recency * 0.2)
    Defaults are loaded from settings.retrieval config.
    """

    # Score weights - CRITICAL for quality (from config)
    vector_weight: float = None
    bm25_weight: float = None
    recency_weight: float = None

    # Fusion method
    fusion_method: FusionMethod = FusionMethod.WEIGHTED_SUM

    # RRF parameter (only for RRF fusion)
    rrf_k: int = 60

    # Retrieval parameters (from config)
    vector_top_k: int = None
    bm25_top_k: int = None
    final_top_k: int = None

    # Recency boost
    enable_recency_boost: bool = False
    recency_field: str = "created_at"   # Metadata field for timestamp
    recency_decay_days: float = 30.0    # Half-life in days

    # Score normalization
    normalize_scores: bool = True

    def __post_init__(self):
        """Load defaults from config if not specified."""
        if self.vector_weight is None:
            self.vector_weight = settings.retrieval.vector_weight
        if self.bm25_weight is None:
            self.bm25_weight = settings.retrieval.bm25_weight
        if self.recency_weight is None:
            self.recency_weight = settings.retrieval.recency_weight
        if self.vector_top_k is None:
            self.vector_top_k = settings.retrieval.top_k_retrieval
        if self.bm25_top_k is None:
            self.bm25_top_k = settings.retrieval.top_k_retrieval
        if self.final_top_k is None:
            self.final_top_k = settings.retrieval.top_k_retrieval
    
    
@dataclass
class HybridSearchResult:
    """Single result from hybrid search."""
    
    chunk_id: str
    combined_score: float  # Final fused score
    
    # Individual scores
    vector_score: float = 0.0
    bm25_score: float = 0.0
    recency_score: float = 0.0
    
    # Content and metadata
    content: str = ""
    metadata: dict = field(default_factory=dict)
    document_id: str = ""
    
    # Source tracking
    from_vector: bool = False
    from_bm25: bool = False
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "combined_score": self.combined_score,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "recency_score": self.recency_score,
            "content": self.content,
            "metadata": self.metadata,
            "document_id": self.document_id,
            "from_vector": self.from_vector,
            "from_bm25": self.from_bm25,
        }
        

@dataclass
class HybridSearchResponse:
    """Response from hybrid search."""

    results: list[HybridSearchResult]
    query: str = ""
    
    # Performance metrics
    latency_ms: float = 0.0
    vector_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    
    # Result statistics
    total_vector_results: int = 0
    total_bm25_results: int = 0
    unique_results: int = 0
    overlap_count: int = 0  # Results found by both methods
    
    # Configuration used 
    config: Optional[HybridSearchConfig] = None    
    
    @property
    def top_score(self) -> float:
        return self.results[0].combined_score if self.results else 0.0
    
    @property
    def overlap_ratio(self) -> float:
        """Ratio of results found by both search methods."""
        if self.unique_results == 0:
            return 0.0
        return self.overlap_count / self.unique_results
    
    
class ScoreFusion:
    """
    Score Fusion strategies for combining search results.
    """
    
    @staticmethod
    def weighted_sum(
        vector_score: float,
        bm25_score: float,
        recency_score: float,
        config: HybridSearchConfig,
    ) -> float:
        """
        Weighted sum fusion.
        
        Score = (Vector * 5) + (BM25 * 3) + (Recency * 0.2)
        """
        return (
            vector_score * config.vector_weight +
            bm25_score * config.bm25_weight +
            recency_score * config.recency_weight
        )
    
    @staticmethod
    def reciprocal_rank_fusion(
        vector_rank: int,
        bm25_rank: int,
        k: int = 60,
    ) -> float:
        """
        Reciprocal Rank Fusion (RRF).
        
        RRF(d) = Σ 1/(k + rank(d))
        
        More robust to score scale differences.
        """
        score = 0.0
        
        if vector_rank is not None:
            score += 1.0 / (k + vector_rank + 1) # +1: to avoid division by zero
        
        if bm25_rank is not None:
            score += 1.0 / (k + bm25_rank + 1)
            
        return score

    @staticmethod
    def normalize_scores(scores: list[float]) -> list[float]:
        """Normalize score to 0-1 range"""
        if not scores:
            return []
        
        min_s = min(scores)
        max_s = max(scores)
        
        if max_s == min_s:
            return [1.0] * len(scores)
        
        return [(s - min_s) / (max_s - min_s) for s in scores]
    
class HybridSearch:
    """
    Hybrid search combining vector and BM25 retrieval.
    
    CRITICAL: Vector-only search WILL FAIL in production
    EXPECTED: +40% quality improvement over vector-only
    
    Usage:
        hybrid = HybridSearch(vector_search, bm25_search)
        results = await hybrid.search(query, query_embedding)
    """
    
    def __init__(
        self,
        vector_search: VectorSearch,
        bm25_search: BM25Search,
        config: Optional[HybridSearchConfig] = None,
    ):
        """
        
        Initialize hybrid search.
        
        Args:
            vector_search: Vector search instance
            bm25_search: BM25 search instance
            config: Hybrid search configuration
        """
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.config = config or HybridSearchConfig()
        
        self._fusion = ScoreFusion()
        
        logger.info(
            f"Initialized HybridSearch: "
            f"vector_weight={self.config.vector_weight}, "
            f"bm25_weight={self.config.bm25_weight}, "
            f"fusion={self.config.fusion_method.value}"
        )
        
    async def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> HybridSearchResponse:
        """
        Perform hybrid search combining vector and BM25.
        
        FOCUS: Score = (Vector * 5) + (BM25 * 3) + (Recency * 0.2)
        
        Args:
            query: Search query text
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            HybridSearchResponse with combined results
        """
        start_time = time.time()
        final_k = top_k or self.config.final_top_k
        
        # Run both search in parallel
        vector_task = self.vector_search.search(
            query_embedding = query_embedding,
            top_k = self.config.vector_top_k,
            filter = filter
        )
        
        bm25_task = asyncio.create_task(
            asyncio.to_thread(
                lambda: self.bm25_search.search(
                    query = query,
                    top_k = self.config.bm25_top_k
                )
            )
        )
        
        # Wait for both
        vector_response, bm25_response = await asyncio.gather(
            vector_task,
            bm25_task,
        )
        
        # Fuse results
        fused_results = self._fuse_results(
            vector_results = vector_response.results,
            bm25_results = bm25_response.results,
        )
        
        # Sort by combined score and take top-k
        fused_results.sort(key=lambda x:x.combined_score, reverse=True)
        final_results = fused_results[:final_k]
        
        # Calculate statistics
        total_time = (time.time() - start_time) * 1000
        
        vector_ids = {r.chunk_id for r in vector_response.results}
        bm25_ids = {r.chunk_id for r in bm25_response.results}
        overlap = vector_ids & bm25_ids
        
        response = HybridSearchResponse(
            results=final_results,
            query=query,
            latency_ms=total_time,
            vector_latency_ms=vector_response.latency_ms,
            bm25_latency_ms=bm25_response.latency_ms,
            total_vector_results=len(vector_response.results),
            total_bm25_results=len(bm25_response.results),
            unique_results=len(vector_ids | bm25_ids),
            overlap_count=len(overlap),
            config=self.config,
        )
        
        logger.debug(
            f"Hybrid search: {len(final_results)} results in {total_time:.2f}ms "
            f"(vector: {len(vector_response.results)}, bm25: {len(bm25_response.results)}, "
            f"overlap: {len(overlap)})" 
        )
        
        return response
    
    def _fuse_results(
        self,
        vector_results: list[VectorSearchResult],
        bm25_results: list[BM25SearchResult]
    ) -> list[HybridSearchResult]:
        """
        Fuse results from vector and BM25 search.
        """
        # Build lookup dictionaries
        vector_by_id: dict[str, tuple[VectorSearchResult, int]] = {
            r.chunk_id: (r, i) for i, r in enumerate(vector_results)
        }
        bm25_by_id: dict[str, tuple[BM25SearchResult, int]] = {
            r.chunk_id: (r, i) for i, r in enumerate(bm25_results)
        }
        
        # Normalize scores if configured
        if self.config.normalize_scores:
            vector_scores = [r.score for r in vector_results]
            bm25_scores = [r.score for r in bm25_results]
            
            norm_vector = dict(zip(
                [r.chunk_id for r in vector_results],
                self._fusion.normalize_scores(vector_scores)
            ))
            norm_bm25 = dict(zip(
                [r.chunk_id for r in bm25_results],
                self._fusion.normalize_scores(bm25_scores)
            ))
        else:
            norm_vector = {r.chunk_id: r.score for r in vector_results}
            norm_bm25 = {r.chunk_id: r.score for r in bm25_results}

        # Collect all unique chunk IDs
        all_ids = set(vector_by_id.keys()) | set(bm25_by_id.keys())
        
        fused = []
        
        for chunk_id in all_ids:
            vector_info = vector_by_id.get(chunk_id)
            bm25_info = bm25_by_id.get(chunk_id)
            
            # Get score
            vector_score =  norm_vector.get(chunk_id, 0.0)
            bm25_score = norm_bm25.get(chunk_id, 0.0)
            
            # Calculate recency score if enabled
            recency_score = 0.0
            if self.config.enable_recency_boost:
                metadata = {}
                if vector_info:
                    metadata = vector_info[0].metadata
                elif bm25_info:
                    metadata = bm25_info[0].metadata
                recency_score = self._calculate_recency_score(metadata)

            
            # Fuse scores based on method
            if self.config.fusion_method == FusionMethod.WEIGHTED_SUM:
                combined_score = self._fusion.weighted_sum(
                    vector_score, bm25_score, recency_score, self.config
                )
            elif self.config.fusion_method == FusionMethod.RECIPROCAL_RANK:
                vector_rank = vector_info[1] if vector_info else None
                bm25_rank = bm25_info[1] if bm25_info else None
                combined_score = self._fusion.reciprocal_rank_fusion(
                    vector_rank, bm25_rank, self.config.rrf_k
                )
            else:
                combined_score = self._fusion.weighted_sum(
                    vector_score, bm25_score, recency_score, self.config
                )
                
            # Get content and metadata from whichever source has it
            content = ""
            metadata = {}
            document_id = ""

            if vector_info:
                content = vector_info[0].content
                metadata = vector_info[0].metadata
                document_id = vector_info[0].document_id
            elif bm25_info:
                content = bm25_info[0].content
                metadata = bm25_info[0].metadata
                document_id = bm25_info[0].document_id
                
            fused.append(HybridSearchResult(
                chunk_id=chunk_id,
                combined_score=combined_score,
                vector_score=vector_score,
                bm25_score=bm25_score,
                recency_score=recency_score,
                content=content,
                metadata=metadata,
                document_id=document_id,
                from_vector=vector_info is not None,
                from_bm25=bm25_info is not None,
            ))

        return fused
    
    def _calculate_recency_score(self, metadata: dict) -> float:
        """
        Calculate recency score based on document timestamp.
        
        Uses exponential decay: score = exp(-days / half_life)
        """
        timestamp = metadata.get(self.config.recency_field)
        if not timestamp:
            return 0.0
        
        try:
            if isinstance(timestamp, str):
                doc_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, (int, float)):
                doc_time = datetime.fromtimestamp(timestamp)
            else:
                return 0.0
            
            days_old = (datetime.now() - doc_time.replace(tzinfo=None)).days
            
            # Exponential decay
            decay_rate = math.log(2) / self.config.recency_decay_days
            score = math.exp(-decay_rate * days_old)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating recency score: {e}")
            return 0.0
    
    async def search_vector_only(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None, 
    ) -> HybridSearchResponse:
        """
        Vector-only search (for comparison/fallback).
        
        WARNING: Vector-only search WILL underperform in production.
        Use hybrid search for best results.
        """
        logger.warning(
            "Using vector-only search. This WILL underperform. "
            "Use hybrid search for production."
        )
        
        response = await self.vector_search.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter,
        )
        
        results = [
            HybridSearchResult(
                chunk_id=r.chunk_id,
                combined_score=r.score,
                vector_score=r.score,
                bm25_score=0.0,
                content=r.content,
                metadata=r.metadata,
                document_id=r.document_id,
                from_vector=True,
                from_bm25=False,
            )
            for r in response.results
        ]
        
        return HybridSearchResponse(
            results=results,
            latency_ms=response.latency_ms,
            vector_latency_ms=response.latency_ms,
            total_vector_results=len(results),
            total_bm25_results=0,
            unique_results=len(results),
            config=self.config,
        )
    
    def update_weights(
        self,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        recency_weight: Optional[float] = None,
    ) -> None:
        """
        Update fusion weights dynamically.
        
        Useful for A/B testing or adaptive weighting.
        """
        if vector_weight is not None:
            self.config.vector_weight = vector_weight
        if bm25_weight is not None:
            self.config.bm25_weight = bm25_weight
        if recency_weight is not None:
            self.config.recency_weight = recency_weight
        
        logger.info(
            f"Updated weights: vector={self.config.vector_weight}, "
            f"bm25={self.config.bm25_weight}, recency={self.config.recency_weight}"
        )
    
    def get_stats(self) -> dict:
        """Get hybrid search statistics and configuration."""
        return {
            "vector_weight": self.config.vector_weight,
            "bm25_weight": self.config.bm25_weight,
            "recency_weight": self.config.recency_weight,
            "fusion_method": self.config.fusion_method.value,
            "vector_top_k": self.config.vector_top_k,
            "bm25_top_k": self.config.bm25_top_k,
            "final_top_k": self.config.final_top_k,
        }
        

def create_hybrid_search(
    vector_search: VectorSearch,
    bm25_search: BM25Search,
    vector_weight: float = 5.0,
    bm25_weight: float = 3.0,
    recency_weight: float = 0.2,
    fusion_method: str = "weighted_sum",
    **kwargs,
) -> HybridSearch:
    """
    Factory function to create hybrid search.
    
    FOCUS: Score = (Vector * 5) + (BM25 * 3) + (Recency * 0.2)
    
    Args:
        vector_search: Vector search instance
        bm25_search: BM25 search instance
        vector_weight: Weight for vector scores (default: 5.0)
        bm25_weight: Weight for BM25 scores (default: 3.0)
        recency_weight: Weight for recency (default: 0.2)
        fusion_method: Fusion method ("weighted_sum", "rrf")
        **kwargs: Additional config options
        
    Returns:
        Configured HybridSearch instance
    """
    config = HybridSearchConfig(
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        recency_weight=recency_weight,
        fusion_method=FusionMethod(fusion_method),
        **kwargs,
    )
    
    return HybridSearch(
        vector_search=vector_search,
        bm25_search=bm25_search,
        config=config
    )