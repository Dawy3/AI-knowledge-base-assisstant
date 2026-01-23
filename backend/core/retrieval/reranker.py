"""
Reranking Layer for RAG Pipeline.

FOCUS: Cross-encoder reranking
MUST: Retrieve top-100 → rerank to top-10
BUDGET: 15-20ms for reranking
EXPECTED: +5-10% precision improvement

Cross-encoders are more accurate than bi-encoders because they
process query and document together, enabling cross-attention.
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankerModel(str, Enum):
    """Available reranker models."""
    
    # Cross-encoder models (sentence-transformers)
    MS_MARCO_MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MS_MARCO_MINILM_L12 = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Larger, more accurate models
    BGE_RERANKER_BASE = "BAAI/bge-reranker-base"
    BGE_RERANKER_LARGE = "BAAI/bge-reranker-large"
    
    # Fast models for tight latency budgets
    TINY_BERT_RERANKER = "cross-encoder/ms-marco-TinyBERT-L-2-v2"


@dataclass
class RerankerConfig:
    """
    Reranker configuration.
    
    BUDGET: 15-20ms for reranking
    """
    model_name: str = RerankerModel.MS_MARCO_MINILM.value

    # Reranking parameters
    top_k_input: int = 100  # Number of candidates to rerank
    top_k_output: int = 10  # Number of results after reranking
    
    # Performance
    batch_size: int = 32  # Batch size for inference
    max_length: int = 512  # Max sequence length

    # Device
    device: str = "cpu"  # "cpu" or "cuda"
    
    # Latency budget (ms)
    latency_budget_ms: float = 20.0

@dataclass
class RerankedResult:
    """Single reranked result."""
    
    chunk_id: str
    rerank_score: float  # Cross-encoder score (0-1)
    original_score: float = 0.0  # Original retrieval score
    original_rank: int = 0
    new_rank: int = 0
    
    # Content and metadata
    content: str = ""
    metadata: dict = field(default_factory=dict)
    document_id: str = ""
    
    @property
    def rank_change(self) -> int:
        """How much the rank changed (positive = improved)."""
        return self.original_rank - self.new_rank
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "rerank_score": self.rerank_score,
            "original_score": self.original_score,
            "original_rank": self.original_rank,
            "new_rank": self.new_rank,
            "rank_change": self.rank_change,
            "content": self.content,
            "metadata": self.metadata,
            "document_id": self.document_id,
        }
        
        
@dataclass
class RerankerResult:
    """Response from reranking operation."""
    
    results: list[RerankedResult]
    query: str = ""
    
    # Performance metrics
    latency_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    
    # Quality metrics
    avg_rank_change: float = 0.0
    within_budget: bool = True
    
    @property
    def top_score(self) -> float:
        return self.results[0].rerank_score if self.results else 0.0

class BaseReranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Rerank documents for a query.
        
        Args:
            query: Search query
            documents: List of documents with 'content' key
            top_k: Number of results to return
            
        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        pass
    


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using sentence-transformers.
    
    FOCUS: Cross-encoder reranking
    EXPECTED: +5-10% precision improvement
    """
    
    def __init__(
        self,
        model_name:str = RerankerModel.MS_MARCO_MINILM.value,
        device: str = 'cpu',
        max_length: int = 512,
        batch_size : int = 32,
    ): 
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                
                self._model  = CrossEncoder(
                    self.model_name,
                    max_length= self.max_length,
                    device= self.device,
                )
                logger.info(f"Loaded cross-encoder model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []
        
        model = self._load_model()
        
        # Prepare query-document pairs
        pairs = [
            [query, doc.get("content", "")]
            for doc in documents
        ]
        
        # Run inference in executor to avoid blocking
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: model.predict(pairs, batch_size=self.batch_size)
        )
        
        # Combine with indices and sort
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return [(idx, float(score)) for idx, score in indexed_scores[:top_k]]


class NoOpReranker(BaseReranker):
    """
    No-op reranker that returns original order.
    
    Useful for A/B testing or when reranking is disabled.
    """
    
    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Return documents in original order with placeholder scores."""
        return [(i, 1.0 - i * 0.01) for i in range(min(len(documents), top_k))]
    


class Reranker:
    """
    Reranking interface for RAG pipeline.
    
    MUST: Retrieve top-100 → rerank to top-10
    BUDGET: 15-20ms for reranking
    
    Usage:
        reranker = Reranker()
        results = await reranker.rerank(query, candidates, top_k=10)
    """
    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        reranker: Optional[BaseReranker] = None,
    ):
        """
        Initialize reranker.
        
        Args:
            config: Reranker configuration
            reranker: Optional custom reranker implementation
        """
        self.config = config or RerankerConfig()
        
        if reranker:
            self._reranker = reranker
        else:
            self._reranker = CrossEncoderReranker(
                model_name=self.config.model_name,
                device=self.config.device,
                max_length=self.config.max_length,
                batch_size=self.config.batch_size,
            )
            
        logger.info(
            f"Initialized Reranker: {self.config.model_name}, "
            f"top-{self.config.top_k_input} → top-{self.config.top_k_output}"
        )
        
    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: Optional[int] = None,
    ) -> RerankerResult:
        """
        Rerank retrieval candidates.
        
        MUST: Input top-100, output top-10
        
        Args:
            query: Search query
            candidates: List of candidate documents with 'chunk_id' and 'content'
            top_k: Number of results to return (default from config)
            
        Returns:
            RerankerResult with reranked documents
        """
        start_time = time.time()
        
        output_k = top_k or self.config.top_k_output
        
        # Limit input to configured maximum
        input_candidates = candidates[:self.config.top_k_input]
        
        if not input_candidates:
            return RerankerResult(
                results=[],
                query=query,
                input_count=0,
                output_count=0,
            )
        
        # Perform reranking
        reranked = await self._reranker.rerank(
            query=query,
            documents=input_candidates,
            top_k=output_k,
        )
        
        # Build results with rank tracking
        results = []
        total_rank_change = 0
        
        for new_rank , (original_idx, score) in enumerate(reranked):
            candidate = input_candidates[original_idx]
            
            result = RerankedResult(
                chunk_id=candidate.get("chunk_id", ""),
                rerank_score=score,
                original_score=candidate.get("score", 0.0),
                original_rank=original_idx,
                new_rank=new_rank,
                content= candidate.get("content", ""),
                metadata= candidate.get("metadata", {}),
                document_id= candidate.get("document_id", ""),
            )
            results.append(result)
            total_rank_change += result.rank_change
            
        latency_ms = (time.time() - start_time) * 1000
        within_budget = latency_ms <= self.config.latency_budget_ms

        if not within_budget:
            logger.warning(
                f"Reranking exceeded latency budget: {latency_ms:.2f}ms > "
                f"{self.config.latency_budget_ms}ms. Consider using a faster model."
            )
        
        avg_rank_change = total_rank_change / len(results) if results else 0.0
        
        logger.debug(
            f"Reranked {len(input_candidates)} → {len(results)} in {latency_ms:.2f}ms, "
            f"avg rank change: {avg_rank_change:.1f}"
        )
        
        return RerankerResult(
            results=results,
            query=query,
            latency_ms=latency_ms,
            input_count=len(input_candidates),
            output_count=len(results),
            avg_rank_change=avg_rank_change,
            within_budget=within_budget,
        )
        
    
    async def rerank_from_search_results(
        self,
        query: str,
        vector_results: list,  # VectorSearchResult
        bm25_results: Optional[list] = None,  # BM25SearchResult
        top_k: Optional[int] = None,
    ) -> RerankerResult:
        """
        Convenience method to rerank from search result objects.
        
        Combines and deduplicates results before reranking.
        """
        # Collect candidates, deduplicating by chunk_id
        seen_ids = set()
        candidates = []
        
        for result in vector_results:
            if result.chunk_id not in seen_ids:
                candidates.append({
                    "chunk_id": result.chunk_id,
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata,
                    "document_id": result.document_id,
                })
                seen_ids.add(result.chunk_id)

        if bm25_results:
            for result in bm25_results:
                if result.chunk_id not in seen_ids:
                    candidates.append({
                        "chunk_id": result.chunk_id,
                        "content": result.content,
                        "score": result.score,
                        "metadata": result.metadata,
                        "document_id": result.document_id,
                    })
                    seen_ids.add(result.chunk_id)
                    
        return await self.rerank(query, candidates, top_k)

    def get_latency_stats(self) -> dict:
        """Get reranker latency statistics."""
        return {
            "model": self.config.model_name,
            "latency_budget_ms": self.config.latency_budget_ms,
            "max_input": self.config.top_k_input,
            "max_output": self.config.top_k_output,
        }


def create_reranker(
    model: str = "fast",
    device: str = "cpu",
    **kwargs,
) -> Reranker:
    """
    Factory function to create reranker.
    
    Args:
        model: Model preset ("fast", "balanced", "accurate") or model name
        device: Device to run on ("cpu" or "cuda")
        **kwargs: Additional configuration options
        
    Returns:
        Configured Reranker instance
    """
    # Model presets optimized for different use cases
    presets = {
        "fast": {
            "model_name": RerankerModel.TINY_BERT_RERANKER.value,
            "latency_budget_ms": 10.0,
        },
        "balanced": {
            "model_name": RerankerModel.MS_MARCO_MINILM.value,
            "latency_budget_ms": 20.0,
        },
        "accurate": {
            "model_name": RerankerModel.BGE_RERANKER_BASE.value,
            "latency_budget_ms": 50.0,
        },
    }
    
    if model in presets:
        config_dict = presets[model]
        config_dict["device"] = device
        config_dict.update(kwargs)
    else:
        config_dict = {
            "model_name" : model,
            "device" : device,
            **kwargs,
        }
        
    config = RerankerConfig(**config_dict)
    return Reranker(config = config)
        

        
        
        
    