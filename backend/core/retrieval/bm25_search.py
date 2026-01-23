"""
BM25 Keyword Search Implementation for RAG Pipeline.

FOCUS: Exact term matching (IDs, codes, names)
MUST: Index with proper tokenization

BM25 excels at:
- Exact ID/code matching (e.g., "ERR-404", "SKU-12345")
- Named entity matching
- Acronym matching
- Technical terms

Used in combination with vector search for hybrid retrieval.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from rank_bm25 import BM25Okapi, BM25Plus

logger = logging.getLogger(__name__)


@dataclass
class BM25SearchResult:
    """Result from BM25 search."""
    
    chunk_id: str
    score: float  # BM25 score (higher is better, not normalized)
    content: str = ""
    metadata: dict = field(default_factory=dict)
    
    # Source tracking
    document_id: str = ""
    chunk_index: int = 0
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
        }


@dataclass
class BM25SearchResponse:
    """Response containing BM25 search results."""
    
    results: list[BM25SearchResult]
    query_tokens: list[str] = field(default_factory=list)
    
    # Performance metrics
    latency_ms: float = 0.0
    total_documents: int = 0
    
    @property
    def top_score(self) -> float:
        return self.results[0].score if self.results else 0.0


class BM25Tokenizer:
    """
    Tokenizer for BM25 indexing and search.
    
    MUST: Index with proper tokenization for exact matching.
    
    Features:
    - Lowercase normalization
    - Punctuation handling
    - Special character preservation for IDs/codes
    - Optional stemming
    """
    
    # Pattern to match words, numbers, and codes (e.g., ERR-404, SKU_123)
    TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*')
    
    # Common stopwords to optionally filter
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
    }
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        preserve_special: bool = True,
        min_token_length: int = 1,
        custom_tokenizer: Optional[Callable[[str], list[str]]] = None,
    ):
        """
        Initialize tokenizer.
        
        Args:
            lowercase: Convert to lowercase
            remove_stopwords: Remove common stopwords
            preserve_special: Preserve special characters in IDs/codes
            min_token_length: Minimum token length to keep
            custom_tokenizer: Optional custom tokenization function
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.preserve_special = preserve_special
        self.min_token_length = min_token_length
        self.custom_tokenizer = custom_tokenizer
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        if self.custom_tokenizer:
            return self.custom_tokenizer(text)
        
        # Lowercase if configured
        if self.lowercase:
            text = text.lower()
        
        # Extract tokens using pattern
        tokens = self.TOKEN_PATTERN.findall(text)
        
        # Filter by length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        
        # Remove stopwords if configured
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.STOPWORDS]
        
        return tokens
    
    def tokenize_query(self, query: str) -> list[str]:
        """
        Tokenize search query.
        
        Same as tokenize but can have different handling for queries.
        """
        return self.tokenize(query)


class BM25Index:
    """
    BM25 index for a collection of documents.
    
    Supports incremental updates and efficient search.
    """
    
    def __init__(
        self,
        tokenizer: Optional[BM25Tokenizer] = None,
        algorithm: str = "okapi",  # "okapi" or "plus"
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize BM25 index.
        
        Args:
            tokenizer: Tokenizer instance
            algorithm: BM25 variant ("okapi" or "plus")
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.tokenizer = tokenizer or BM25Tokenizer()
        self.algorithm = algorithm
        self.k1 = k1
        self.b = b
        
        # Index storage
        self._documents: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: Optional[BM25Okapi | BM25Plus] = None
        self._id_to_index: dict[str, int] = {}
    
    def build(
        self,
        documents: list[dict],
    ) -> int:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of documents with 'chunk_id' and 'content'
            
        Returns:
            Number of documents indexed
        """
        self._documents = documents
        self._tokenized_corpus = []
        self._id_to_index = {}
        
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            tokens = self.tokenizer.tokenize(content)
            self._tokenized_corpus.append(tokens)
            self._id_to_index[doc["chunk_id"]] = i
        
        # Build BM25 index
        if self.algorithm == "plus":
            self._bm25 = BM25Plus(self._tokenized_corpus, k1=self.k1, b=self.b)
        else:
            self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        
        logger.info(f"Built BM25 index with {len(documents)} documents")
        return len(documents)
    
    def add_documents(self, documents: list[dict]) -> int:
        """
        Add documents to existing index.
        
        Note: This rebuilds the index. For frequent updates,
        consider batch updates or a different approach.
        """
        all_docs = self._documents + documents
        return self.build(all_docs)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[tuple[int, float]]:
        """
        Search the index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum score threshold
            
        Returns:
            List of (document_index, score) tuples
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built")
            return []
        
        query_tokens = self.tokenizer.tokenize_query(query)
        
        if not query_tokens:
            return []
        
        scores = self._bm25.get_scores(query_tokens)
        
        # Get top-k indices with scores
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed_scores[:top_k]:
            if score > score_threshold:
                results.append((idx, float(score)))
        
        return results
    
    def get_document(self, index: int) -> Optional[dict]:
        """Get document by index."""
        if 0 <= index < len(self._documents):
            return self._documents[index]
        return None
    
    @property
    def size(self) -> int:
        """Number of documents in index."""
        return len(self._documents)


class BM25Search:
    """
    BM25 search interface for RAG pipeline.
    
    FOCUS: Exact term matching (IDs, codes, names)
    
    Usage:
        bm25 = BM25Search()
        bm25.index(chunks)
        results = bm25.search("ERR-404")
    """
    
    def __init__(
        self,
        tokenizer: Optional[BM25Tokenizer] = None,
        algorithm: str = "okapi",
        k1: float = 1.5,
        b: float = 0.75,
        default_top_k: int = 100,
    ):
        """
        Initialize BM25 search.
        
        Args:
            tokenizer: Custom tokenizer
            algorithm: BM25 variant
            k1: Term frequency saturation
            b: Length normalization
            default_top_k: Default number of results
        """
        self.default_top_k = default_top_k
        self._index = BM25Index(
            tokenizer=tokenizer,
            algorithm=algorithm,
            k1=k1,
            b=b,
        )
    
    def index(self, chunks: list[dict]) -> int:
        """
        Index chunks for BM25 search.
        
        Args:
            chunks: List of chunks with 'chunk_id' and 'content'
            
        Returns:
            Number of chunks indexed
        """
        return self._index.build(chunks)
    
    def add_chunks(self, chunks: list[dict]) -> int:
        """Add chunks to existing index."""
        return self._index.add_documents(chunks)
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: float = 0.0,
    ) -> BM25SearchResponse:
        """
        Search for matching chunks.
        
        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum score
            
        Returns:
            BM25SearchResponse with results
        """
        start_time = time.time()
        
        k = top_k or self.default_top_k
        
        search_results = self._index.search(
            query=query,
            top_k=k,
            score_threshold=score_threshold,
        )
        
        results = []
        for idx, score in search_results:
            doc = self._index.get_document(idx)
            if doc:
                results.append(BM25SearchResult(
                    chunk_id=doc["chunk_id"],
                    score=score,
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    document_id=doc.get("document_id", ""),
                    chunk_index=doc.get("chunk_index", 0),
                ))
        
        latency_ms = (time.time() - start_time) * 1000
        
        query_tokens = self._index.tokenizer.tokenize_query(query)
        
        logger.debug(
            f"BM25 search for '{query}' ({len(query_tokens)} tokens) "
            f"returned {len(results)} results in {latency_ms:.2f}ms"
        )
        
        return BM25SearchResponse(
            results=results,
            query_tokens=query_tokens,
            latency_ms=latency_ms,
            total_documents=self._index.size,
        )
    
    def normalize_scores(
        self,
        results: list[BM25SearchResult],
        max_score: Optional[float] = None,
    ) -> list[BM25SearchResult]:
        """
        Normalize BM25 scores to 0-1 range.
        
        Useful for combining with vector search scores.
        """
        if not results:
            return results
        
        max_s = max_score or max(r.score for r in results)
        
        if max_s <= 0:
            return results
        
        for result in results:
            result.score = result.score / max_s
        
        return results
    
    @property
    def index_size(self) -> int:
        """Number of documents in the index."""
        return self._index.size


def create_bm25_search(
    remove_stopwords: bool = False,
    preserve_special: bool = True,
    **kwargs,
) -> BM25Search:
    """
    Factory function to create BM25 search.
    
    Args:
        remove_stopwords: Whether to remove stopwords
        preserve_special: Preserve special characters in codes
        **kwargs: Additional arguments for BM25Search
        
    Returns:
        Configured BM25Search instance
    """
    tokenizer = BM25Tokenizer(
        remove_stopwords=remove_stopwords,
        preserve_special=preserve_special,
    )
    
    return BM25Search(tokenizer=tokenizer, **kwargs)

