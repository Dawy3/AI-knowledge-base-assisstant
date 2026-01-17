"""
Main Retrieval Pipeline:
- Query routing
- Multi-query generation
- Hybrid search
- Reranking
- Result fusion
"""
from typing import List, Dict, Optional
import logging
from collections import defaultdict

from core.embeddings import EmbeddingManager
from core.hybrid_search import HybridSearchEngine
from retrieval.query_router import QueryRouter, QueryType
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import BiEncoderReranker
from config.settings import settings

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    """
    Production ready retrieval pipeline with all optimization 
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        hybrid_search : HybridSearchEngine,
        enable_routing : bool = settings.ENABLE_QUERY_ROUTING,
        enable_reranking: bool = settings.USE_RERANKING
    ):
        self.embedding_manager = embedding_manager
        self.hybrid_search = hybrid_search
        self.enable_routing = enable_routing
        self.enable_reranking = enable_reranking
        
        # Initialize component 
        if self.enable_routing:
            self.router = QueryRouter(use_llm=True)
        
        self.query_rewriter = QueryRewriter()
        
        if self.enable_reranking:
            self.reranker = BiEncoderReranker()
            
        
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_multi_query: bool = True,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Main retrieval pipeline
        
        Args:
            query: User query
            top_k: Number of final results to return
            use_multi_query: Enable multi-query generation
            filter_dict: Pinecone metadata filter
            
        Returns:
            Dict with results and metadata
        """
        logger.info(f"Retrieve for query: {query}")
        
        # Step 1: Query Routing (Optinal)
        if self.enable_routing:
            classification = self.router.classify_query(query)
            logger.info(f"Query classified as: {classification['type']}")
            
            if classification['type'] == QueryType.REJECTION:
                return {
                    "status" : "rejected",
                    "reason" : classification['reasoning'],
                    "results": []
                }
            
            if classification["type"] == QueryType.CLARIFICATION: # Ambigous query
                return {
                    "status" : "need_clarification",
                    "reason" : classification["reasoning"],
                    "results": []
                }
                
        # Step 2: Multi-query Generation (optional)
        queries = [query]
        if use_multi_query:
            queries = self.query_rewriter.rewrite_query(query, num_variants=3)
            logger.info(f"Generated {len(queries)} query variants")
            
        # Step 3: Retrieve for each query variant
        all_results = []
        for q in queries:
            # Generate embedding
            query_embedding = self.embedding_manager.embed_query(q)
            
            # Hybrid search (vector + keyword + recency)
            results  = self.hybrid_search.search(
                query= q,
                query_embedding= query_embedding,
                k = settings.INITIAL_RETRIEVAL_K,
                filter_func= self._create_filter_func(filter_dict) if filter_dict else None
            )
            
            all_results.append(results)
            
        # Step 4: Duplicate and merge results
        merged_results = self._merge_results(all_results)
        logger.info(f"Merged to {len(merged_results)} unique results")
        
        # Step 5: Reranking (optional)
        if self.enable_reranking and merged_results:
            # Prepare documents for reranking
            docs_for_rerank = [
                {
                    "content" : self._get_chunk_content(r["id"]),
                    **r
                }
                for r in merged_results[:settings.INITIAL_RETRIEVAL_K]
            ]
            
            final_results = self.reranker.rerank(
                query=query,
                documents= docs_for_rerank,
                top_n= top_k
            )
            logger.info(f"Reranked to top {len(final_results)} results")
        else:
            final_results = merged_results[:top_k]    
        
        return {
            "status": "success",
            "query": query,
            "num_variants": len(queries),
            "total_retrieved": len(all_results),
            "after_dedup": len(merged_results),
            "final_count": len(final_results),
            "results": final_results
        }
    
    def _create_filter_func(self, filter_dict: Dict) -> callable:
        """
        Create filter function from Pinecone filter dict
        Note: This is a placeholder - actual filtering happens in Pinecone
        """
        def filter_func(metadata: Dict) -> bool:
            # This would be used for post-filtering if needed
            # In production with Pinecone, pre-filtering is preferred
            for key, condition in filter_dict.itmes(): 
                if "@eq" in condition:
                    if metadata.get(key) != condition["@eq"]: # If this chunk’s metadata does NOT match the filter → reject it
                        return False
            return True
        
        return filter_func
    
    def _get_chunk_content(self, chunk_id: int) -> str:
        """
        Retrieve chunk content from vector store
        
        In production, you might want to:
        - Cache frequently accessed chunks
        - Store content separately in a document store
        """
        # Access from hybrid search corpus
        if chunk_id < len(self.hybrid_search.corpus):
            return self.hybrid_search.corpus[chunk_id]
        return ""
                
    def _merge_results(self, results: List[Dict]) -> List[Dict]:
        """
        Merge and deduplicate results from multiple queries
        Uses Reciprocal Rank Fusion (RRF)
        """
        if not results:
            return []
        
        # Group by document ID
        doc_scores = defaultdict(lambda: {
            "hybrid_score": 0.0,
            "count": 0,
            "metadata": None,
            "id": None
        })
        
        K = 60  # RRF constant
        
        for rank, result in enumerate(results):
            doc_id = result["id"]
            
            # Reciprocal Rank Fusion score
            rrf_score = 1.0 / (K + rank + 1)
            
            doc_scores[doc_id]["hybrid_score"] += rrf_score
            doc_scores[doc_id]["count"] += 1
            doc_scores[doc_id]["id"] = doc_id
            
            if doc_scores[doc_id]["metadata"] is None:
                doc_scores[doc_id]["metadata"] = result.get("metadata", {})
        
        # Convert to list and sort
        merged = list(doc_scores.values())
        merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return merged

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve for multiple queries in batch
        
        Args:
            queries: List of queries
            top_k: Results per query
            
        Returns:
            List of retrieval results
        """
        results = []
        for query in queries:
            result = self.retrieve(query, top_k=top_k)
            results.append(result)
        
        return results
    
    def get_similar_documents(
        self,
        doc_id: int,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find documents similar to a given document
        
        Args:
            doc_id: Document ID
            top_k: Number of similar docs to return
            
        Returns:
            List of similar documents
        """
        # Get the document's embedding
        # This assumes you have access to stored embeddings
        # In production, retrieve from vector store
        content = self._get_chunk_content(doc_id)
        
        if not content:
            return []
        
        # Embed the content
        embedding = self.embedding_manager.embed_documents([content])[0]
        
        # Search for similar documents
        results = self.hybrid_search.search(
            query=content,
            query_embedding=embedding,
            k=top_k + 1  # +1 to exclude self
        )
        
        # Filter out the original document
        similar = [r for r in results if r["id"] != doc_id]
        
        return similar[:top_k]