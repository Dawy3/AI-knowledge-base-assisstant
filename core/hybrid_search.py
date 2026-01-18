from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Optional

from core.vector_store import PineconeVectorStore

class HybridSearchEngine:
    """
    Combine vector search (semantic) with keyword search (BM25), and recency
    for robust retrieval using a custom weight formula.
    """ 
    # Configurable weights based on the formula:
    # Score = (VectorScore * 5) + (BM25 * 3) + (RecencyScore * 0.2)
    VECTOR_WEIGHT = 5.0
    KEYWORD_WEIGHT = 3.0
    RECENCY_WEIGHT = 0.2
    
    def __init__(self, vector_store: PineconeVectorStore):
        self.vector_store = vector_store
        
        # BM25 for keywrod search
        self.bm25 = None
        self.doc_ids = []
        self.corpus = []
        
    def index_documents(self, doc_ids: List[int], texts: List[str]):
        """Index Document for keyword search"""
        self.doc_ids = doc_ids
        self.corpus = texts
        
        # Tokenize and build BM25 index
        tokenized_corpus = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    
    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 100,
        filter_func: Optional[callable] = None
    ) -> List[Dict]:
        """
        Hybrid Search combining vector, keyword, and recency signals.
        Formula: (Vector * 5) + (BM25 * 3) + (Recency * 0.2) 
        """
        # 1. Vector Search
        vector_ids, vector_scores = self.vector_store.search(
            query_embedding,
            k=k,
            filter_func= filter_func
        ) 
        
        # 2. Keyword search (BM25)
        tokenized_query = query.lower().split()
        keyword_scores = self.bm25.get_scores(tokenized_query)
        
        # 3. Normalize base score to [0, 1]
        vector_scores_norm = self._normalize_scores(vector_scores)
        keyword_scores_norm = self._normalize_scores(keyword_scores)
        
        # Temporary storage for fusion
        results = {}
        
        # --- Stage A: Vector Contribution ---
        for doc_id, score in zip(vector_ids, vector_scores_norm):
            results[doc_id] = {
                "id" : doc_id,
                "vector_score" : score,
                "keyword_score": 0.0,
                "recency_score": 0.0,
                "metadata" : self.vector_store.id_to_metadata.get(doc_id, {})
            }
            
        # --- Stage B: Keyword Contribution ---
        for doc_id, score in zip(self.doc_ids, keyword_scores_norm):
            if doc_id not in results:
                # If doc was not found by vector search, init it
                results[doc_id] = {
                    "id" : doc_id,
                    "vector_score" : 0.0,
                    "keyword_score": score,
                    "recency_score": 0.0,
                    "metadata" : self.vector_store.id_to_metadata.get(doc_id, {})
                }
            else:
                results[doc_id]["keyword_score"] = score
                
        # --- Stage C: Recency Contribution ---
        # Extract timestamps and normalize them to create RecencyScore
        # Assumes metadata has a 'timestamp' or 'created_at' field (numeric or sortable)
        timestamp = []
        doc_ids_list = list(results.keys())
        
        for doc_id in doc_ids_list:
            meta = results[doc_id]["metadata"]
            # Default to 0 if no timestamp found
            ts = meta.get("timestamp", meta.get("created_at", 0))
            timestamp.append(ts)
            
        recency_score_norm = self._normalize_scores(timestamp)
        
        # Apply RecencyScore to results
        for doc_id, r_score in zip(doc_ids_list, recency_score_norm):
            results[doc_id]["recency_score"] = r_score
            
        # --- Stage D: Final Weighted Formula ---
        # Score = (vector * 5) + (BM25 * 3) + (Recency * 0.2)
        final_results = []
        for doc_id, data in results.items():
            final_score = (
                (data["vector_score"] * self.VECTOR_WEIGHT) +
                (data["keyword_score"] * self.KEYWORD_WEIGHT) +
                (data["recency_score"] * self.RECENCY_WEIGHT) 
            )
            data["hybrid_score"] = final_score
            final_results.append(data)
            
        # Sort by hybrid score and return top-k
        sorted_results = sorted(
            final_results,
            key=lambda x: x["hybrid_score"],
            reverse= True
        )
        
        return sorted_results[:k]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalization to scale values between 0 and 1"""
        if not scores:
            return []
        
        scores_arr = np.array(scores)
        min_score = scores_arr.min()
        max_score = scores_arr.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return ((scores_arr - min_score) / (max_score - min_score)).tolist()        
        
        