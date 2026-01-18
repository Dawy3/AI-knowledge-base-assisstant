"""Cross-Encoder Reranker"""
from sentence_transformers import CrossEncoder
from typing import List, Dict
from config.settings import settings

class BiEncoderReranker:
    """
    Post-retrieval optimization:
    - Retrieve top-100 with fast hybrid search
    - Rerank with slower but more accurate cross-encoder
    """
    
    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        self.model = CrossEncoder(model_name)
        
    def rerank(
        self,
        query:str,
        documents: List[Dict],
        top_n: int =5 
    ) -> List[Dict]: 
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Original query
            documents: Retrieved documents with metadata
            top_n: Number of documents to return
            
        Returns:
            Reranked documents with scores
        """
        if not documents:
            return []
        
        # Prepare query document pairs
        pairs = [(query, doc.get("content", "")) for doc in documents]
        
        # Get reranking scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for i, (doc, score) in enumerate(zip(documents, scores)):
            doc["rerank_score"] = float(score)
            doc["original_rank"] = i
            
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[ :top_n]
        