"""
Reranker: Re-rank retrieved documents using cross-encoder
"""
from typing import List, Dict
import logging
from sentence_transformers import CrossEncoder

from config.settings import settings

logger = logging.getLogger(__name__)


class BiEncoderReranker:
    """
    Rerank retrieved documents using a cross-encoder model
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        logger.info(f"Loading reranker model: {self.model_name}")
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info("✅ Reranker model loaded")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_n: int = 5
    ) -> List[Dict]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: User query
            documents: List of document dicts with 'content' key
            top_n: Number of top results to return
            
        Returns:
            Reranked list of documents
        """
        if not self.model or not documents:
            return documents[:top_n]
        
        try:
            # Prepare pairs for scoring
            pairs = [[query, doc.get("content", "")] for doc in documents]
            
            # Get scores
            scores = self.model.predict(pairs)
            
            # Add rerank scores to documents
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            
            # Sort by rerank score
            reranked = sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return reranked[:top_n]
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return documents[:top_n]

