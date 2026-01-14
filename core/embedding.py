import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List , Union
import hashlib
from datetime import datetime
from functools import lru_cache

from config.settings import settings

class EmbeddingManager:
    """
    Manage embeddings with production best practices:
    - Same model for index and query
    - vector normalization 
    - Batch Processing
    - Embedding Caching 
    - Query Prefixes
    - Version Control
    """
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.model_name= model_name
        self.version = settings.EMBEDDING_VERSION
        self.embedding_cache = {}
        
        
    def _get_cahce_key(self, text:str ) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize embedding vector"""
        if settings.NORMALIZE_EMBEDDING:
            return vector / np.linalg.norm(vector)
        return vector
    
    @lru_cache(maxsize=10000)
    def _embed_single_cached(self, text:str) -> np.ndarray:
        """Cache individual embedding in memeory"""   
        embedding = self.model.encode(text, convert_to_numpy=True)
        return self._normalize_vector(embedding)
    
    
    def embed_documents(self, texts: List[str], use_prefix: bool = True)-> np.ndarray:
        """
        Embed documents in batches with optional prefix
        
        Args:
            texts: List of document texts
            use_prefix: Add document prefix for better embedding quality
            
        Returns:
            Normalized embedding matrix
        """
        
        if use_prefix and settings.USE_QUERY_PREFIX:
            texts= [f"{settings.DOCUMENT_PREFIX}{text}" for text in texts]
            
            
        # Batches encode for efficiency 
        embeddings = self.model.encode(
            texts,
            batch_size= settings.BATCH_SIZE,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy= True
        )
        
        
        # Normalize all vectors
        if settings.NORMALIZE_EMBEDDING:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
        return embeddings
    
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed query with caching and prefix
        
        Args:
            query: Query text
            
        Return: 
            Normalized embedding vector 
        """
        # Add query prefix in enabled
        if settings.USE_QUERY_PREFIX:
            prefixed_query = f"{settings.QUERY_PREFIX}{query}"
        else:
            prefixed_query = query
            
        # Use cached embedding if available
        cache_key = self._get_cahce_key(prefixed_query)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Generate and cache embedding
        embedding = self._embed_single_cached(prefixed_query)
        self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def get_version_metadata(self) -> Dict :
        """Return embedding version metadata for tracking"""
        return {
            "model_name" : self.model_name,
            "version" : self.version,
            "dimension" : settings.EMBEDDING_DIMENSION,
            "normalized" : settings.NORMALIZE_EMBEDDING,
            "timestamp" : datetime.now().isoformat()
        }

        
        
        
        
        
     