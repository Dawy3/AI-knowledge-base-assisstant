from pydantic import BaseSetting
from typing import Optional
import yaml

class RAGSettings(BaseSetting):
    # embedding configuration
    EMBEDDING_MODEL: str = "sentence-transformeres/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_VERSION: str = "v1.0.0"
    NORMALIZE_EMBEDDING: bool = True
    BATCH_SIZE: int =32
    USE_QUERY_PREFIX: bool = True
    QUERY_PREFIX: str = "query: "
    DOCUMENT_PREFIX : str = "passage: "
    
    # Chunking configuration 
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    CHUNKING_METHOD: str = "recursive"
    
    # Vector Store Configuration
    PINECONE_API_KEY: str

    # INDEX_TYPE: str = "HNSW"
    # HNSW_M: int = 16
    # HNSW_EF_CONSTRUCTION: int = 200
    # HNSW_EF_SEARCH: int = 100
    
    # Retrieval Configuration
    HYBRID_ALPHA: float = 0.5 # 0=keywrod only , 1= vector only
    INITIAL_RETRIEVAL_K: int = 100
    RERANK_TOP_N: int = 5
    USE_RERANKING: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Caching Configuration
    ENABLE_SEMANTIC_CACHE: bool = True
    CACHE_SIMILARITY_THRESHOL: float = 0.95
    CACHE_TTL: int = 3600 # 1 hour 
   
   # Query Routing
    ENABLE_QUERY_ROUTING: bool = True
    ROUTING_MODEL: str = "gpt-4o-mini"
    
    # Monitoring & Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True
    ENABLE_AB_TESTING: bool = True
    
    # Generation
    LLM_MODEL: str = "gpt-4o"
    MAX_CONTEXT_LENGTH: int = 4000
    TEMPERATURE: float = 0.7
    
    class Config:
        env_file = ".env"

settings = RAGSettings() 