from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pydantic import Field

class RAGSettings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file= ".env",
        env_file_encoding= "utf-8",
        case_sensitive=True
    )
    
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
    
    OPENROUTER_API_KEY :str = Field(..., description="Openrrouter API key")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM : str = Field(default="gpt-4o-mini", description="The generator model")
    EMBEDDING_MODEL : str = "sentence-transformers/all-MiniLM-L6-v2"
    DIMENSION: int = 384
    
    # LangSmith (Monitoring)
    LANGSMITH_API_KEY : Optional[str] = Field(default=None, description="LangSmith API key")
    LANGSMITH_PROJECT : str = Field(default="crm-ai-assistant", description="LangSmith API key for tracing")
    LANGSMITH_ENDPOINT: str= Field(default="https://api.smith.langchain.com", description="LangSmith endpoint")
    
    # Database (PostgreSQL for checkpointing)
    POSTGRES_HOST: str = Field(default="localhost", description="PostgerSQL host")
    POSTGRS_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_USER: str = Field(default="postgres", description="PostgreSQL user")
    POSTGRES_PASSWORD: str = Field(default="postgres", description="PostgreSQL Password")
    POSTGRES_DB: str = Field(default="crm_ai", description="PostgreSQL database name")
    
    @property
    def POSTGRES_URL(self) -> str:
        """Build PostgreSQL connection URL"""
        return f"posgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRS_PORT}/{self.POSTGRES_DB}"
    
    #Redis (semantic caching and session management)
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis Password")
    
    @property
    def REDIS_URL(self) -> str:
        """Build Redis connection URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


    
    # Retrieval Configuration
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
    ROUTING_MODEL: str = "tngtech/deepseek-r1t2-chimera:free" #"gpt-4o-mini"
    
    # Monitoring & Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True
    ENABLE_AB_TESTING: bool = True
    
    # Generation
    MAX_CONTEXT_LENGTH: int = 4000
    
    

settings = RAGSettings() 