"""
Embedding Model Configuration.

FOCUS: Domain-matched model selection
MUST: Verify max tokens handle your chunks

Model selection guidelines:
- General purpose: OpenAI text-embedding-3-small/large
- Code/Technical: CodeBERT, StarCoder embeddings
- Legal: Legal-BERT
- Medical: PubMedBERT, BioBERT
- Multilingual: multilingual-e5-large

CRITICAL: Index and query embeddings MUST use the same model.
Changing models requires full reindexing.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EmbeddingModelProvider(str, Enum):
    """Support embedding model providers."""
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    VOYAGE = "voyage"
    LOCAL = "local"
    AZURE = "azure"
    
@dataclass
class EmbeddingModel:
    """
    Embedding model specification.
    
    Contains all configuration needed to use an embedding model
    consistently across indexing and querying.
    """
    
    # Model identification
    name: str
    provider: EmbeddingModelProvider
    version: str = "v1"
    
    # Model specifications
    dimensions: int = 1536
    max_tokens: int = 8191  # MUST handle your chunk size
    
    # Performance characteristics
    batch_size: int = 100  # Recommended batch size
    max_batch_size: int = 500  # Maximum safe batch size
    
    # API configuration
    api_endpoint: Optional[str] = None
    api_key_env: str = ""  # Environment variable name for API key
    
    # Cost tracking (per 1M tokens)
    cost_per_million_tokens: float = 0.0
    
    # Domain specialization
    domain: str = "general"  # general, code, legal, medical, etc.
    
    # Additional model parameters
    normalize_embeddings: bool = True
    truncation_strategy: str = "end"  # end, start, middle
    
    # Metadata
    description: str = ""
    
    @property
    def model_id(self) -> str:
        """Unique model identifier for tracking."""
        return f"{self.provider.value}/{self.name}/{self.version}"
    
    def validate_chunk_size(self, chunk_tokens: int) -> bool:
        """
        Validate that chunk size fits within model's max tokens.
        
        MUST: Verify max tokens handle your chunks.
        """
        return chunk_tokens <= self.max_tokens
    
    def get_recommended_batch_size(self, avg_chunk_tokens: int) -> int:
        """
        Calculate recommended batch size based on chunk size.
        
        Larger chunks = smaller batches to avoid API limits.
        """
        # Estimate based on total tokens per batch
        # Most APIs have ~100k token limit per batch
        tokens_per_batch = 100_000
        estimated_batch = tokens_per_batch // max(avg_chunk_tokens, 100)
        
        return min(estimated_batch, self.max_batch_size)
    
    
SUPPORTED_MODELS: dict[str, EmbeddingModel] = {
    # OpenAI Models
    "openai/text-embedding-3-small": EmbeddingModel(
        name="text-embedding-3-small",
        provider=EmbeddingModelProvider.OPENAI,
        version="v1",
        dimensions=1536,
        max_tokens=8191,
        batch_size=100,
        max_batch_size=500,
        api_key_env="OPENAI_API_KEY",
        cost_per_million_tokens=0.02,
        domain="general",
        description="OpenAI's efficient embedding model. Good balance of quality and cost.",
    ),
    "openai/text-embedding-3-large": EmbeddingModel(
        name="text-embedding-3-large",
        provider=EmbeddingModelProvider.OPENAI,
        version="v1",
        dimensions=3072,
        max_tokens=8191,
        batch_size=100,
        max_batch_size=500,
        api_key_env="OPENAI_API_KEY",
        cost_per_million_tokens=0.13,
        domain="general",
        description="OpenAI's highest quality embedding model.",
    ),
    "openai/text-embedding-ada-002": EmbeddingModel(
        name="text-embedding-ada-002",
        provider=EmbeddingModelProvider.OPENAI,
        version="v2",
        dimensions=1536,
        max_tokens=8191,
        batch_size=100,
        max_batch_size=500,
        api_key_env="OPENAI_API_KEY",
        cost_per_million_tokens=0.10,
        domain="general",
        description="Legacy OpenAI model. Consider text-embedding-3-small instead.",
    ),
    
    # Cohere Models
    "cohere/embed-english-v3.0": EmbeddingModel(
        name="embed-english-v3.0",
        provider=EmbeddingModelProvider.COHERE,
        version="v3",
        dimensions=1024,
        max_tokens=512,
        batch_size=96,
        max_batch_size=96,
        api_key_env="COHERE_API_KEY",
        cost_per_million_tokens=0.10,
        domain="general",
        description="Cohere's English embedding model with excellent retrieval quality.",
    ),
    "cohere/embed-multilingual-v3.0": EmbeddingModel(
        name="embed-multilingual-v3.0",
        provider=EmbeddingModelProvider.COHERE,
        version="v3",
        dimensions=1024,
        max_tokens=512,
        batch_size=96,
        max_batch_size=96,
        api_key_env="COHERE_API_KEY",
        cost_per_million_tokens=0.10,
        domain="multilingual",
        description="Cohere's multilingual model supporting 100+ languages.",
    ),
    
    # Voyage AI Models
    "voyage/voyage-large-2": EmbeddingModel(
        name="voyage-large-2",
        provider=EmbeddingModelProvider.VOYAGE,
        version="v2",
        dimensions=1536,
        max_tokens=16000,
        batch_size=128,
        max_batch_size=128,
        api_key_env="VOYAGE_API_KEY",
        cost_per_million_tokens=0.12,
        domain="general",
        description="Voyage's high-quality embedding model with long context.",
    ),
    "voyage/voyage-code-2": EmbeddingModel(
        name="voyage-code-2",
        provider=EmbeddingModelProvider.VOYAGE,
        version="v2",
        dimensions=1536,
        max_tokens=16000,
        batch_size=128,
        max_batch_size=128,
        api_key_env="VOYAGE_API_KEY",
        cost_per_million_tokens=0.12,
        domain="code",
        description="Voyage's code-specialized embedding model.",
    ),
    
    # HuggingFace / Local Models
    "huggingface/e5-large-v2": EmbeddingModel(
        name="intfloat/e5-large-v2",
        provider=EmbeddingModelProvider.HUGGINGFACE,
        version="v2",
        dimensions=1024,
        max_tokens=512,
        batch_size=32,
        max_batch_size=64,
        api_key_env="HF_TOKEN",
        cost_per_million_tokens=0.0,  # Free if self-hosted
        domain="general",
        description="Open-source model with excellent quality. Good for self-hosting.",
    ),
    "huggingface/multilingual-e5-large": EmbeddingModel(
        name="intfloat/multilingual-e5-large",
        provider=EmbeddingModelProvider.HUGGINGFACE,
        version="v1",
        dimensions=1024,
        max_tokens=512,
        batch_size=32,
        max_batch_size=64,
        api_key_env="HF_TOKEN",
        cost_per_million_tokens=0.0,
        domain="multilingual",
        description="Multilingual E5 model supporting 100+ languages.",
    ),
    "huggingface/bge-large-en-v1.5": EmbeddingModel(
        name="BAAI/bge-large-en-v1.5",
        provider=EmbeddingModelProvider.HUGGINGFACE,
        version="v1.5",
        dimensions=1024,
        max_tokens=512,
        batch_size=32,
        max_batch_size=64,
        api_key_env="HF_TOKEN",
        cost_per_million_tokens=0.0,
        domain="general",
        description="BGE model with excellent retrieval performance.",
    ),
    "huggingface/minilm-l6-v2": EmbeddingModel(
    name="sentence-transformers/all-MiniLM-L6-v2",
    provider=EmbeddingModelProvider.HUGGINGFACE,
    version="v1",
    dimensions=384,
    max_tokens=256,
    batch_size=64,
    max_batch_size=128,
    api_key_env="HF_TOKEN",
    cost_per_million_tokens=0.0,
    domain="general",
    description="Lightweight, fast, low-quality baseline. Suitable for demos and small datasets only.",
    ),
    
    # Domain-specific models
    "huggingface/legal-bert": EmbeddingModel(
        name="nlpaueb/legal-bert-base-uncased",
        provider=EmbeddingModelProvider.HUGGINGFACE,
        version="v1",
        dimensions=768,
        max_tokens=512,
        batch_size=32,
        max_batch_size=64,
        api_key_env="HF_TOKEN",
        cost_per_million_tokens=0.0,
        domain="legal",
        description="BERT fine-tuned on legal documents.",
    ),
    "huggingface/pubmedbert": EmbeddingModel(
        name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        provider=EmbeddingModelProvider.HUGGINGFACE,
        version="v1",
        dimensions=768,
        max_tokens=512,
        batch_size=32,
        max_batch_size=64,
        api_key_env="HF_TOKEN",
        cost_per_million_tokens=0.0,
        domain="medical",
        description="BERT fine-tuned on PubMed abstracts.",
    ),
}

def get_embedding_model(
    model_key: str,
    custom_config: Optional[dict] = None,
) -> EmbeddingModel:
    """
    Get embedding model by key with optional customization.
    
    Args:
        model_key: Model identifier (e.g., "openai/text-embedding-3-small")
        custom_config: Optional dict to override model settings
        
    Returns:
        Configured EmbeddingModel instance
        
    Example:
        >>> model = get_embedding_model("openai/text-embedding-3-small")
        >>> model.validate_chunk_size(512)
        True
    """
    if model_key not in SUPPORTED_MODELS:
        available = list(SUPPORTED_MODELS.keys())
        raise ValueError(
            f"Unknown model: {model_key}. Availabel models: {available}"
        )
        
    model = SUPPORTED_MODELS[model_key]
    
    # Apply custom configuration if provided
    if custom_config:
        model_dict = {
            "name": model.name,
            "provider": model.provider,
            "version": model.version,
            "dimensions": model.dimensions,
            "max_tokens": model.max_tokens,
            "batch_size": model.batch_size,
            "max_batch_size": model.max_batch_size,
            "api_endpoint": model.api_endpoint,
            "api_key_env": model.api_key_env,
            "cost_per_million_tokens": model.cost_per_million_tokens,
            "domain": model.domain,
            "normalize_embeddings": model.normalize_embeddings,
            "truncation_strategy": model.truncation_strategy,
            "description": model.description,
        }
        model_dict.update(custom_config)
        model = EmbeddingModel(**model_dict)
    
    return model


def get_model_for_domain(domain: str) -> EmbeddingModel:
    """
    Get recommended model for a specific domain.
    
    Args:
        domain: Domain name (general, code, legal, medical, multilingual)
        
    Returns:
        Recommended EmbeddingModel for the domain
    """
    domain_recommendations = {
        "general": "openai/text-embedding-3-small",
        "code": "voyage/voyage-code-2",
        "legal": "huggingface/legal-bert",
        "medical": "huggingface/pubmedbert",
        "multilingual": "cohere/embed-multilingual-v3.0",
    }
    
    model_key = domain_recommendations.get(domain, "openai/text-embedding-3-small")
    return get_embedding_model(model_key)


def list_models_by_provider(provider: EmbeddingModelProvider) -> list[EmbeddingModel]:
    """List all models from a specific provider."""
    return [
        model for model in SUPPORTED_MODELS.values()
        if model.provider == provider
    ]
        
def validate_model_compatibility(
    index_model_id: str,
    query_model_id: str,
) -> bool:
    """
    CRITICAL: Validate that index and query models match.
    
    Mismatched models will produce garbage results.
    """
    return index_model_id == query_model_id
    