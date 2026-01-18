# ============================================================================
# rag_system.py - Core RAG System (Separated from FastAPI)
# ============================================================================

"""
Core RAG System - Can be used by:
1. FastAPI application (main.py)
2. Evaluation scripts (retrieval_eval.py, generation_eval.py)
3. Testing scripts
4. Jupyter notebooks
"""

import os
import logging
from typing import List, Dict, Optional
import redis
from openai import OpenAI

from config.settings import settings
from core.embeddings import EmbeddingManager
from core.vector_store import PineconeVectorStore
from core.hybrid_search import HybridSearchEngine
from core.document_processor import DocumentProcessor
from retrieval.retriever import AdvancedRetriever
from caching.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Core RAG System
    
    This class contains all RAG logic and can be used independently
    of the FastAPI application.
    """
    
    def __init__(self, enable_cache: bool = True):
        """
        Initialize RAG System
        
        Args:
            enable_cache: Enable semantic caching (requires Redis)
        """
        self.initialized = False
        self.enable_cache = enable_cache
        
        # Components (will be initialized)
        self.embedding_manager = None
        self.vector_store = None
        self.hybrid_search = None
        self.document_processor = None
        self.retriever = None
        self.cache = None
        self.llm_client = None
    
    def initialize(self):
        """Initialize all RAG components"""
        if self.initialized:
            logger.info("RAG System already initialized")
            return
        
        logger.info("=" * 80)
        logger.info("Initializing RAG System...")
        logger.info("=" * 80)
        
        # 1. Embedding Manager
        logger.info("Loading embedding model...")
        self.embedding_manager = EmbeddingManager()
        logger.info(f"✅ Loaded: {settings.EMBEDDING_MODEL}")
        
        # 2. Vector Store (Pinecone)
        logger.info("Connecting to Pinecone...")
        self.vector_store = PineconeVectorStore(
            dimension=settings.EMBEDDING_DIMENSION,
            index_name="rag-knowledge-base"
        )
        logger.info("✅ Pinecone connected")
        
        # 3. Hybrid Search
        logger.info("Initializing hybrid search...")
        self.hybrid_search = HybridSearchEngine(self.vector_store)
        logger.info("✅ Hybrid search ready")
        
        # 4. Document Processor
        logger.info("Setting up document processor...")
        self.document_processor = DocumentProcessor(
            vector_store=self.vector_store,
            hybrid_search=self.hybrid_search,
            embedding_manager=self.embedding_manager
        )
        logger.info("✅ Document processor ready")
        
        # 5. Retriever
        logger.info("Initializing retriever...")
        self.retriever = AdvancedRetriever(
            embedding_manager=self.embedding_manager,
            hybrid_search=self.hybrid_search
        )
        logger.info("✅ Retriever ready")
        
        # 6. Semantic Cache (optional)
        if self.enable_cache and settings.ENABLE_SEMANTIC_CACHE:
            logger.info("Connecting to Redis for caching...")
            try:
                redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    db=0,
                    decode_responses=False
                )
                redis_client.ping()
                
                self.cache = SemanticCache(
                    redis_client=redis_client,
                    embedding_manager=self.embedding_manager
                )
                logger.info("✅ Semantic cache enabled")
            except Exception as e:
                logger.warning(f"⚠️  Redis unavailable: {e}")
                logger.warning("⚠️  Cache disabled - continuing without cache")
                self.cache = None
        else:
            logger.info("⚠️  Semantic cache disabled")
            self.cache = None
        
        # 7. LLM Client
        logger.info("Initializing LLM client...")
        api_key = settings.OPENROUTER_API_KEY
        base_url = settings.OPENROUTER_BASE_URL
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment. "
                "Please set it in .env file or export it."
            )
        
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("✅ LLM client ready")
        
        self.initialized = True
        logger.info("=" * 80)
        logger.info("🎉 RAG System fully initialized!")
        logger.info("=" * 80)
    
    def add_documents(
        self,
        documents: List[Dict[str, str]],
        show_progress: bool = True
    ) -> Dict:
        """
        Add documents to knowledge base
        
        Args:
            documents: List of dicts with 'text', 'source', and optional 'metadata'
            show_progress: Show processing progress
            
        Returns:
            Processing statistics
        """
        if not self.initialized:
            self.initialize()
        
        logger.info(f"Adding {len(documents)} documents to knowledge base")
        
        result = self.document_processor.process_batch(
            documents=documents,
            show_progress=show_progress
        )
        
        logger.info(
            f"Processing complete: {result['success']} success, "
            f"{result['skipped']} skipped, {result['errors']} errors"
        )
        
        return result
    
    def add_document(
        self,
        text: str,
        source: str = "manual_input",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Add a single document
        
        Args:
            text: Document text
            source: Document source identifier
            metadata: Optional metadata
            
        Returns:
            Processing result
        """
        if not self.initialized:
            self.initialize()
        
        return self.document_processor.process_text(
            text=text,
            metadata=metadata,
            source=source
        )
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True,
        return_sources: bool = True
    ) -> Dict:
        """
        Query the RAG system
        
        Args:
            query: User question
            top_k: Number of context chunks
            use_cache: Use semantic cache
            return_sources: Include source documents
            
        Returns:
            Answer with metadata
        """
        if not self.initialized:
            self.initialize()
        
        logger.info(f"Processing query: '{query}'")
        
        # Check cache
        if use_cache and self.cache:
            cached_result = self.cache.get(query)
            if cached_result:
                logger.info("✅ Cache HIT - returning cached result")
                return cached_result
        
        # Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(
            query=query,
            top_k=top_k
        )
        
        # Check if query was rejected
        if retrieval_result["status"] != "success":
            return {
                "status": retrieval_result["status"],
                "query": query,
                "answer": f"Cannot answer: {retrieval_result.get('reason', 'Unknown reason')}",
                "sources": []
            }
        
        # Generate answer
        answer_result = self._generate_answer(query, retrieval_result["results"])
        
        # Build response
        response = {
            "status": "success",
            "query": query,
            "answer": answer_result["answer"],
            "retrieval_stats": {
                "num_variants": retrieval_result.get("num_variants", 1),
                "total_retrieved": retrieval_result.get("total_retrieved", 0),
                "final_count": retrieval_result.get("final_count", 0)
            },
            "token_usage": {
                "prompt_tokens": answer_result["prompt_tokens"],
                "completion_tokens": answer_result["completion_tokens"],
                "total_tokens": answer_result["total_tokens"]
            }
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "chunk_id": r["id"],
                    "score": r.get("rerank_score", r.get("hybrid_score", 0)),
                    "metadata": r.get("metadata", {})
                }
                for r in retrieval_result["results"]
            ]
        
        # Cache the result
        if use_cache and self.cache:
            self.cache.set(query, response)
        
        return response
    
    def _generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """
        Generate answer using LLM with retrieved context
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Generated answer with token counts
        """
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            content = self.retriever._get_chunk_content(chunk["id"])
            score = chunk.get("rerank_score", chunk.get("hybrid_score", 0))
            context_parts.append(
                f"[Source {i+1}, Score: {score:.3f}]\n{content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Truncate if too long
        max_context_tokens = settings.MAX_CONTEXT_LENGTH
        # Rough estimate: 1 token ≈ 4 chars
        if len(context) > max_context_tokens * 4:
            context = context[:max_context_tokens * 4] + "\n... [context truncated]"
        
        # Build prompt
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Be concise but comprehensive
- Cite sources when possible (e.g., "According to Source 1...")

Answer:"""

        try:
            # Call OpenAI-compatible API (OpenRouter)
            response = self.llm_client.chat.completions.create(
                model=settings.LLM,
                max_tokens=1024,
                temperature=settings.TEMPERATURE,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        if not self.initialized:
            return {"error": "System not initialized"}
        
        stats = {
            "system": "Advanced RAG Knowledge Assistant",
            "initialized": self.initialized,
            "document_processor": self.document_processor.get_stats(),
            "embedding": self.embedding_manager.get_version_metadata()
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats
    
    def clear_cache(self) -> bool:
        """Clear semantic cache"""
        if self.cache:
            return self.cache.clear()
        return False


# Convenience function for quick initialization
def create_rag_system(enable_cache: bool = True) -> RAGSystem:
    """
    Create and initialize RAG system
    
    Args:
        enable_cache: Enable semantic caching
        
    Returns:
        Initialized RAGSystem instance
    """
    rag = RAGSystem(enable_cache=enable_cache)
    rag.initialize()
    return rag

