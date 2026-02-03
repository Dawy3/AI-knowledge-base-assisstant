"""
Query Endpoint (Single Q&A).

FOCUS: Fast synchronous responses
MUST: Check semantic cache FIRST (50-70% savings)
PRODUCTION: Full RAG pipeline with hybrid search, reranking
"""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.api.v1.dependencies import (
    AuthenticatedContext,
    CurrentUserId,
    DbSession,
    RateLimited,
)
from backend.core.config import settings
from backend.core.query.classifier import QueryClassifier
from backend.core.query.router import QueryRouter, ModelTier
from backend.core.generation.llm_client import LLMClient
from backend.core.generation.prompt_manager import PromptManager
from backend.core.generation.context_builder import ContextBuilder
from backend.core.caching.semantic_cache import SemanticCache
from backend.core.retrieval.hybrid_search import HybridSearch, create_hybrid_search
from backend.core.retrieval.vector_search import VectorSearch
from backend.core.retrieval.bm25_search import BM25Search
from backend.core.retrieval.reranker import Reranker, create_reranker
from backend.core.embedding.generator import create_embedding_generator
from backend.db.models import QueryLog as QueryLogModel
from backend.monitoring.logging import QueryLogger
from backend.monitoring.metrics import MetricsCollector
from backend.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
query_logger = QueryLogger()
metrics = MetricsCollector()

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Single query request."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer",
    )

    # Search options
    use_cache: bool = Field(
        default=True,
        description="Check semantic cache first (recommended)",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of context chunks to retrieve",
    )

    # Generation options
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Maximum tokens in response",
    )
    include_sources: bool = Field(
        default=True,
        description="Include source references in response",
    )

    # Advanced
    model_tier: Optional[str] = Field(
        None,
        description="Force specific model tier",
    )
    filter: Optional[dict] = Field(
        None,
        description="Metadata filter for retrieval",
    )


class SourceReference(BaseModel):
    """Reference to source document."""
    chunk_id: str
    document_id: str
    content_preview: str
    score: float
    metadata: Optional[dict] = None


class QueryResponse(BaseModel):
    """Query response with sources."""
    query_id: str
    answer: str

    # Cache info
    cached: bool = False
    cache_similarity: Optional[float] = None

    # Sources
    sources: Optional[list[SourceReference]] = None

    # Metadata
    model: str
    query_type: str
    latency_ms: float

    # Token usage
    tokens: Optional[dict] = None


class ClassificationResponse(BaseModel):
    """Query classification result."""
    query: str
    category: str
    confidence: float
    intent: Optional[str] = None
    complexity_score: float
    suggested_model: str
    use_cache: bool
    latency_ms: float


# =============================================================================
# Service Singletons (Production)
# =============================================================================

_embedding_generator = None
_semantic_cache = None
_hybrid_search = None
_reranker = None


def get_embedding_generator():
    """Get or create embedding generator singleton."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = create_embedding_generator()
    return _embedding_generator


def get_semantic_cache():
    """Get or create semantic cache singleton."""
    global _semantic_cache
    if _semantic_cache is None and settings.cache.semantic_cache_enabled:
        try:
            import redis
            redis_client = redis.Redis.from_url(settings.cache.redis_url)
            redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory cache: {e}")
            redis_client = None

        embedding_gen = get_embedding_generator()

        # Sync wrapper for async embed function (required by SemanticCache)
        # Uses thread pool to avoid "event loop already running" error
        def sync_embed(text):
            import asyncio
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, embedding_gen.embed_texts([text]))
                result = future.result()
                return result.embeddings[0]

        _semantic_cache = SemanticCache(
            embed_func=sync_embed,
            redis_client=redis_client,
            similarity_threshold=settings.cache.semantic_cache_threshold,
            ttl_seconds=settings.cache.semantic_cache_ttl,
        )
    return _semantic_cache


async def get_hybrid_search():
    """Get or create hybrid search singleton."""
    global _hybrid_search
    if _hybrid_search is None:
        vector_store = get_vector_store()

        # Ensure vector store is connected
        try:
            await vector_store.connect()
        except Exception as e:
            logger.debug(f"Vector store connect (may already be connected): {e}")

        vector_search = VectorSearch(vector_store)
        bm25_search = BM25Search()

        # Build BM25 index from vector store documents
        try:
            all_docs = await vector_store.get_all_documents()
            if all_docs:
                bm25_search.index(all_docs)
                logger.info(f"BM25 index built with {len(all_docs)} documents")
            else:
                logger.warning("No documents in vector store, BM25 index empty")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}", exc_info=True)

        _hybrid_search = create_hybrid_search(
            vector_search=vector_search,
            bm25_search=bm25_search,
            vector_weight=settings.retrieval.vector_weight,
            bm25_weight=settings.retrieval.bm25_weight,
            recency_weight=settings.retrieval.recency_weight,
        )
    return _hybrid_search


def get_reranker():
    """Get or create reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = create_reranker(model="balanced")
    return _reranker


def get_llm_client() -> LLMClient:
    """Get LLM client instance."""
    return LLMClient()


def get_query_router() -> QueryRouter:
    """Get query router instance."""
    return QueryRouter()


def get_classifier() -> QueryClassifier:
    """Get query classifier instance."""
    return QueryClassifier()


def get_prompt_manager() -> PromptManager:
    """Get prompt manager instance."""
    return PromptManager()


def get_context_builder() -> ContextBuilder:
    """Get context builder instance."""
    return ContextBuilder()


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "",
    response_model=QueryResponse,
    responses={
        200: {"description": "Successful response"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)
async def query(
    request: QueryRequest,
    db: DbSession,
    ctx: AuthenticatedContext,
    llm_client: LLMClient = Depends(get_llm_client),
    query_router: QueryRouter = Depends(get_query_router),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
    context_builder: ContextBuilder = Depends(get_context_builder),
):
    """
    Single Q&A query endpoint.

    FOCUS: Fast synchronous responses
    MUST: Check semantic cache FIRST (50-70% savings)

    PRODUCTION Flow:
    1. Check semantic cache → return if hit
    2. Classify query → determine complexity
    3. Retrieve context (hybrid search: vector + BM25)
    4. Rerank top-100 → top-10
    5. Generate answer
    6. Cache response
    7. Return with sources
    """
    start_time = time.perf_counter()
    query_id = str(uuid.uuid4())

    # Start logging
    log = query_logger.start_query(
        query=request.question,
        user_id=ctx.get("user_id"),
        query_id=query_id,
    )

    try:
        # =================================================================
        # Step 1: Check Semantic Cache FIRST
        # MUST: 50-70% cost savings from cache hits
        # =================================================================
        if request.use_cache:
            semantic_cache = get_semantic_cache()
            if semantic_cache:
                cache_result = await semantic_cache.get(request.question)
                if cache_result.hit:
                    metrics.record_cache("semantic", hit=True)
                    query_logger.log_cache_hit(log, cache_result.layer)

                    total_latency = (time.perf_counter() - start_time) * 1000
                    log.total_latency_ms = total_latency
                    query_logger.end_query(log)

                    # Save to database
                    await _save_query_log(db, log, query_id, cached=True)

                    return QueryResponse(
                        query_id=query_id,
                        answer=cache_result.response,
                        cached=True,
                        cache_similarity=cache_result.similarity,
                        model="cached",
                        query_type="cached",
                        latency_ms=total_latency,
                    )

        metrics.record_cache("semantic", hit=False)

        # =================================================================
        # Step 2: Classify and Route Query
        # =================================================================
        if request.model_tier:
            tier = ModelTier(request.model_tier)
            query_type = "user_specified"
        else:
            routing_decision = query_router.route(request.question)
            tier = routing_decision.tier
            query_type = routing_decision.category.value

        log.query_type = query_type
        metrics.record_routing(tier.value)

        # Determine context size based on query type
        if query_type in ("simple", "faq"):
            context_type = "simple"
        else:
            context_type = "complex"

        # =================================================================
        # Step 3: Retrieve Context (Hybrid Search)
        # CRITICAL: Vector-only search WILL FAIL in production
        # =================================================================
        retrieval_start = time.perf_counter()

        # Generate query embedding
        embedding_gen = get_embedding_generator()
        query_embedding = await embedding_gen.embed_query(request.question)

        # Hybrid search
        hybrid_search = await get_hybrid_search()
        search_response = await hybrid_search.search(
            query=request.question,
            query_embedding=query_embedding,
            top_k=settings.retrieval.top_k_retrieval,  # Get top-100
            filter=request.filter,
        )

        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000

        # =================================================================
        # Step 4: Rerank top-100 → top-10
        # EXPECTED: +5-10% precision improvement
        # =================================================================
        rerank_start = time.perf_counter()
        reranker = get_reranker()

        candidates = [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": r.combined_score,
                "metadata": r.metadata,
                "document_id": r.document_id,
            }
            for r in search_response.results
        ]

        rerank_result = await reranker.rerank(
            query=request.question,
            candidates=candidates,
            top_k=request.top_k,
        )

        rerank_latency = (time.perf_counter() - rerank_start) * 1000

        query_logger.log_retrieval(
            log=log,
            chunks=len(rerank_result.results),
            latency_ms=retrieval_latency + rerank_latency,
            search_type="hybrid+rerank",
        )

        metrics.record_retrieval(
            latency=(retrieval_latency + rerank_latency) / 1000,
            results=len(rerank_result.results),
            search_type="hybrid",
        )

        # =================================================================
        # Step 5: Build Context and Generate
        # =================================================================
        chunk_dicts = [
            {"content": r.content, "score": r.rerank_score, "metadata": r.metadata}
            for r in rerank_result.results
        ]

        contexts = context_builder.build(
            chunks=chunk_dicts,
            query_type=context_type,
        )

        # Build prompt
        system_prompt, user_prompt = prompt_manager.build_rag_prompt(
            query=request.question,
            contexts=contexts,
            query_type=query_type if query_type == "out_of_scope" else "normal",
        )

        # Generate response
        gen_start = time.perf_counter()
        response = await llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            tier=tier,
            max_tokens=request.max_tokens,
        )
        gen_latency = (time.perf_counter() - gen_start) * 1000

        # Log generation
        query_logger.log_generation(
            log=log,
            response=response.content,
            model=response.model,
            latency_ms=gen_latency,
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
        )

        metrics.record_generation(
            latency=gen_latency / 1000,
            model=response.model,
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
        )

        # =================================================================
        # Step 6: Cache Response
        # =================================================================
        if request.use_cache:
            semantic_cache = get_semantic_cache()
            if semantic_cache:
                await semantic_cache.set(
                    query=request.question,
                    response=response.content,
                    metadata={"model": response.model, "query_type": query_type},
                )

        # =================================================================
        # Step 7: Build Response with Sources
        # =================================================================
        sources = None
        if request.include_sources and rerank_result.results:
            sources = [
                SourceReference(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content_preview=r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    score=r.rerank_score,
                    metadata=r.metadata,
                )
                for r in rerank_result.results[:5]  # Top 5 sources
            ]

        total_latency = (time.perf_counter() - start_time) * 1000
        log.total_latency_ms = total_latency

        # Complete logging
        query_logger.end_query(log)

        # Save to database
        await _save_query_log(db, log, query_id, cached=False)

        return QueryResponse(
            query_id=query_id,
            answer=response.content,
            cached=False,
            sources=sources,
            model=response.model,
            query_type=query_type,
            latency_ms=total_latency,
            tokens=response.usage,
        )

    except Exception as e:
        query_logger.log_error(log, str(e))
        logger.exception(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}",
        )


async def _save_query_log(db, log, query_id: str, cached: bool):
    """Save query log to database."""
    try:
        db_log = QueryLogModel(
            id=uuid.UUID(query_id),
            query_id=query_id,
            query=log.query,
            query_type=log.query_type if hasattr(log, 'query_type') else None,
            response=log.response if hasattr(log, 'response') else None,
            model=log.model if hasattr(log, 'model') else None,
            total_latency_ms=log.total_latency_ms if hasattr(log, 'total_latency_ms') else None,
            cache_hit=cached,
            user_id=log.user_id if hasattr(log, 'user_id') else None,
        )
        db.add(db_log)
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to save query log: {e}")


class ClassifyRequest(BaseModel):
    """Classification request."""
    question: str = Field(..., min_length=1, max_length=2000)


@router.post("/classify", response_model=ClassificationResponse)
async def classify_query(
    request: ClassifyRequest,
    classifier: QueryClassifier = Depends(get_classifier),
    query_router: QueryRouter = Depends(get_query_router),
    _rate_limit: RateLimited = None,
):
    """
    Classify a query without generating a response.

    Useful for:
    - Pre-flight checks
    - Routing decisions
    - Analytics
    """
    start_time = time.perf_counter()

    # Classify
    result = classifier.classify(request.question)

    # Get routing suggestion
    routing = query_router.route(request.question, classification=result)

    latency_ms = (time.perf_counter() - start_time) * 1000

    return ClassificationResponse(
        query=request.question,
        category=result.category.value,
        confidence=result.confidence,
        intent=result.intent.value if result.intent else None,
        complexity_score=result.complexity_score,
        suggested_model=routing.tier.value,
        use_cache=result.use_cache,
        latency_ms=latency_ms,
    )


@router.get("/health")
async def query_health():
    """Health check for query endpoint dependencies."""
    health_status = {
        "status": "healthy",
        "cache_enabled": settings.cache.semantic_cache_enabled,
        "vector_store": settings.database.vector_store,
    }

    # Check semantic cache
    try:
        cache = get_semantic_cache()
        if cache:
            health_status["cache_status"] = "connected"
            health_status["cache_stats"] = cache.get_stats()
    except Exception as e:
        health_status["cache_status"] = f"error: {e}"

    return health_status


@router.post("/cache/clear")
async def clear_cache():
    """Clear the semantic cache."""
    try:
        cache = get_semantic_cache()
        if cache:
            cache.clear()
            return {"status": "success", "message": "Cache cleared"}
        return {"status": "warning", "message": "Cache not initialized"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return {"status": "error", "message": str(e)}
