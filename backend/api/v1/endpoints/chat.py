"""
Chat Endpoint (Streaming + Non-Streaming).

FOCUS: Stream responses, log every query+response
MUST: Implement request validation, rate limiting
PRODUCTION: Full RAG pipeline with semantic cache, hybrid search, reranking
"""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.v1.dependencies import (
    AuthenticatedContext,
    CurrentUserId,
    DbSession,
    RateLimited,
)
from backend.core.config import settings
from backend.core.query.transformer import QueryTransformer, create_query_transformer
from backend.core.query.router import QueryRouter, ModelTier
from backend.core.generation.llm_client import LLMClient
from backend.core.generation.prompt_manager import PromptManager
from backend.core.generation.context_builder import ContextBuilder
from backend.core.caching.semantic_cache import SemanticCache
from backend.core.retrieval.hybrid_search import create_hybrid_search
from backend.core.retrieval.vector_search import VectorSearch
from backend.core.retrieval.bm25_search import BM25Search
from backend.core.retrieval.reranker import create_reranker
from backend.core.retrieval.relevance import RelevanceFilter, create_relevance_filter
from backend.core.embedding.generator import create_embedding_generator
from backend.core.memory.conversation import ConversationMemory
from backend.db.models import Conversation, Message, QueryLog as QueryLogModel
from backend.monitoring.logging import QueryLogger, QueryLog
from backend.monitoring.metrics import MetricsCollector
from backend.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
query_logger = QueryLogger()
metrics = MetricsCollector()

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    """Chat request with conversation history."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User message",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Conversation ID for context continuity",
    )
    history: Optional[list[ChatMessage]] = Field(
        default=[],
        max_length=20,
        description="Conversation history (last N messages)",
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming response",
    )

    # Search options
    use_cache: bool = Field(
        default=True,
        description="Check semantic cache first (50-70% savings)",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of context chunks to retrieve",
    )

    # Advanced options
    model_tier: Optional[str] = Field(
        None,
        description="Force specific model tier (tier_1, tier_2, tier_3)",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum tokens in response",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
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


class ChatResponse(BaseModel):
    """Chat response."""
    query_id: str
    message: str
    model: str
    conversation_id: str

    # Cache info
    cached: bool = False
    cache_similarity: Optional[float] = None

    # Metadata
    latency_ms: float
    tokens_used: Optional[dict] = None

    # Source tracking (for RAG)
    sources: Optional[list[SourceReference]] = None


# =============================================================================
# Service Singletons (Production)
# =============================================================================

_embedding_generator = None
_semantic_cache = None
_hybrid_search = None
_reranker = None
_query_transformer = None
_relevance_filter = None
_conversation_memories: dict[str, ConversationMemory] = {}


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
            redis_client.ping()  # Test connection
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


def get_query_transformer() -> QueryTransformer:
    """Get or create query transformer singleton."""
    global _query_transformer
    if _query_transformer is None:
        _query_transformer = create_query_transformer(num_variations=4)
    return _query_transformer


def get_relevance_filter() -> RelevanceFilter:
    """Get or create relevance filter singleton."""
    global _relevance_filter
    if _relevance_filter is None:
        _relevance_filter = create_relevance_filter(
            min_score=0.6,  # Drop chunks below 0.6 relevance
            max_chunks=10,
            strategy="threshold",
        )
    return _relevance_filter


def get_conversation_memory(conversation_id: str) -> ConversationMemory:
    """Get or create conversation memory for a conversation."""
    global _conversation_memories
    if conversation_id not in _conversation_memories:
        _conversation_memories[conversation_id] = ConversationMemory(
            conversation_id=conversation_id
        )
    return _conversation_memories[conversation_id]


def get_llm_client() -> LLMClient:
    """Get LLM client instance."""
    return LLMClient()


def get_query_router() -> QueryRouter:
    """Get query router instance."""
    return QueryRouter()


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
    response_model=ChatResponse,
    responses={
        200: {"description": "Successful response"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)
async def chat(
    request: ChatRequest,
    db: DbSession,
    ctx: AuthenticatedContext,
    llm_client: LLMClient = Depends(get_llm_client),
    query_router: QueryRouter = Depends(get_query_router),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
    context_builder: ContextBuilder = Depends(get_context_builder),
):
    """
    Chat endpoint with RAG-augmented responses.

    FOCUS: Stream responses, log every query+response
    FULL PRODUCTION PIPELINE:
    1. Check semantic cache → return if hit (50-70% cost savings)
    2. Route query → select model tier (router internally classifies)
    3. Transform query → generate 3-5 variations (+15-25% recall)
    4. Compress conversation history → sliding window (60-70% token reduction)
    5. Retrieve context (hybrid search: vector + BM25)
    6. Rerank top-100 → top-10 (+5-10% precision)
    7. Filter by relevance → drop below 0.6 (eliminate 30% noise)
    8. Generate response with context
    9. Cache response for future queries
    10. Log everything for evaluation
    """
    start_time = time.perf_counter()
    query_id = str(uuid.uuid4())
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Start query logging
    log = query_logger.start_query(
        query=request.message,
        user_id=ctx.get("user_id"),
        query_id=query_id,
    )

    try:
        # =================================================================
        # Step 1: Check Semantic Cache FIRST (for ALL requests)
        # MUST: 50-70% cost savings from cache hits
        # =================================================================
        if request.use_cache:
            semantic_cache = get_semantic_cache()
            if semantic_cache:
                cache_result = await semantic_cache.get(request.message)
                if cache_result.hit:
                    metrics.record_cache("semantic", hit=True)
                    query_logger.log_cache_hit(log, cache_result.layer)

                    total_latency = (time.perf_counter() - start_time) * 1000
                    log.total_latency_ms = total_latency
                    log.cache_hit = True
                    query_logger.end_query(log)

                    # Store in DB
                    await _save_query_log(db, log, query_id, conversation_id)

                    # Handle streaming cache hit
                    if request.stream:
                        async def stream_cached():
                            yield f"data: {cache_result.response}\n\n"
                            yield "data: [DONE]\n\n"

                        return StreamingResponse(
                            stream_cached(),
                            media_type="text/event-stream",
                            headers={
                                "X-Query-ID": query_id,
                                "X-Conversation-ID": conversation_id,
                                "X-Cache-Hit": "true",
                                "X-Cache-Similarity": str(cache_result.similarity),
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                            },
                        )

                    return ChatResponse(
                        query_id=query_id,
                        message=cache_result.response,
                        model="cached",
                        conversation_id=conversation_id,
                        cached=True,
                        cache_similarity=cache_result.similarity,
                        latency_ms=total_latency,
                    )

        metrics.record_cache("semantic", hit=False)

        # Handle streaming separately (after cache check)
        if request.stream:
            return await _stream_response(
                request=request,
                db=db,
                query_id=query_id,
                conversation_id=conversation_id,
                log=log,
                llm_client=llm_client,
                query_router=query_router,
                prompt_manager=prompt_manager,
                context_builder=context_builder,
            )

        # =================================================================
        # Step 2: Classify and Route Query
        # =================================================================
        if request.model_tier and request.model_tier in [t.value for t in ModelTier]:
            tier = ModelTier(request.model_tier)
            query_type = "user_specified"
        else:
            routing_decision = query_router.route(request.message)
            tier = routing_decision.tier
            query_type = routing_decision.category.value

        log.query_type = query_type
        metrics.record_routing(tier.value)

        # =================================================================
        # Step 3: Transform Query (Multi-query generation)
        # EXPECTED: +15-25% recall improvement
        # =================================================================
        transformer = get_query_transformer()
        transformed = await transformer.transform(request.message)
        all_queries = transformed.all_queries  # Original + variations
        logger.info(f"Query transformed: {len(all_queries)} variations generated")

        # =================================================================
        # Step 4: Compress Conversation History
        # EXPECTED: 60-70% token reduction for long conversations
        # =================================================================
        conv_memory = get_conversation_memory(conversation_id)

        # Add history from request to memory (if not already there)
        if request.history:
            for msg in request.history:
                conv_memory.add(msg.role, msg.content)

        # Get compressed history
        compressed_history = conv_memory.get_history()
        logger.debug(f"History compressed: {len(request.history or [])} messages → {len(compressed_history)} entries")

        # =================================================================
        # Step 5: Retrieve Context (Hybrid Search with Multi-query)
        # CRITICAL: Vector-only search WILL FAIL in production
        # =================================================================
        retrieval_start = time.perf_counter()

        # Generate query embedding for primary query
        embedding_gen = get_embedding_generator()
        query_embedding = await embedding_gen.embed_query(request.message)

        # Hybrid search with all query variations
        hybrid_search = await get_hybrid_search()

        # Collect results from all query variations
        all_results = []
        seen_chunk_ids = set()

        for query_variation in all_queries:
            # Generate embedding for variation
            variation_embedding = await embedding_gen.embed_query(query_variation)

            search_response = await hybrid_search.search(
                query=query_variation,
                query_embedding=variation_embedding,
                top_k=settings.retrieval.top_k_retrieval,  # Get top-100 per query
                filter=request.filter,
            )

            # Deduplicate results
            for result in search_response.results:
                if result.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(result.chunk_id)
                    all_results.append(result)

        # Sort by combined score and take top candidates for reranking
        all_results.sort(key=lambda r: r.combined_score, reverse=True)
        candidates_for_rerank = all_results[:100]  # Top 100 for reranking

        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000

        # =================================================================
        # Step 6: Rerank with Cross-Encoder
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
            for r in candidates_for_rerank
        ]

        rerank_result = await reranker.rerank(
            query=request.message,
            candidates=candidates,
            top_k=request.top_k * 2,  # Get more for filtering
        )

        rerank_latency = (time.perf_counter() - rerank_start) * 1000

        # =================================================================
        # Step 7: Filter by Relevance
        # EXPECTED: Reduce chunks, eliminate 30% noise
        # =================================================================
        relevance_filter = get_relevance_filter()

        filter_chunks = [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": r.rerank_score,
                "metadata": r.metadata,
                "document_id": r.document_id,
            }
            for r in rerank_result.results
        ]

        filtered_result = relevance_filter.filter(filter_chunks)

        logger.info(
            f"Relevance filtering: {filtered_result.input_count} → {filtered_result.output_count} chunks "
            f"({filtered_result.filter_rate:.1%} filtered, threshold={filtered_result.threshold_used:.2f})"
        )

        # Convert filtered results to expected format
        class FilteredChunkResult:
            def __init__(self, chunk):
                self.chunk_id = chunk.chunk_id
                self.content = chunk.content
                self.rerank_score = chunk.relevance_score
                self.metadata = chunk.metadata
                self.document_id = chunk.document_id

        final_results = [FilteredChunkResult(c) for c in filtered_result.chunks[:request.top_k]]

        query_logger.log_retrieval(
            log=log,
            chunks=len(final_results),
            latency_ms=retrieval_latency + rerank_latency,
            search_type="hybrid+rerank+filter",
        )

        metrics.record_retrieval(
            latency=(retrieval_latency + rerank_latency) / 1000,
            results=len(final_results),
            search_type="hybrid+rerank+filter",
        )

        # =================================================================
        # Step 8: Build Context and Generate
        # =================================================================
        # Determine context size based on query complexity
        if query_type in ("simple", "faq"):
            context_type = "simple"
        else:
            context_type = "complex"

        # Build context from filtered results
        chunk_dicts = [
            {
                "content": r.content,
                "score": r.rerank_score,
                "metadata": r.metadata,
            }
            for r in final_results
        ]

        contexts = context_builder.build(
            chunks=chunk_dicts,
            query_type=context_type,
        )

        # Build prompt with compressed history (from ConversationMemory)
        system_prompt, user_prompt = prompt_manager.build_rag_prompt(
            query=request.message,
            contexts=contexts,
            history=compressed_history if compressed_history else None,
            query_type=query_type if query_type == "out_of_scope" else "normal",
        )

        # Generate response
        gen_start = time.perf_counter()
        response = await llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            tier=tier,
            temperature=request.temperature,
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
        # Step 9: Cache Response
        # =================================================================
        if request.use_cache:
            semantic_cache = get_semantic_cache()
            if semantic_cache:
                await semantic_cache.set(
                    query=request.message,
                    response=response.content,
                    metadata={
                        "model": response.model,
                        "query_type": query_type,
                    }
                )

        # =================================================================
        # Step 10: Build Response with Sources
        # =================================================================
        # Update conversation memory with new messages
        conv_memory.add("user", request.message)
        conv_memory.add("assistant", response.content)

        sources = [
            SourceReference(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content_preview=r.content[:200] + "..." if len(r.content) > 200 else r.content,
                score=r.rerank_score,
                metadata=r.metadata,
            )
            for r in final_results[:5]  # Top 5 sources
        ]

        total_latency = (time.perf_counter() - start_time) * 1000
        log.total_latency_ms = total_latency

        # Complete logging
        query_logger.end_query(log)

        # Store in database
        await _save_query_log(db, log, query_id, conversation_id)
        await _save_conversation(db, conversation_id, request.message, response.content, response.model, ctx.get("user_id"))

        return ChatResponse(
            query_id=query_id,
            message=response.content,
            model=response.model,
            conversation_id=conversation_id,
            cached=False,
            latency_ms=total_latency,
            tokens_used=response.usage,
            sources=sources,
        )

    except Exception as e:
        query_logger.log_error(log, str(e))
        logger.exception(f"Chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}",
        )


async def _stream_response(
    request: ChatRequest,
    db: AsyncSession,
    query_id: str,
    conversation_id: str,
    log: QueryLog,
    llm_client: LLMClient,
    query_router: QueryRouter,
    prompt_manager: PromptManager,
    context_builder: ContextBuilder,
) -> StreamingResponse:
    """
    Handle streaming chat response with FULL RAG pipeline.

    Same pipeline as non-streaming:
    1. Route query
    2. Transform query (multi-query)
    3. Compress history (ConversationMemory)
    4. Hybrid search with all variations
    5. Rerank with cross-encoder
    6. Filter by relevance
    7. Generate with streaming
    """
    # Route query
    if request.model_tier and request.model_tier in [t.value for t in ModelTier]:
        tier = ModelTier(request.model_tier)
    else:
        routing_decision = query_router.route(request.message)
        tier = routing_decision.tier

    # Transform query (multi-query generation)
    transformer = get_query_transformer()
    transformed = await transformer.transform(request.message)
    all_queries = transformed.all_queries

    # Compress conversation history
    conv_memory = get_conversation_memory(conversation_id)
    if request.history:
        for msg in request.history:
            conv_memory.add(msg.role, msg.content)
    compressed_history = conv_memory.get_history()

    # Hybrid search with all query variations
    embedding_gen = get_embedding_generator()
    hybrid_search = await get_hybrid_search()

    all_results = []
    seen_chunk_ids = set()

    for query_variation in all_queries:
        variation_embedding = await embedding_gen.embed_query(query_variation)
        search_response = await hybrid_search.search(
            query=query_variation,
            query_embedding=variation_embedding,
            top_k=settings.retrieval.top_k_retrieval,
            filter=request.filter,
        )
        for result in search_response.results:
            if result.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(result.chunk_id)
                all_results.append(result)

    all_results.sort(key=lambda r: r.combined_score, reverse=True)
    candidates_for_rerank = all_results[:100]

    # Rerank with cross-encoder
    reranker = get_reranker()
    candidates = [
        {
            "chunk_id": r.chunk_id,
            "content": r.content,
            "score": r.combined_score,
            "metadata": r.metadata,
            "document_id": r.document_id,
        }
        for r in candidates_for_rerank
    ]

    rerank_result = await reranker.rerank(
        query=request.message,
        candidates=candidates,
        top_k=request.top_k * 2,
    )

    # Filter by relevance
    relevance_filter = get_relevance_filter()
    filter_chunks = [
        {
            "chunk_id": r.chunk_id,
            "content": r.content,
            "score": r.rerank_score,
            "metadata": r.metadata,
            "document_id": r.document_id,
        }
        for r in rerank_result.results
    ]
    filtered_result = relevance_filter.filter(filter_chunks)
    final_results = filtered_result.chunks[:request.top_k]

    # Build context from filtered results
    chunk_dicts = [
        {"content": c.content, "score": c.relevance_score}
        for c in final_results
    ]
    contexts = context_builder.build(chunks=chunk_dicts, query_type="complex")

    # Build prompt with compressed history
    system_prompt, user_prompt = prompt_manager.build_rag_prompt(
        query=request.message,
        contexts=contexts,
        history=compressed_history if compressed_history else None,
    )

    async def generate_stream():
        """Generator for streaming response."""
        full_response = ""
        start_time = time.perf_counter()

        try:
            async for chunk in llm_client.stream(
                prompt=user_prompt,
                system_prompt=system_prompt,
                tier=tier,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                full_response += chunk
                # SSE format
                yield f"data: {chunk}\n\n"

            # Send done marker
            yield "data: [DONE]\n\n"

            # Log after completion
            gen_latency = (time.perf_counter() - start_time) * 1000
            query_logger.log_generation(
                log=log,
                response=full_response,
                model=llm_client.get_model_for_tier(tier),
                latency_ms=gen_latency,
            )
            query_logger.end_query(log)

            # Update conversation memory
            conv_memory.add("user", request.message)
            conv_memory.add("assistant", full_response)

            # Cache the response
            if request.use_cache:
                logger.info(f"Streaming complete. full_response length={len(full_response)}, preview='{full_response[:100] if full_response else '(empty)'}...'")
                semantic_cache = get_semantic_cache()
                if semantic_cache:
                    await semantic_cache.set(
                        query=request.message,
                        response=full_response,
                    )
                else:
                    logger.warning("Semantic cache not available for storing response")

        except Exception as e:
            query_logger.log_error(log, str(e))
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "X-Query-ID": query_id,
            "X-Conversation-ID": conversation_id,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


async def _save_query_log(
    db: AsyncSession,
    log: QueryLog,
    query_id: str,
    conversation_id: str,
):
    """Save query log to database."""
    try:
        db_log = QueryLogModel(
            id=uuid.UUID(query_id),
            query_id=query_id,
            query=log.query,
            query_type=log.query_type,
            chunks_retrieved=[],  # JSONB column expects list, logging tracks count separately
            retrieval_latency_ms=log.retrieval_latency_ms if hasattr(log, 'retrieval_latency_ms') else None,
            search_type=log.search_type if hasattr(log, 'search_type') else None,
            response=log.response if hasattr(log, 'response') else None,
            model=log.model if hasattr(log, 'model') else None,
            generation_latency_ms=log.generation_latency_ms if hasattr(log, 'generation_latency_ms') else None,
            total_latency_ms=log.total_latency_ms,
            cache_hit=log.cache_hit if hasattr(log, 'cache_hit') else False,
            user_id=log.user_id,
            conversation_id=uuid.UUID(conversation_id) if conversation_id else None,
        )
        db.add(db_log)
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to save query log: {e}")


async def _save_conversation(
    db: AsyncSession,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    model: str,
    user_id: Optional[str],
):
    """Save conversation messages to database."""
    try:
        # Get or create conversation
        conv_uuid = uuid.UUID(conversation_id)
        result = await db.execute(
            select(Conversation).where(Conversation.id == conv_uuid)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            conversation = Conversation(
                id=conv_uuid,
                user_id=user_id,
            )
            db.add(conversation)

        # Add user message
        user_msg = Message(
            conversation_id=conv_uuid,
            role="user",
            content=user_message,
        )
        db.add(user_msg)

        # Add assistant message
        assistant_msg = Message(
            conversation_id=conv_uuid,
            role="assistant",
            content=assistant_message,
            model=model,
        )
        db.add(assistant_msg)

        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to save conversation: {e}")


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: DbSession,
    ctx: AuthenticatedContext,
    llm_client: LLMClient = Depends(get_llm_client),
    query_router: QueryRouter = Depends(get_query_router),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
    context_builder: ContextBuilder = Depends(get_context_builder),
):
    """
    Dedicated streaming endpoint.

    Alias for chat with stream=True.
    """
    request.stream = True
    return await chat(
        request=request,
        db=db,
        ctx=ctx,
        llm_client=llm_client,
        query_router=query_router,
        prompt_manager=prompt_manager,
        context_builder=context_builder,
    )
