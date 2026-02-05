---
  RAG Pipeline Architecture

  This project is a production-grade Retrieval-Augmented Generation (RAG) Knowledge Assistant. The backend is built with
   FastAPI and has two major pipelines: a Document Ingestion Pipeline and a Query/Chat Pipeline.

  ---
  Pipeline 1: Document Ingestion (Upload Flow)

  Triggered when a user uploads a document via POST /api/v1/documents/upload.

  Step 1 — Document Processing (services/document_processor.py)

  - Purpose: Extract raw text from uploaded files.
  - Supports PDF (pypdf), DOCX (python-docx), TXT/MD, CSV/Excel (pandas), and HTML (BeautifulSoup).
  - Returns a ProcessedDocument with the extracted text, metadata, and page count.

  Step 2 — Text Preprocessing (core/chunking/preprocessor.py)

  - Purpose: Clean and normalize raw text before chunking.
  - Pipeline: Unicode normalization → control character removal → zero-width char removal → quote/dash normalization →
  whitespace normalization → optional URL/email removal → custom pattern removal.

  Step 3 — Chunking (core/chunking/strategies.py)

  - Purpose: Split text into retrieval-friendly segments (~512 tokens, 50 token overlap).
  - Uses LangChain text splitters. Strategies available: recursive (default, best for mixed content), fixed
  (token-based), sentence, semantic (SentenceTransformer-aware), page, and document (whole text).
  - Each chunk gets a unique ID, token count, character offsets, and metadata.

  Step 4 — Embedding Generation (core/embedding/generator.py)

  - Purpose: Convert text chunks into vector representations for semantic search.
  - Supports OpenAI, Cohere, and HuggingFace/local models.
  - Processes in batches (100-500 per call) with exponential backoff retries.
  - Uses the Embedding Cache (core/caching/embedding_cache.py) — keyed by content_hash + model_id — to skip re-embedding
   unchanged content (10-20% cost savings).
  - Critical rule: NEVER mix embedding models between indexing and querying.

  Step 5 — Vector Storage (services/vector_store/, core/retrieval/vector_search.py)

  - Purpose: Store embeddings in a vector database for similarity search.
  - Supports Qdrant (default, HNSW-native), Pinecone (managed), PGVector, and in-memory (dev).
  - HNSW index configured with M=16, ef_search=100 for 95%+ recall.
  - Chunks are also stored in PostgreSQL (db/models.py) for metadata tracking.

  Step 6 — BM25 Index Building (core/retrieval/bm25_search.py)

  - Purpose: Build a keyword-based search index for exact term matching.
  - Uses rank_bm25 (BM25Okapi or BM25Plus). Custom tokenizer preserves special characters like error codes and IDs
  (e.g., ERR-404, SKU-12345).
  - The BM25 index is built from all documents in the vector store at startup.

  ---
  Pipeline 2: Query/Chat (Chat Flow)

  Triggered when a user sends a message via POST /api/v1/chat. The full 10-step pipeline is defined in
  api/v1/endpoints/chat.py:

  Step 1 — Query Classification & Routing (core/query/classifier.py + core/query/router.py)

  - Purpose: Classify the query type and route to an appropriate model tier to optimize cost vs quality.
  - Classifier: Rule-based (regex patterns, heuristics) targeting <12ms. Categorizes queries as: SIMPLE, FAQ, COMPLEX,
  CONVERSATIONAL, OUT_OF_SCOPE, AMBIGUOUS. Also detects intent (factual, procedural, comparative, troubleshooting, etc.)
   and computes a complexity score (0-1).
  - Router: 3-tier model routing pattern:
    - Tier 1 (70% of queries): Simple model (GPT-3.5) for straightforward Q&A.
    - Tier 2 (25%): Medium model (GPT-4o-mini) for moderate complexity.
    - Tier 3 (5%): Best model (GPT-4) for complex reasoning.
    - When confidence is low, bumps to a higher tier (prefers quality over cost).
    - Has an AdaptiveRouter that auto-adjusts thresholds every N queries to maintain the 70/25/5 distribution.

  Step 2 — Semantic Cache Check (core/caching/semantic_cache.py)

  - Purpose: Return cached responses for similar/identical queries (50-70% cost savings).
  - 3-layer cache:
    - Layer 1 (Exact): SHA-256 hash lookup — instant match for identical queries.
    - Layer 2 (Semantic): Embeds the query, computes cosine similarity against cached queries (threshold default 0.92).
    - Layer 3 (Cross-encoder validation): Optional cross-encoder to filter false positive cache hits.
  - If cache hit → return immediately (skipping all retrieval/generation steps).
  - Backed by Redis (persistent) or in-memory (fallback).

  Step 3 — Query Transformation (core/query/transformer.py)

  - Purpose: Generate 3-5 query variations for multi-query retrieval (+15-25% recall improvement).
  - Rule-based transformer: Extracts keywords, generates question-form variations, applies synonym replacement. Fast, no
   LLM call.
  - LLM-based transformer (optional): Uses LangChain to generate semantically diverse variations. Also supports HyDE
  (Hypothetical Document Embeddings) — generates a hypothetical answer passage and uses its embedding for search.

  Step 4 — Conversation History Compression (core/memory/conversation.py)

  - Purpose: Maintain conversation context while reducing token usage (60-70% reduction).
  - Sliding window strategy: Last 3 messages kept in full; older messages compressed into a truncated summary.
  - Prevents sending full history with every request.

  Step 5 — Hybrid Retrieval (core/retrieval/hybrid_search.py)

  - Purpose: Retrieve relevant chunks using both semantic and keyword search (+40% quality vs vector-only).
  - For each query variation (from Step 3):
    - Vector Search (core/retrieval/vector_search.py): Embeds query, searches vector DB for semantically similar chunks
  (top-100).
    - BM25 Search (core/retrieval/bm25_search.py): Keyword search for exact term matches.
    - Both run in parallel using asyncio.gather.
  - Score Fusion: Combines scores using Score = (Vector × 5) + (BM25 × 3) + (Recency × 0.2). Also supports Reciprocal
  Rank Fusion (RRF). Scores are min-max normalized before fusion.
  - Results are deduplicated across query variations, sorted by combined score.

  Step 6 — Cross-Encoder Reranking (core/retrieval/reranker.py)

  - Purpose: Rerank top-100 retrieval candidates down to top-10 with higher precision (+5-10%).
  - Uses sentence-transformers CrossEncoder models (default: ms-marco-MiniLM-L-6-v2).
  - Cross-encoders are more accurate than bi-encoders because they process query+document together with cross-attention.
  - Latency budget: 15-20ms. Model presets: fast (TinyBERT), balanced (MiniLM), accurate (BGE-reranker).

  Step 7 — Relevance Filtering (core/retrieval/relevance.py)

  - Purpose: Drop low-quality chunks below a relevance threshold to reduce noise (~30% filtered out).
  - Default: drop chunks scoring below 0.6 (after min-max normalization).
  - Strategies: threshold (fixed cutoff), dynamic (mean - std), top_k, percentile.
  - Guarantees at least min_chunks=1 are always returned.

  Step 8 — Context Building & Prompt Construction

  - Context Builder (core/generation/context_builder.py): Dynamically sizes context based on query complexity:
    - Simple queries → 800 tokens of context (40% token savings).
    - Complex queries → 4000 tokens (full context).
  - Prompt Manager (core/generation/prompt_manager.py): Builds compressed prompts (2800→900 tokens target, 60-70%
  reduction). Uses minimal templates. Edge cases (no context, out-of-scope, ambiguous) get dedicated short prompts.
  History is included only when present.

  Step 9 — LLM Generation (core/generation/llm_client.py)

  - Purpose: Generate the final response using the selected model.
  - Supports OpenRouter, OpenAI, and local LLM providers.
  - Executes the tier-based model selection from Step 1.
  - Uses httpx for async HTTP calls with retry (exponential backoff via tenacity).
  - Supports both streaming (SSE) and non-streaming responses.
  - Automatic fallback to Tier 1 model if the primary model fails.

  Step 10 — Cache, Log & Return

  - Cache response: Stores the query-response pair in semantic cache for future hits.
  - Update conversation memory: Adds the user message and assistant response to the sliding window.
  - Log everything: Query log saved to PostgreSQL (query, type, latency, model used, cache hit, etc.). Conversation
  messages saved separately.
  - Metrics recording: Tracks routing distribution, retrieval latency, generation latency, cache hit rates.
  - Return response with sources (top-5 source chunks with content preview and scores).

  ---
  Supporting Infrastructure
  Module: main.py
  Purpose: FastAPI app with lifespan (startup/shutdown), CORS, request logging middleware, health checks
  ────────────────────────────────────────
  Module: core/config.py
  Purpose: Centralized pydantic-settings config loaded from .env — embedding, chunking, retrieval, cache, LLM, context,
    monitoring, database
  ────────────────────────────────────────
  Module: api/v1/dependencies.py
  Purpose: DI for DB sessions, API key auth, rate limiting (in-memory sliding window)
  ────────────────────────────────────────
  Module: api/v1/endpoints/feedback.py
  Purpose: Thumbs up/down + 1-5 star ratings, exports evaluation datasets for RAGAS
  ────────────────────────────────────────
  Module: monitoring/
  Purpose: Structured JSON logging, metrics collection, distributed tracing
  ────────────────────────────────────────
  Module: evaluation/
  Purpose: Generation evaluation, LLM-as-judge, retrieval evaluation with benchmark test sets
  ────────────────────────────────────────
  Module: db/
  Purpose: SQLAlchemy async models for conversations, messages, documents, chunks, query logs, feedback
