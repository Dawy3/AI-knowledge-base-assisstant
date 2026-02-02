# Case Study: AI Knowledge Assistant

## Overview

Built a production-ready RAG (Retrieval-Augmented Generation) system that enables users to query their document knowledge base through a conversational AI interface.

## Business Problem

Organizations struggle to extract insights from growing document repositories. Traditional search returns documents, not answers. Users need an intelligent assistant that understands context and provides direct, cited responses.

---

## Challenges & Solutions

### 1. Cost — LLM API Expenses

**Problem:** Every query hits the LLM API. Repeated/similar questions multiply costs rapidly at scale.

**Solutions:**
- **Semantic Caching** — Cache responses keyed by query embeddings. Similar questions (cosine similarity > 0.95) return cached answers instantly
- **LLM Router** — Route simple queries to GPT-3.5-turbo, complex ones to GPT-4. Reduces average cost per query by ~60%
- **Smaller Embedding Model** — `text-embedding-3-small` vs large variant. 5x cheaper, minimal accuracy loss

---

### 2. Latency — Slow Response Times

**Problem:** Users expect sub-second responses. Vector search + LLM generation creates noticeable delays.

**Solutions:**
- **Qdrant with HNSW Index** — Dedicated vector DB with HNSW (m=16, ef_construct=100). Approximate nearest neighbor gives 10x faster search with 95%+ recall
- **Streaming Responses** — Server-Sent Events stream tokens as generated. Perceived latency drops from 3s to <500ms first token
- **Connection Pooling** — SQLAlchemy async pool reuses DB connections. Eliminates connection overhead per request

---

### 3. Retrieval Quality — Irrelevant Context

**Problem:** Wrong chunks retrieved = hallucinated or off-topic answers. Garbage in, garbage out.

**Solutions:**
- **Recursive Chunking with Overlap** — 512 token chunks with 50 token overlap. Preserves context across boundaries
- **Query Expansion** — LLM rewrites user query into multiple semantic variants. Improves recall for ambiguous questions
- **Top-K with Relevance Threshold** — Retrieve top 5 chunks, filter by minimum similarity score (0.7). Prevents low-quality context injection

---

### 4. Document Diversity — Multiple Formats

**Problem:** Users upload PDF, DOCX, TXT, MD files. Each format requires different parsing logic.

**Solutions:**
- **Format-Specific Extractors** — Dedicated parsers per file type (PyPDF2, python-docx, markdown)
- **Unified Text Pipeline** — All extractors output plain text → same chunking/embedding flow
- **Metadata Preservation** — Store source filename, page numbers for citation tracking

---

### 5. Infrastructure Complexity — Deployment Overhead

**Problem:** Typical RAG stacks require separate vector DB, message queue, cache layer. Complex to deploy and maintain.

**Solutions:**
- **Qdrant for Vectors, PostgreSQL for Relational** — Clear separation of concerns. Qdrant handles vector similarity search (optimized HNSW). PostgreSQL stores conversations, documents, chunks content, feedback, query logs
- **Single Docker Compose** — Entire stack (API, Qdrant, PostgreSQL, frontend) in one `docker-compose.yml`. Deploy anywhere with `docker compose up`
- **Embedded Caching** — In-memory semantic cache, no Redis dependency for MVP

---

### 6. User Experience — Intrusive UI

**Problem:** Full-page chat interfaces disrupt user workflows. Need non-intrusive integration.

**Solutions:**
- **Sidebar Widget Pattern** — Floating action button + slide-out panel. Overlays existing app without navigation changes
- **Zustand State Management** — Lightweight store for open/close state, message history. No prop drilling
- **Real-time Streaming Display** — Tokens render as they arrive. Users see progress immediately

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 14, React, Tailwind CSS, Zustand |
| Backend | FastAPI, SQLAlchemy, LangChain |
| Vector DB | Qdrant (HNSW indexing) |
| Relational DB | PostgreSQL (conversations, documents, feedback) |
| LLM | OpenAI GPT-4 / GPT-3.5 (routed) |
| Embeddings | text-embedding-3-small |
| Infrastructure | Docker Compose |

## Results

| Metric | Value |
|--------|-------|
| Avg. Response Time | <2s (cached: <100ms) |
| Retrieval Accuracy | 85%+ relevance |
| Cost Reduction | ~60% via routing + caching |
| Deployment | Single docker-compose |

## Future Improvements

- Multi-tenant support with user isolation
- Hybrid search (semantic + keyword BM25)
- Document versioning and incremental updates
- Analytics dashboard for query patterns
