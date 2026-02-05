
# 

Uploading Video Project 1.mp4…

AI Knowledge Assistant

A production-grade RAG (Retrieval-Augmented Generation) system that lets you upload documents and ask questions answered directly from your knowledge base. It combines hybrid search (vector + BM25), cross-encoder reranking, semantic caching, and 3-tier model routing to deliver accurate, cost-optimized responses with source citations.

![Next.js](https://img.shields.io/badge/Next.js-14-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Qdrant](https://img.shields.io/badge/Qdrant-HNSW-red)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Next.js   │────>│   FastAPI   │────>│   Qdrant    │
│   Frontend  │     │   Backend   │     │  (vectors)  │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────┼──────┐
                    │             │
               ┌────▼────┐  ┌────▼────┐  ┌───────────┐
               │ LLM API │  │PostgreSQL│  │   Redis   │
               │(OpenAI/ │  │(metadata)│  │  (cache)  │
               │OpenRouter)  └─────────┘  └───────────┘
               └─────────┘
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- An LLM API key (OpenAI or OpenRouter)
- Qdrant instance (local via Docker or [Qdrant Cloud](https://cloud.qdrant.io))

## Getting Started with Docker

### 1. Clone the repository

```bash
git clone <repo-url>
cd ai-knowledge-assistant
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
# LLM Provider (choose one)
LLM_PROVIDER=openai             # or: openrouter, local
LLM_MODEL=openai/gpt-4o-mini        # model name
OPENROUTER_API_KEY=sk-or-xxx         # if using OpenRouter
OPENAI_API_KEY=sk-xxx                # if using OpenAI

# Embedding
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Qdrant
QDRANT_URL=http://localhost:6333     # or your Qdrant Cloud URL
QDRANT_API_KEY=                      # required for Qdrant Cloud
QDRANT_COLLECTION=documents

# PostgreSQL (uses local container by default)
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/rag_db

# Redis (optional, falls back to in-memory cache)
REDIS_URL=redis://localhost:6379/0
```

### 3. Start all services

```bash
docker-compose up -d
```

This starts:

| Service      | URL                          | Description              |
|-------------|-------------------------------|--------------------------|
| Frontend    | http://localhost:3000          | Next.js web UI           |
| Backend API | http://localhost:8000          | FastAPI server           |
| API Docs    | http://localhost:8000/docs     | Swagger UI (dev only)    |
| PostgreSQL  | localhost:5432                 | Relational database      |

### 4. Verify services are running

```bash
# Check all containers
docker-compose ps

# Check backend health
curl http://localhost:8000/health

# Check full readiness (DB + vector store + cache)
curl http://localhost:8000/health/ready
```

### 5. Start using

1. Open http://localhost:3000
2. Upload documents (PDF, DOCX, TXT, CSV, HTML)
3. Ask questions and get answers grounded in your documents

## Useful Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Rebuild after code changes
docker-compose up -d --build

# Stop services
docker-compose down

# Stop and remove all data (volumes)
docker-compose down -v
```

## API Endpoints

| Method | Endpoint                         | Description                |
|--------|----------------------------------|----------------------------|
| POST   | `/api/v1/chat`                   | Send message, get response |
| POST   | `/api/v1/chat/stream`            | Streaming chat response    |
| POST   | `/api/v1/documents/upload`       | Upload a document          |
| POST   | `/api/v1/documents/batch`        | Upload multiple documents  |
| GET    | `/api/v1/documents`              | List all documents         |
| GET    | `/api/v1/documents/{id}/status`  | Check processing status    |
| DELETE | `/api/v1/documents/{id}`         | Delete a document          |
| POST   | `/api/v1/feedback/thumbs`        | Submit thumbs up/down      |
| POST   | `/api/v1/feedback/rating`        | Submit 1-5 rating          |
| GET    | `/api/v1/feedback/stats`         | View feedback statistics   |

## Project Structure

```
├── BACKEND/
│   ├── main.py                    # FastAPI app entry point
│   ├── api/v1/endpoints/          # Chat, documents, feedback endpoints
│   ├── core/
│   │   ├── query/                 # Query classification & routing
│   │   ├── caching/               # Semantic + embedding cache
│   │   ├── chunking/              # Text preprocessing & chunking
│   │   ├── embedding/             # Embedding generation (OpenAI/Cohere/HF)
│   │   ├── retrieval/             # Hybrid search, reranker, relevance filter
│   │   ├── generation/            # LLM client, prompt manager, context builder
│   │   └── memory/                # Conversation history compression
│   ├── services/                  # Document processor, vector store adapters
│   ├── db/                        # SQLAlchemy models & migrations
│   ├── monitoring/                # Logging, metrics, tracing
│   └── evaluation/                # RAG evaluation & benchmarks
├── FRONTEND/
│   └── src/
│       ├── app/                   # Next.js pages
│       ├── components/            # Chat, documents, assistant sidebar
│       └── lib/                   # API client & state store
├── TESTS/                         # Unit, integration, evaluation tests
├── docker-compose.yml             # Local development stack
└── .env.example                   # Environment variable template
```

## License

MIT
