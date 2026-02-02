# AI Knowledge Assistant

A RAG-powered AI assistant that answers questions from your documents. Designed as an embeddable sidebar widget.

![Next.js](https://img.shields.io/badge/Next.js-14-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Qdrant](https://img.shields.io/badge/Qdrant-HNSW-red)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

## Features

- **RAG-Powered Responses** — Answers grounded in your actual documents
- **Document Upload** — Support for PDF, DOCX, TXT, Markdown
- **Semantic Caching** — Instant responses for similar queries
- **Streaming** — Real-time response generation
- **Source Citations** — View relevant document chunks with scores
- **Embeddable Widget** — AI-style sidebar for any web app

## Quick Start

```bash
# Clone and start
git clone <repo-url>
cd ai-knowledge-assistant
docker-compose up -d

# Access
# Frontend: http://localhost:3000
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Qdrant: http://localhost:6333/dashboard
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Next.js   │────▶│   FastAPI   │────▶│   Qdrant    │
│   Frontend  │     │   Backend   │     │  (vectors)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┼──────┐     ┌─────────────┐
                    │             │────▶│ PostgreSQL  │
                    │   LLM API   │     │ (relational)│
                    └─────────────┘     └─────────────┘
```

## Project Structure

```
├── backend/
│   ├── api/v1/endpoints/    # REST endpoints
│   ├── core/
│   │   ├── generation/      # LLM client
│   │   ├── ingestion/       # Document processing
│   │   └── retrieval/       # Vector search
│   └── models/              # Database models
├── frontend/
│   └── src/
│       ├── app/             # Next.js pages
│       ├── components/
│       │   ├── assistant/   # Sidebar widget
│       │   ├── chat/        # Chat components
│       │   ├── documents/   # Upload & manage
│       │   └── landing/     # Landing page
│       └── lib/             # API client & store
└── docker-compose.yml
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat` | Send message, get response |
| POST | `/api/v1/documents/upload` | Upload document |
| GET | `/api/v1/documents` | List documents |
| DELETE | `/api/v1/documents/{id}` | Delete document |
| POST | `/api/v1/feedback` | Submit feedback |

## Environment Variables

```env
# Backend
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://...
EMBEDDING_MODEL=text-embedding-3-small

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=           # Optional, for Qdrant Cloud
QDRANT_COLLECTION=documents

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## License

MIT
