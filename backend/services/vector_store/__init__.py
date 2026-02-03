"""
Vector Store Factory.

PRODUCTION: Select vector store based on scale:
- PGVector: <10M vectors (easy if already using PostgreSQL)
- Qdrant: 10M-500M vectors (good performance, easy to manage)
- Pinecone: Managed service (no infrastructure to manage)
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from backend.core.config import settings

logger = logging.getLogger(__name__)

# Singleton instance
_vector_store = None


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def connect(self):
        """Connect to the vector store."""
        pass

    @abstractmethod
    async def upsert(self, vectors: list[dict]) -> int:
        """
        Upsert vectors.

        Args:
            vectors: List of {"id": str, "values": list[float], "metadata": dict}

        Returns:
            Number of vectors upserted
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar vectors.

        Returns:
            List of {"id": str, "score": float, "metadata": dict}
        """
        pass

    @abstractmethod
    async def delete(
        self,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ) -> None:
        """Delete vectors by IDs or filter."""
        pass

    @abstractmethod
    async def close(self):
        """Close connection."""
        pass

    async def get_all_documents(self) -> list[dict]:
        """
        Fetch all documents from the store.

        Returns:
            List of {"chunk_id": str, "content": str, "metadata": dict}
        """
        return []  # Default implementation returns empty list


class PGVectorAdapter(BaseVectorStore):
    """Adapter for PGVector store."""

    def __init__(self):
        from backend.services.vector_store.pgvector import PGVectorStore
        self._store = PGVectorStore(
            dsn=settings.database.postgres_url,
            dimension=settings.embedding.dimensions,
        )

    async def connect(self):
        await self._store.connect()

    async def upsert(self, vectors: list[dict]) -> int:
        ids = [v["id"] for v in vectors]
        embeddings = [v["values"] for v in vectors]
        metadata = [v.get("metadata", {}) for v in vectors]
        return await self._store.upsert(ids, embeddings, metadata)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        return await self._store.search(query_embedding, top_k, filter)

    async def delete(
        self,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ) -> None:
        await self._store.delete(ids, filter)

    async def close(self):
        await self._store.close()


class QdrantAdapter(BaseVectorStore):
    """Adapter for Qdrant store."""

    def __init__(self):
        from backend.services.vector_store.qdrant import QdrantStore
        self._store = QdrantStore(
            url=settings.database.qdrant_url,
            api_key=settings.database.qdrant_api_key,
            collection=settings.database.qdrant_collection,
            dimension=settings.embedding.dimensions,
        )

    async def connect(self):
        await self._store.connect()

    async def upsert(self, vectors: list[dict]) -> int:
        ids = [v["id"] for v in vectors]
        embeddings = [v["values"] for v in vectors]
        metadata = [v.get("metadata", {}) for v in vectors]
        return await self._store.upsert(ids, embeddings, metadata)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        return await self._store.search(query_embedding, top_k, filter)

    async def delete(
        self,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ) -> None:
        await self._store.delete(ids, filter)

    async def close(self):
        await self._store.close()

    async def get_all_documents(self) -> list[dict]:
        return await self._store.get_all_documents()


class PineconeAdapter(BaseVectorStore):
    """Adapter for Pinecone store."""

    def __init__(self):
        from backend.services.vector_store.pinecone import PineconeStore
        self._store = PineconeStore()

    async def connect(self):
        await self._store.connect()

    async def upsert(self, vectors: list[dict]) -> int:
        return await self._store.upsert(vectors)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        return await self._store.search(query_embedding, top_k, filter)

    async def delete(
        self,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ) -> None:
        await self._store.delete(ids, filter)

    async def close(self):
        pass  # Pinecone doesn't need explicit close


def get_vector_store() -> BaseVectorStore:
    """
    Get vector store singleton based on configuration.

    PRODUCTION selection:
    - pgvector: <10M vectors
    - qdrant: 10M-500M vectors
    - pinecone: Managed service
    """
    global _vector_store

    if _vector_store is None:
        store_type = settings.database.vector_store.lower()

        if store_type == "pgvector":
            _vector_store = PGVectorAdapter()
            logger.info("Using PGVector store")
        elif store_type == "qdrant":
            _vector_store = QdrantAdapter()
            logger.info("Using Qdrant store")
        elif store_type == "pinecone":
            _vector_store = PineconeAdapter()
            logger.info("Using Pinecone store")
        else:
            raise ValueError(f"Unknown vector store: {store_type}")

    return _vector_store


async def init_vector_store():
    """Initialize and connect to vector store."""
    store = get_vector_store()
    await store.connect()
    logger.info("Vector store initialized")
    return store


async def close_vector_store():
    """Close vector store connection."""
    global _vector_store
    if _vector_store:
        await _vector_store.close()
        _vector_store = None
        logger.info("Vector store closed")
