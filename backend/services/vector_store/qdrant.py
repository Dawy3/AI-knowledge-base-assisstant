"""
Qdrant Vector Store.

Self-hosted option with good performance.
Good for 10M-500M vectors.
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams, PointIdsList, FilterSelector

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Qdrant vector store wrapper.

    Usage:
        store = QdrantStore(url="http://localhost:6333", collection="my-docs")
        await store.connect()
        await store.upsert(ids, embeddings, metadata)
        results = await store.search(query_embedding, top_k=10)
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection: str = "documents",
        dimension: int = 1536,
        distance: str = "cosine",
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
    ):
        self.url = url
        self.api_key = api_key
        self.collection = collection
        self.dimension = dimension
        self.client: Optional[QdrantClient] = None

        # Map distance metric
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        self.distance = distance_map.get(distance, Distance.COSINE)

        # HNSW index configuration
        self.hnsw_config = HnswConfigDiff(
            m=hnsw_m,
            ef_construct=hnsw_ef_construct,
        )

    async def connect(self):
        """Connect to Qdrant and ensure collection exists."""
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self._ensure_collection()
        logger.info(f"QdrantStore connected: {self.collection}")

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=self.distance
                ),
                hnsw_config=self.hnsw_config,
            )
            logger.info(f"Created collection: {self.collection}")
        else:
            logger.info(f"Collection already exists: {self.collection}")

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: Optional[list[dict]] = None,
        batch_size: int = 100,
    ) -> int:
        """Upsert vectors."""
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        metadata = metadata or [{} for _ in ids]

        points = [
            PointStruct(id=id_, vector=emb, payload=meta)
            for id_, emb, meta in zip(ids, embeddings, metadata)
        ]

        # Batch upsert
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)
            total += len(batch)

        logger.info(f"Upserted {total} vectors to {self.collection}")
        return total

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """Search for similar vectors."""
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=filter
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "metadata": hit.payload or {},
            }
            for hit in results
        ]

    async def delete(
        self,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ) -> None:
        """Delete vectors by IDs or filter."""
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        if ids:
            self.client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=ids),
            )
            logger.info(f"Deleted {len(ids)} vectors from {self.collection}")
        elif filter:
            self.client.delete(
                collection_name=self.collection,
                points_selector=FilterSelector(filter=filter),
            )
            logger.info(f"Deleted vectors with filter from {self.collection}")

    async def close(self):
        """Close the client connection."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("QdrantStore connection closed")

    def stats(self) -> dict:
        """Get collection statistics."""
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        info = self.client.get_collection(self.collection)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.name,
        }
