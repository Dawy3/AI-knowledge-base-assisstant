"""
Qdrant Vector Store.

Self-hosted option with good performance.
Good for 10M-500M vectors.
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

class QdrantStore:
    """
    Qdrant vector store wrapper.
    
    Usage:
        store = QdrantStore(url="http://localhost:6333", collection="my-docs")
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
    ):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.dimension = dimension
        
        # Map distance metric
        distance_map = {
            "cosine" : Distance.COSINE,     # Default
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        
        self.distance = distance_map.get(distance, Distance.COSINE)
        
        self._ensure_collection()
        logger.info(f"QdrantStore initialized: {collection}")
        
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collection().collections]
        
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=self.distance
                ),
            )
            logger.info(f"Created collection: {self.collection}")
            
    def upsert(
        self,
        ids: list[str],
        embeddings : list[list[float]],
        metadata: Optional[list[dict]] = None,
        batch_size: int = 100,
    ) -> int:
        """Upsert vectors."""
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
            
        return total
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """Search for similar vectors."""
        results = self.client.search(
            collection_name=self.collection,
            query_vector= query_embedding,
            limit=top_k,
            query_filter=filter
        )
        
        return [
            {
                "id" : hit.id,
                "score": hit.score,
                "metadata": hit.metadata or {},
            }
            for hit in results
        ]
        
    
    def delete(
        self,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ) -> None:
        """Delete vectors."""
        if ids:
            self.client.delete(
                collection_name=self.collection,
                points_selector=ids,
            )
        elif filter:
            self.client.delete(
                collection_name=self.collection,
                points_selector=filter,
            )
        
    def stats(self) -> dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }
