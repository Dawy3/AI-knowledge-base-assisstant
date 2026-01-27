"""
Pinecone Vector Store.

Managed service - handles scaling, replication, backups automatically.
Recommended for most production use cases.
"""

import logging
from typing import Optional

from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


class PineconeStore:
    """
    Pinecone vector store wrapper.
    
    Usage:
        store = PineconeStore(api_key="...", index_name="my-index")
        await store.upsert(ids, embeddings, metadata)
        results = await store.search(query_embedding, top_k=10)
    """
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str = "us-east-1",
        dimension: int = 1536,
        metric: str = "cosine",
    ):
        self.client = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.environment = environment
        
        # Get or create index
        self._ensure_index()
        self.index = self.client.Index(index_name)
        
        logger.info(f"Pinecone Initialized: {index_name}")
        
    def _ensure_index(self):
        """Create index if it doesn't exist."""
        existing = [i.name for i in self.client.list_indexes()]
        
        if self.index_name not in existing:
            self.client.create_index(
                name= self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec= ServerlessSpec(cloud="aws", region=self.environment),
            )
            logger.info(f"Created index: {self.index_name}")
            
    
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: Optional[list[dict]] = None,
        namespace: str = "",
        batch_size: int = 100,
    ) -> int:
        """
        Upsert vectors.
        
        Returns number of vectors upserted.
        """
        metadata = metadata or [{} for _ in ids]

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metadata)
        ]
        
        # Batch upsert
        total = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+ batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            total += len(batch)
            
        return total
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        namespace: str = "",
        filter: Optional[dict] = None,
        include_metadata: bool = True,
    ) -> list[dict]:
        """
        Search for similar vectors.
        
        Returns list of {id, score, metadata}.
        """
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata,
        )
        
        return [
            {
                "id": match.id,
                "score" : match.score,
                "metadata": match.metadata  or {}
            }
            for match in results.matches
        ]
    
    def delete(
        self,
        ids: Optional[list[str]] = None,
        namespace: str = "",
        delete_all: bool = False,
        filter: Optional[dict] = None,  
    ) -> None:
        """Delete vectors by ID, filter, or all."""
        if delete_all:
            self.index.delete(delete_all=True, namespace=namespace)
        elif filter:
            self.index.delete(filter=filter, namespace=namespace)
        elif ids:
            self.index.delete(ids=ids, namespace=namespace)
            
    
    def stats(self) -> dict:
        """Get index statistics."""
        return self.index.describe_index_stats()