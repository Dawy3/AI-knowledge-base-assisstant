"""
Vector Search Implementation for RAG Pipeline.

FOCUS: HNSW for <100M vectors (95%+ recall)
MUST: Configure M=16, ef_search=100 for HNSW
OPTIONS: IVF_PQ for >100M vectors

Supports multiple backends:
- Pinecone (managed, scalable)
- Qdrant (open-source, HNSW native)
- In-memory (development/testing)
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from pinecone import Pinecone
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

import numpy as np

logger = logging.getLogger(__name__)


class VectorIndexType(str, Enum):
    """Vector Index Type"""
    HNSW = "hnsw"       # <100M vectors, 95%+ recall
    IVF_PQ = "ivg_pq"   # >100M vectors, memory efficient
    FLAT = "flat"      # Exact search, small datasets only
    
@dataclass
class VectorSearchResult:
    """Result from vector Search."""
    
    
    chunk_id: str
    score: float # Similarity score (0-1, higher is better)
    content: str = ""
    metadata : dict = field(default_factory=dict)
    
    # Source tracking
    document_id: str = ""
    chunk_index: int = 0
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
        }
        

@dataclass
class VectorSearchResponse:
    """Response containing multiple search results."""
    
    results: list[VectorSearchResult]
    query_embedding: list[float] = field(default_factory=list)
    
    # Performance metrics
    latency_ms: float = 0.0
    total_candidates: int = 0
    
    @property
    def top_score(self) -> float:
        return self.results[0].score if self.results else 0.0


@dataclass
class HNSWConfig:
    """
    HNSW index configuration.
    
    MUST: Configure M=16, ef_search=100 for 95%+ recall
    """
    
    # Graph construction parameters
    m: int = 16  # Number of connections per layer (16 recommended)
    ef_construction: int = 200  # Size of dynamic candidate list during construction
    
    # Search parameters
    ef_search: int = 100  # Size of dynamic candidate list during search
    
    # Performance tuning
    num_threads: int = 4
    
    def validate(self) -> None:
        """Validate HNSW parameters."""
        if self.m < 4 or self.m > 64:
            raise ValueError(f"M should be between 4-64, got {self.m}")
        if self.ef_search < self.m:
            logger.warning(
                f"ef_search ({self.ef_search}) should be >= M ({self.m}) for good recall"
            )
    

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """Insert or update vectors."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete(self, ids: list[str]) -> int:
        """Delete Vectors by Id."""
        pass
    
    
class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone vector store implementation.
    
    Managed service with automatic scaling and high availability.
    """
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str = "us-east-1",
        namespace: str = "",
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.environment = environment
        self.namespace = namespace
        self._index = None

         
    async def _get_index(self):
        """Lazy initialization of Pinecone index."""
        if self._index is None:
            try:
                pc = Pinecone(api_key=self.api_key)
                self._index=  pc.Index(self.index_name)
                
                logger.info(f"Connected to Pinecone index: {self.index_name}")
            except ImportError:
                raise ImportError("pinecone-client required: pip install pinecone-client")
        return self._index
    
    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """Insert or update vectors in Pinecone."""
        index = await self._get_index()
    
        # Prepare vectors
        vectors = []
        for i, (_id, embedding) in enumerate(index):
            
            vector = {
                "id": _id,
                "value" : embedding
            }
            if metadata and i < len(metadata):
                vector["metadata"] = metadata[i]
            vectors.append(vector)
            
        # Batch upsert (Pinecone limit is 100 per batch)
        batch_size = 100
        total_upserted = 0
        
        loop  = asyncio.get_event_loop()
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            result = await loop.run_in_executor(
                None,
                lambda b=batch: index.upsert(vectors=b, namespace=self.namespace)
            )
            total_upserted += result.upserted_count
            
        return total_upserted
            
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors in Pinecone."""
        index = await self._get_index()
        
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector= query_embedding,
                top_k = top_k,
                filter = filter,
                include_metadata=True,
                namespace= self.namespace,
            )
        )
        
        search_results = []
        for match in result.matches:
            search_results.append(VectorSearchResult(
                chunk_id= match.id,
                score=match.score,
                content=match.metadata.get("content", "") if match.metadata else "",
                metadata=match.metadata or {},
                document_id=match.metadata.get("document_id", "") if match.metadata else "",
                chunk_index=match.metadata.get("chunk_index", 0) if match.metadata else 0,
            ))
            
        return search_results
    
    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID from Pinecone."""
        index = await self._get_index()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: index.delete(ids=ids, namespace=self.namespace)
        )
        
        return len(ids)


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store implementation with HNSW index.

    Open-source vector database with native HNSW support.
    Configurable M and ef parameters for 95%+ recall.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "documents",
        dimensions: int = 1536,
        hnsw_config: Optional[HNSWConfig] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.hnsw_config = hnsw_config or HNSWConfig()
        self._client: Optional[QdrantClient] = None

        # Connection parameters
        self._host = host
        self._port = port
        self._api_key = api_key
        self._url = url

    async def _get_client(self) -> QdrantClient:
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            try:
                if self._url:
                    # Cloud or remote URL connection
                    self._client = QdrantClient(
                        url=self._url,
                        api_key=self._api_key,
                    )
                else:
                    # Local connection
                    self._client = QdrantClient(
                        host=self._host,
                        port=self._port,
                    )

                # Create collection if it doesn't exist
                await self._ensure_collection()

                logger.info(f"Connected to Qdrant collection: {self.collection_name}")
            except ImportError:
                raise ImportError("qdrant-client required: pip install qdrant-client")
        return self._client

    async def _ensure_collection(self) -> None:
        """Create collection with HNSW config if it doesn't exist."""
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimensions,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=self.hnsw_config.m,
                        ef_construct=self.hnsw_config.ef_construction,
                    ),
                ),
            )
            logger.info(
                f"Created Qdrant collection '{self.collection_name}' with HNSW: "
                f"M={self.hnsw_config.m}, ef_construct={self.hnsw_config.ef_construction}"
            )

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """Insert or update vectors in Qdrant."""
        client = await self._get_client()

        points = []
        for i, (id_, embedding) in enumerate(zip(ids, embeddings)):
            payload = metadata[i] if metadata and i < len(metadata) else {}
            points.append(
                PointStruct(
                    id=id_,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Batch upsert
        batch_size = 100
        total_upserted = 0

        loop = asyncio.get_event_loop()

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await loop.run_in_executor(
                None,
                lambda b=batch: client.upsert(
                    collection_name=self.collection_name,
                    points=b,
                )
            )
            total_upserted += len(batch)

        return total_upserted

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors in Qdrant using HNSW."""
        client = await self._get_client()

        # Convert filter to Qdrant format
        qdrant_filter = self._build_filter(filter) if filter else None

        loop = asyncio.get_event_loop()

        results = await loop.run_in_executor(
            None,
            lambda: client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                search_params={
                    "hnsw_ef": self.hnsw_config.ef_search,
                },
            )
        )

        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(VectorSearchResult(
                chunk_id=str(hit.id),
                score=hit.score,
                content=payload.get("content", ""),
                metadata=payload,
                document_id=payload.get("document_id", ""),
                chunk_index=payload.get("chunk_index", 0),
            ))

        return search_results

    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID from Qdrant."""
        client = await self._get_client()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: client.delete(
                collection_name=self.collection_name,
                points_selector=ids,
            )
        )

        return len(ids)

    def _build_filter(self, filter: dict) -> Filter:
        """Convert generic filter dict to Qdrant Filter."""
        conditions = []

        for key, value in filter.items():
            if isinstance(value, dict):
                # Handle operators
                for op, op_value in value.items():
                    if op == "$eq":
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=op_value))
                        )
                    elif op == "$in":
                        # Qdrant handles $in differently - add each as OR
                        for v in op_value:
                            conditions.append(
                                FieldCondition(key=key, match=MatchValue(value=v))
                            )
                    elif op in ("$gt", "$gte", "$lt", "$lte"):
                        range_params = {}
                        if op == "$gt":
                            range_params["gt"] = op_value
                        elif op == "$gte":
                            range_params["gte"] = op_value
                        elif op == "$lt":
                            range_params["lt"] = op_value
                        elif op == "$lte":
                            range_params["lte"] = op_value
                        conditions.append(
                            FieldCondition(key=key, range=Range(**range_params))
                        )
            else:
                # Direct equality
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=conditions) if conditions else None


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store for development and testing.
    
    Uses brute-force search (FLAT index).
    NOT recommended for production with >10K vectors.
    """
    
    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}
    
    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """Insert or update vectors in memory."""
        for i, (id_, embedding) in enumerate(zip(ids, embeddings)):
            self._vectors[id_] = np.array(embedding, dtype=np.float32)
            if metadata and i < len(metadata):
                self._metadata[id_] = metadata[i]
            else:
                self._metadata[id_] = {}
        
        return len(ids)
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using cosine similarity."""
        if not self._vectors:
            return []
        
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        
        scores = []
        for id_, vector in self._vectors.items():
            # Apply metadata filter if provided
            if filter:
                meta = self._metadata.get(id_, {})
                if not self._matches_filter(meta, filter):
                    continue
            
            # Cosine similarity
            vector_norm = vector / (np.linalg.norm(vector) + 1e-9)
            score = float(np.dot(query_norm, vector_norm))
            scores.append((id_, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for id_, score in scores[:top_k]:
            meta = self._metadata.get(id_, {})
            results.append(VectorSearchResult(
                chunk_id=id_,
                score=score,
                content=meta.get("content", ""),
                metadata=meta,
                document_id=meta.get("document_id", ""),
                chunk_index=meta.get("chunk_index", 0),
            ))
        
        return results
    
    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID from memory."""
        deleted = 0
        for id_ in ids:
            if id_ in self._vectors:
                del self._vectors[id_]
                del self._metadata[id_]
                deleted += 1
        return deleted
    
    def _matches_filter(self, metadata: dict, filter: dict) -> bool:
        """Check if metadata matches filter conditions."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            
            if isinstance(value, dict):
                # Handle operators like $eq, $in, $gt, etc.
                for op, op_value in value.items():
                    if op == "$eq" and metadata[key] != op_value:
                        return False
                    elif op == "$in" and metadata[key] not in op_value:
                        return False
                    elif op == "$gt" and not (metadata[key] > op_value):
                        return False
                    elif op == "$gte" and not (metadata[key] >= op_value):
                        return False
                    elif op == "$lt" and not (metadata[key] < op_value):
                        return False
                    elif op == "$lte" and not (metadata[key] <= op_value):
                        return False
            elif metadata[key] != value:
                return False
        
        return True
    

class VectorSearch:
    """
    Vector search interface for RAG pipeline.
    
    FOCUS: HNSW for <100M vectors (95%+ recall)
    
    Provides unified interface for different vector store backends.
    
    Usage:
        search = VectorSearch(store=PineconeVectorStore(...))
        results = await search.search(query_embedding, top_k=100)
    """
    def __init__(
        self,
        store : BaseVectorStore,
        hnsw_config: Optional[HNSWConfig] = None,
        default_top_k: int = 100,
    ):
        """
        Initialize vector search.
        
        Args:
            store: Vector store backend
            hnsw_config: HNSW configuration (if applicable)
            default_top_k: Default number of results to return
        """
        self.store = store
        self.hnsw_config = hnsw_config or HNSWConfig()
        self.default_top_k = default_top_k
        
        # Validate HNSW config
        self.hnsw_config.validate()
        
        logger.info(
            f"Initialized VectorSearch with HNSW config: "
            f"M={self.hnsw_config.m}, ef_search={self.hnsw_config.ef_search}"
        )

    async def index(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """
        Index chunks with their embeddings.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and metadata
            embeddings: Corresponding embeddings
            
        Returns:
            Number of chunks indexed
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")
        
        ids = [c["chunk_id"] for c in chunks]
        metadata = []
        
        for chunk in chunks:
            meta  = {
                "content": chunk.get("content", ""),
                "document_id": chunk.get("document_id", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "source_file": chunk.get("source_file", ""),
            }
            # Add any additional metadata
            if "metadata" in chunk:
                meta.update(chunk["metadata"])
            metadata.append(meta)
            
        return await self.store.upsert(ids, embeddings, metadata)
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> VectorSearchResponse:
        """
        Search for similar chunks.
        
        MUST: Retrieve top-100 for reranking to top-10
        
        Args:
            query_embedding: Query vector
            top_k: Number of results (default: 100 for reranking)
            filter: Optional metadata filter
            
        Returns:
            VectorSearchResponse with results and metrics
        """
        start_time = time.time()
        
        k = top_k or self.default_top_k
        
        results = await self.store.search(
            query_embedding=query_embedding,
            top_k=k,
            filter=filter,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.debug(
            f"Vector search returned {len(results)} results in {latency_ms:.2f}ms"
        )
        
        return VectorSearchResponse(
            results=results,
            query_embedding=query_embedding,
            latency_ms=latency_ms,
            total_candidates=len(results),
        )
        
    async def delete_by_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.
        
        Note: This requires metadata filtering support in the store.
        For Pinecone, you may need to track chunk IDs separately.
        """
        # This is a simplified implementation
        # Production systems should maintain a document->chunk_ids mapping
        logger.warning(
            "delete_by_document requires chunk ID tracking. "
            "Consider maintaining a separate document->chunks index."
        )
        return 0
    
    async def get_stats(self) -> dict:
        """Get vector store statistics."""
        # Implementation depends on store backend
        return {
            "hnsw_m": self.hnsw_config.m,
            "hnsw_ef_search": self.hnsw_config.ef_search,
            "default_top_k": self.default_top_k,
        }


def vector_store_search(
    backend: str = "pinecone",
    **kwargs,
) -> VectorSearch:
    """
    Factory function to create vector search with specified backend.

    Args:
        backend: Backend type ("pinecone", "qdrant", "memory")
        **kwargs: Backend-specific arguments

    Returns:
        Configured VectorSearch instance
    """

    if backend == "pinecone":
        store = PineconeVectorStore(
            api_key=kwargs.get("api_key", ""),
            index_name=kwargs.get("index_name", ""),
            environment=kwargs.get("environment", "us-east-1"),
            namespace=kwargs.get("namespace", ""),
        )
    elif backend == "qdrant":
        store = QdrantVectorStore(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 6333),
            collection_name=kwargs.get("collection_name", "documents"),
            dimensions=kwargs.get("dimensions", 1536),
            hnsw_config=kwargs.get("hnsw_config"),
            api_key=kwargs.get("api_key"),
            url=kwargs.get("url"),
        )
    elif backend == "memory":
        store = InMemoryVectorStore(
            dimensions=kwargs.get("dimensions", 1536)
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return VectorSearch(
        store=store,
        hnsw_config=kwargs.get("hnsw_config"),
        default_top_k=kwargs.get("default_top_k", 100),
    )

        