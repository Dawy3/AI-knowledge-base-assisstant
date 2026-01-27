"""
PGVector Store.

PostgreSQL extension for vector similarity search.
Good for <10M vectors, easy if already using PostgreSQL.
"""

import logging
import json
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)


class PGVectorStore:
    """
    PGVector store using asyncpg.
    
    Usage:
        store = PGVectorStore(dsn="postgresql://...")
        await store.connect()
        await store.upsert(ids, embeddings, metadata)
        results = await store.search(query_embedding, top_k=10)
    """
    
    def __init__(
        self,
        dsn: str,
        table: str = "embeddings",
        dimension: int = 1536,
    ):
        self.dsn = dsn
        self.table = table
        self.dimension = dimension
        self.pool: Optional[asyncpg.Pool] = None
        
    async def connect(self):
        """Connect and ensure table exists."""
        self.pool = await asyncpg.create_pool(self.dsn)
        await self._ensure_table()
        logger.info(f"PGVectorStore connected: {self.table}")

    async def _ensure_table(self):
        """Create table and index if not exists."""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self.dimension}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW()
                )     
            """)
            
            # Create HNSW index for fast search
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx
                ON {self.table}
                USING hnsw (embedding vector_cosine_ops)
            """)
    
    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """Upsert vectors."""
        
        metadata = metadata or [{} for _ in ids]
        
        async with self.pool.acquire() as conn:
            # Use copy for bulk insert
            await conn.executemany(
                f"""
                INSERT INTO {self.table} (id, embedding, metadata)
                VALUES ($1, $2, $3)
                ON CONFLICT (id) DO UPDATE
                SET embedding = $2, metadata = $3
                """,
                [
                    (id_, str(emb), json.dumps(meta))
                    for id_, emb, meta in zip(ids, embeddings, metadata)
                ],
            )
        
        return len(ids)
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """Search for similar vectors using cosine distance."""
        async with self.pool.acquire() as conn:
            # Build filter clause
            filter_clause = ""
            if filter:
                conditions = [f"metadata->>'{k}' = '{v}'" for k, v in filter.items()]
                filter_clause = "WHERE " + " AND ".join(conditions)
            
            rows = await conn.fetch(
                f"""
                SELECT id, 1 - (embedding <=> $1) as score, metadata
                FROM {self.table}
                {filter_clause}
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                str(query_embedding),
                top_k,
            )
            
            return [
                {
                    "id": row["id"],
                    "score": float(row["score"]),
                    "metadata": row["metadata"] or {},
                }
                for row in rows
            ]
    
    async def delete(
        self,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ) -> None:
        """Delete vectors."""
        async with self.pool.acquire() as conn:
            if ids:
                await conn.execute(
                    f"DELETE FROM {self.table} WHERE id = ANY($1)",
                    ids,
                )
            elif filter:
                conditions = [f"metadata->>'{k}' = '{v}'" for k, v in filter.items()]
                where = " AND ".join(conditions)
                await conn.execute(f"DELETE FROM {self.table} WHERE {where}")
    
    async def stats(self) -> dict:
        """Get table statistics."""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.table}")
            return {"vectors_count": count}
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()

            