"""
Database Session Management.

Uses SQLAlchemy async for non-blocking database operations.
PRODUCTION: Uses PostgreSQL with proper connection pooling.
"""

import logging
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from backend.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Database URLs from Config
# =============================================================================

def get_database_url() -> str:
    """Get PostgreSQL database URL from settings."""
    url = settings.database.postgres_url
    if not url:
        raise ValueError(
            "PostgreSQL URL not configured. Set DATABASE__POSTGRES_URL in .env file "
            "or use vector store only (Qdrant/Pinecone) without PostgreSQL features."
        )
    return url


def get_async_database_url() -> str:
    """Get async PostgreSQL URL (asyncpg driver)."""
    url = get_database_url()
    # Convert to async URL
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://")
    elif url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://")
    return url


# =============================================================================
# Lazy Engine Initialization
# =============================================================================

_sync_engine = None
_async_engine = None
_session_local = None
_async_session_local = None


def get_sync_engine():
    """Get or create sync engine (lazy initialization)."""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            pool_size=settings.database.postgres_pool_size,
            max_overflow=settings.database.postgres_pool_size * 2,
            echo=settings.debug,
        )
        logger.info(f"Created sync database engine: pool_size={settings.database.postgres_pool_size}")
    return _sync_engine


def get_async_engine():
    """Get or create async engine (lazy initialization)."""
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            get_async_database_url(),
            pool_pre_ping=True,
            pool_size=settings.database.postgres_pool_size,
            max_overflow=settings.database.postgres_pool_size * 2,
            echo=settings.debug,
        )
        logger.info(f"Created async database engine: pool_size={settings.database.postgres_pool_size}")
    return _async_engine


# =============================================================================
# Session Factories
# =============================================================================

def get_session_local():
    """Get or create sync session factory."""
    global _session_local
    if _session_local is None:
        _session_local = sessionmaker(
            bind=get_sync_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _session_local


def get_async_session_local():
    """Get or create async session factory."""
    global _async_session_local
    if _async_session_local is None:
        _async_session_local = async_sessionmaker(
            bind=get_async_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _async_session_local


# Backwards compatibility aliases
@property
def SessionLocal():
    return get_session_local()


@property
def AsyncSessionLocal():
    return get_async_session_local()


# =============================================================================
# Session Dependencies
# =============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database session.

    Usage in FastAPI:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    session_factory = get_async_session_local()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_db():
    """Get sync database session (for scripts/migrations)."""
    session_factory = get_session_local()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# Connection Management
# =============================================================================

async def init_db():
    """Initialize database connection and create tables if needed."""
    from backend.db.models import Base

    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully")


async def close_db():
    """Close database connections."""
    global _async_engine, _sync_engine

    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
        logger.info("Closed async database engine")

    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None
        logger.info("Closed sync database engine")


async def check_db_connection() -> bool:
    """Check if database connection is healthy."""
    from sqlalchemy import text
    try:
        async with get_async_session_local()() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
