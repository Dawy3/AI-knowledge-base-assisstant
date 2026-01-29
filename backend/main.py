"""
FastAPI Application Entry Point.

FOCUS: Health checks, CORS, middleware setup
MUST: Add request logging, error handling
PRODUCTION: Initialize all services on startup
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.core.config import settings
from backend.api.v1.router import api_router
from backend.monitoring.logging import setup_logging, request_id_var
from backend.monitoring.metrics import MetricsCollector

# Setup structured logging
setup_logging(
    level=settings.monitoring.log_level,
    json_format=settings.monitoring.log_format == "json",
)

logger = logging.getLogger(__name__)
metrics = MetricsCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    PRODUCTION:
    - Startup: Initialize all connections and services
    - Shutdown: Close connections gracefully
    """
    # =================================================================
    # Startup
    # =================================================================
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} "
        f"in {settings.environment.value} mode"
    )

    # Initialize database
    try:
        from backend.db.session import init_db
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Don't fail startup - allow graceful degradation

    # Initialize vector store
    try:
        from backend.services.vector_store import init_vector_store
        await init_vector_store()
        logger.info("Vector store initialized")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")

    # Initialize Redis for caching (optional)
    try:
        if settings.cache.semantic_cache_enabled:
            import redis
            redis_client = redis.Redis.from_url(settings.cache.redis_url)
            redis_client.ping()
            app.state.redis = redis_client
            logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis not available (using in-memory cache): {e}")
        app.state.redis = None

    # Warm up embedding model
    try:
        from backend.core.embedding.generator import create_embedding_generator
        embedding_gen = create_embedding_generator()
        # Warm up with a test embedding
        await embedding_gen.embed_texts(["warmup"])
        logger.info(f"Embedding model loaded: {embedding_gen.model_id}")
    except Exception as e:
        logger.warning(f"Embedding model warmup failed: {e}")

    logger.info("Application startup complete")

    yield

    # =================================================================
    # Shutdown
    # =================================================================
    logger.info("Shutting down application...")

    # Close database connections
    try:
        from backend.db.session import close_db
        await close_db()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")

    # Close vector store
    try:
        from backend.services.vector_store import close_vector_store
        await close_vector_store()
        logger.info("Vector store closed")
    except Exception as e:
        logger.error(f"Error closing vector store: {e}")

    # Close Redis
    if hasattr(app.state, 'redis') and app.state.redis:
        try:
            app.state.redis.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")

    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG-based Knowledge Assistant API",
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    openapi_url="/openapi.json" if settings.is_development else None,
    lifespan=lifespan,
)


# =============================================================================
# Middleware
# =============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Callable):
    """
    Log every request with timing and request ID.

    MUST: Add request logging for debugging and monitoring.
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    request_id_var.set(request_id)

    # Add request ID to response headers
    start_time = time.perf_counter()

    # Log request
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={"extra_data": {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
        }},
    )

    try:
        response = await call_next(request)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"

        # Log response
        logger.info(
            f"Request completed: {response.status_code} in {latency_ms:.2f}ms",
            extra={"extra_data": {
                "request_id": request_id,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            }},
        )

        # Record metrics
        metrics.record_request(
            endpoint=request.url.path,
            status="success" if response.status_code < 400 else "error",
        )

        return response

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"Request failed: {str(e)}",
            extra={"extra_data": {
                "request_id": request_id,
                "error": str(e),
                "latency_ms": latency_ms,
            }},
        )
        raise


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with clear messages."""
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
        })

    logger.warning(
        f"Validation error: {errors}",
        extra={"extra_data": {"errors": errors}},
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": errors,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    MUST: Never expose internal errors to clients in production.
    """
    request_id = request_id_var.get("")

    logger.exception(
        f"Unhandled exception: {str(exc)}",
        extra={"extra_data": {"request_id": request_id}},
    )

    # Don't expose internal details in production
    if settings.is_production:
        detail = "An internal error occurred. Please try again later."
    else:
        detail = str(exc)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": detail,
            "request_id": request_id,
        },
    )


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Basic health check endpoint.

    Returns 200 if the application is running.
    """
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment.value,
    }


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check for Kubernetes/load balancers.

    Checks if all dependencies are available.
    """
    checks = {}

    # Check database
    try:
        from backend.db.session import check_db_connection
        checks["database"] = await check_db_connection()
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        checks["database"] = False

    # Check vector store
    try:
        from backend.services.vector_store import get_vector_store
        vector_store = get_vector_store()
        checks["vector_store"] = vector_store is not None
    except Exception as e:
        logger.warning(f"Vector store health check failed: {e}")
        checks["vector_store"] = False

    # Check cache (optional)
    try:
        if hasattr(app.state, 'redis') and app.state.redis:
            app.state.redis.ping()
            checks["cache"] = True
        else:
            checks["cache"] = True  # In-memory fallback is OK
    except Exception as e:
        logger.warning(f"Cache health check failed: {e}")
        checks["cache"] = False

    all_healthy = all(checks.values())

    return JSONResponse(
        status_code=status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
        },
    )


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """
    Liveness check for Kubernetes.

    Returns 200 if the application is alive (not deadlocked).
    """
    return {"status": "alive"}


# =============================================================================
# Include API Routers
# =============================================================================

app.include_router(api_router, prefix=settings.api_prefix)


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": f"{settings.api_prefix}/docs" if settings.is_development else None,
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.monitoring.log_level.lower(),
    )
