"""
FastAPI Dependencies.

FOCUS: DB sessions, auth, rate limiters
Provides reusable dependencies injected into route handlers.
"""

import time
from collections import defaultdict
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import Settings, get_settings
from backend.db.session import AsyncSessionLocal, AsyncSession, get_async_session_local


# =============================================================================
# Database Dependencies
# =============================================================================

async def get_db():
    """
    Get async database session.

    Usage:
        @router.get("/items")
        async def get_items(db: DbSession):
            ...
    """
    session_factory = get_async_session_local()  # Use the correct function
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Type alias for cleaner route signatures
DbSession = Annotated[AsyncSession, Depends(get_db)]


# =============================================================================
# Settings Dependency
# =============================================================================

def get_current_settings() -> Settings:
    """Get application settings."""
    return get_settings()


CurrentSettings = Annotated[Settings, Depends(get_current_settings)]


# =============================================================================
# Authentication Dependencies
# =============================================================================

async def verify_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None,
    settings: Settings = Depends(get_current_settings),
) -> str:
    """
    Verify API key from header.

    Usage:
        @router.get("/protected")
        async def protected_route(api_key: ValidApiKey):
            ...
    """
    if settings.is_development and not x_api_key:
        return "dev-mode"

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # TODO: Validate against stored API keys in production
    # For now, accept any non-empty key in non-production
    if settings.is_production:
        # Add proper API key validation here
        pass

    return x_api_key


ValidApiKey = Annotated[str, Depends(verify_api_key)]


async def get_current_user_id(
    x_user_id: Annotated[Optional[str], Header()] = None,
    api_key: str = Depends(verify_api_key),
) -> Optional[str]:
    """
    Get current user ID from header.

    Optional - allows anonymous access but enables user-specific features.
    """
    return x_user_id


CurrentUserId = Annotated[Optional[str], Depends(get_current_user_id)]


# =============================================================================
# Rate Limiting Dependencies
# =============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter.

    For production, use Redis-based rate limiting.
    """

    def __init__(self):
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        window_start = now - window_seconds

        # Clean old requests
        self.requests[key] = [
            ts for ts in self.requests[key] if ts > window_start
        ]

        if len(self.requests[key]) >= max_requests:
            return False

        self.requests[key].append(now)
        return True

    def get_retry_after(self, key: str, window_seconds: int) -> int:
        """Get seconds until rate limit resets."""
        if not self.requests[key]:
            return 0
        oldest = min(self.requests[key])
        return max(0, int(window_seconds - (time.time() - oldest)))


# Global rate limiter instance
_rate_limiter = RateLimiter()


async def check_rate_limit(
    request: Request,
    settings: Settings = Depends(get_current_settings),
) -> None:
    """
    Check rate limit for current request.

    Uses client IP as rate limit key.
    """
    client_ip = request.client.host if request.client else "unknown"

    if not _rate_limiter.is_allowed(
        key=client_ip,
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window,
    ):
        retry_after = _rate_limiter.get_retry_after(
            client_ip, settings.rate_limit_window
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


RateLimited = Annotated[None, Depends(check_rate_limit)]


# =============================================================================
# Combined Dependencies for Common Patterns
# =============================================================================

async def get_authenticated_context(
    db: DbSession,
    user_id: CurrentUserId,
    settings: CurrentSettings,
    _rate_limit: RateLimited,
) -> dict:
    """
    Combined dependency for authenticated endpoints.

    Provides db session, user context, and enforces rate limiting.

    Usage:
        @router.post("/chat")
        async def chat(ctx: AuthenticatedContext):
            db = ctx["db"]
            user_id = ctx["user_id"]
    """
    return {
        "db": db,
        "user_id": user_id,
        "settings": settings,
    }


AuthenticatedContext = Annotated[dict, Depends(get_authenticated_context)]
