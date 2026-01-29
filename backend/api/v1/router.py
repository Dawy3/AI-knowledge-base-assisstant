"""
API v1 Router.

Aggregates all v1 endpoint routers.
"""

from fastapi import APIRouter

from backend.api.v1.endpoints import chat, query, feedback, documents

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"],
)

api_router.include_router(
    query.router,
    prefix="/query",
    tags=["Query"],
)

api_router.include_router(
    feedback.router,
    prefix="/feedback",
    tags=["Feedback"],
)

api_router.include_router(
    documents.router,
    prefix="/documents",
    tags=["Documents"],
)
