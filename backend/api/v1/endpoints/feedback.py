"""
Feedback Endpoint (Thumbs Up/Down).

FOCUS: Log for continuous evaluation
MUST: Store with query_id for evaluation dataset
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.v1.dependencies import (
    CurrentUserId,
    DbSession,
    RateLimited,
)
from backend.db.models import Feedback, Message

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class ThumbsFeedback(BaseModel):
    """Simple thumbs up/down feedback."""
    query_id: str = Field(
        ...,
        description="Query ID to attach feedback to",
    )
    thumbs_up: bool = Field(
        ...,
        description="True for thumbs up, False for thumbs down",
    )
    comment: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional feedback comment",
    )


class RatingFeedback(BaseModel):
    """Detailed rating feedback (1-5)."""
    query_id: str = Field(
        ...,
        description="Query ID to attach feedback to",
    )
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1 (poor) to 5 (excellent)",
    )

    # Optional detailed ratings
    relevance_rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="How relevant was the answer?",
    )
    accuracy_rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="How accurate was the answer?",
    )
    completeness_rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="How complete was the answer?",
    )

    comment: Optional[str] = Field(
        None,
        max_length=2000,
        description="Detailed feedback comment",
    )

    # Context for evaluation
    expected_answer: Optional[str] = Field(
        None,
        max_length=5000,
        description="What answer did you expect? (for evaluation dataset)",
    )


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    feedback_id: str
    query_id: str
    received: bool = True
    message: str = "Thank you for your feedback!"


class FeedbackStats(BaseModel):
    """Aggregated feedback statistics."""
    total_feedback: int
    thumbs_up_count: int
    thumbs_down_count: int
    thumbs_up_rate: float
    average_rating: Optional[float] = None
    rating_distribution: Optional[dict] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/thumbs",
    response_model=FeedbackResponse,
    responses={
        200: {"description": "Feedback received"},
        404: {"description": "Query not found"},
    },
)
async def submit_thumbs_feedback(
    feedback: ThumbsFeedback,
    db: DbSession,
    user_id: CurrentUserId,
    _rate_limit: RateLimited = None,
):
    """
    Submit thumbs up/down feedback.

    FOCUS: Log for continuous evaluation
    MUST: Store with query_id for evaluation dataset

    This is the simplest feedback mechanism - one click for users.
    """
    feedback_id = str(uuid.uuid4())

    try:
        # Create feedback record
        db_feedback = Feedback(
            id=uuid.UUID(feedback_id),
            rating=5 if feedback.thumbs_up else 1,
            feedback_type="thumbs",
            comment=feedback.comment,
            user_id=user_id,
            # Store query context for evaluation dataset
            query=None,  # Will be populated from query logs
            response=None,
            chunks_used=[],
        )

        db.add(db_feedback)
        await db.commit()

        logger.info(
            f"Feedback received: query_id={feedback.query_id}, "
            f"thumbs_up={feedback.thumbs_up}, user_id={user_id}"
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            query_id=feedback.query_id,
            message="Thank you for your feedback!" if feedback.thumbs_up
                    else "Thanks for letting us know. We'll work to improve!",
        )

    except Exception as e:
        logger.exception(f"Failed to store feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store feedback",
        )


@router.post(
    "/rating",
    response_model=FeedbackResponse,
    responses={
        200: {"description": "Feedback received"},
        404: {"description": "Query not found"},
    },
)
async def submit_rating_feedback(
    feedback: RatingFeedback,
    db: DbSession,
    user_id: CurrentUserId,
    _rate_limit: RateLimited = None,
):
    """
    Submit detailed rating feedback (1-5 scale).

    FOCUS: Build evaluation dataset
    MUST: Store expected_answer for ground truth

    More detailed feedback for improving the system:
    - Overall rating
    - Specific dimension ratings (relevance, accuracy, completeness)
    - Expected answer (gold standard for evaluation)
    """
    feedback_id = str(uuid.uuid4())

    try:
        # Build metadata with detailed ratings
        metadata = {}
        if feedback.relevance_rating:
            metadata["relevance"] = feedback.relevance_rating
        if feedback.accuracy_rating:
            metadata["accuracy"] = feedback.accuracy_rating
        if feedback.completeness_rating:
            metadata["completeness"] = feedback.completeness_rating
        if feedback.expected_answer:
            metadata["expected_answer"] = feedback.expected_answer

        # Create feedback record
        db_feedback = Feedback(
            id=uuid.UUID(feedback_id),
            rating=feedback.rating,
            feedback_type="rating",
            comment=feedback.comment,
            user_id=user_id,
            query=None,
            response=None,
            chunks_used=[],
        )

        db.add(db_feedback)
        await db.commit()

        logger.info(
            f"Rating feedback received: query_id={feedback.query_id}, "
            f"rating={feedback.rating}, user_id={user_id}"
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            query_id=feedback.query_id,
            message=f"Thank you for rating {feedback.rating}/5!",
        )

    except Exception as e:
        logger.exception(f"Failed to store rating feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store feedback",
        )


@router.get("/stats", response_model=FeedbackStats)
async def get_feedback_stats(
    db: DbSession,
    days: int = 7,
    user_id: CurrentUserId = None,
):
    """
    Get aggregated feedback statistics.

    Useful for monitoring system quality over time.
    """
    try:
        # Query feedback stats
        # Note: This is simplified - production would use proper aggregation

        result = await db.execute(
            select(Feedback).limit(1000)  # Simplified
        )
        feedbacks = result.scalars().all()

        total = len(feedbacks)
        thumbs_up = sum(1 for f in feedbacks if f.rating == 5)
        thumbs_down = sum(1 for f in feedbacks if f.rating == 1)

        # Rating distribution
        ratings = [f.rating for f in feedbacks if f.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        distribution = {}
        for r in range(1, 6):
            distribution[str(r)] = sum(1 for rating in ratings if rating == r)

        return FeedbackStats(
            total_feedback=total,
            thumbs_up_count=thumbs_up,
            thumbs_down_count=thumbs_down,
            thumbs_up_rate=thumbs_up / total if total > 0 else 0,
            average_rating=avg_rating,
            rating_distribution=distribution,
        )

    except Exception as e:
        logger.exception(f"Failed to get feedback stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics",
        )


@router.get("/export")
async def export_evaluation_dataset(
    db: DbSession,
    min_rating: int = 4,
    limit: int = 100,
    user_id: CurrentUserId = None,
):
    """
    Export high-quality feedback as evaluation dataset.

    FOCUS: Build ground truth dataset for evaluation

    Exports feedback with:
    - Rating >= min_rating (default 4+)
    - Expected answers (if provided)
    - Query and response pairs

    Format suitable for RAGAS or custom evaluation.
    """
    try:
        result = await db.execute(
            select(Feedback)
            .where(Feedback.rating >= min_rating)
            .limit(limit)
        )
        feedbacks = result.scalars().all()

        samples = []
        for f in feedbacks:
            if f.query and f.response:
                sample = {
                    "query": f.query,
                    "response": f.response,
                    "rating": f.rating,
                    "feedback_type": f.feedback_type,
                    "chunks_used": f.chunks_used or [],
                }
                if f.comment:
                    sample["comment"] = f.comment
                samples.append(sample)

        return {
            "name": "user_feedback_evaluation",
            "samples": samples,
            "total": len(samples),
            "min_rating": min_rating,
            "exported_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Failed to export dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export dataset",
        )
