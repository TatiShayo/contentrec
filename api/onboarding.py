"""FastAPI router for DPP-based user onboarding."""

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
from starlette.concurrency import run_in_threadpool
from utils.dpp import DPPSelector
from data.items import get_all_items, get_items_by_ids
from data.feedback import add_feedback

router = APIRouter()

class QuizRequest(BaseModel):
    model_config = {"extra": "forbid"}

    n_quiz: int = Field(8, ge=3, le=15, description="Number of quiz items to return")

class OnboardingSubmit(BaseModel):
    model_config = {"extra": "forbid"}

    user_id: str = Field(..., min_length=1, max_length=256, description="Unique identifier for the cold-start user")
    ratings: Dict[str, float] = Field(..., description="Dictionary mapping item_id -> rating (e.g. 1.0 for like, -1.0 or 0.0 for dislike/skip)")

    @field_validator("ratings")
    @classmethod
    def _limit_ratings_size(cls, v):
        if len(v) > 200:
            raise ValueError("ratings may not contain more than 200 entries")
        return v

@router.post("/onboarding/quiz")
async def get_onboarding_quiz(request: Request, body: Optional[QuizRequest] = None):
    """Retrieve a diverse set of onboarding items chosen via Determinantal Point Process (DPP)."""
    n_quiz = body.n_quiz if body else 8
    
    # Run in thread pool to prevent blocking the event loop
    def select_items():
        all_items = get_all_items()
        return DPPSelector.select_diverse_items(all_items, pool_size=50, n_quiz=n_quiz)
        
    quiz_items = await run_in_threadpool(select_items)
    
    if not quiz_items:
        raise HTTPException(status_code=404, detail="No items available to generate quiz.")
        
    return {
        "items": [
            {
                "item_id": item["item_id"],
                "title": item["title"],
                "category": item["category"],
                "tags": item["tags"]
            } for item in quiz_items
        ]
    }

@router.post("/onboarding/submit")
async def submit_onboarding_ratings(submit: OnboardingSubmit, request: Request):
    """Submit ratings for onboarding items, logging them to database to bootstrap user taste profiles."""
    engine = getattr(request.app.state, 'engine', None)
    
    # 1. Log ratings as feedback in SQLite
    def log_feedback():
        logged_count = 0
        for item_id, rating in submit.ratings.items():
            # If rating > 0: positive interaction ('like'), else neutral/negative ('view')
            event = "like" if rating > 0 else "view"
            add_feedback(
                user_id=submit.user_id,
                item_id=item_id,
                event_type=event,
                dwell_time=30.0 if rating > 0 else 5.0
            )
            logged_count += 1
        return logged_count
        
    count = await run_in_threadpool(log_feedback)
    
    # 2. Invalidate cache for this user
    if engine is not None:
        engine.cache.invalidate_user(submit.user_id)
        
        # Increment retraining counter
        engine.new_feedback_counter += count
        
    return {
        "status": "ok",
        "user_id": submit.user_id,
        "processed_ratings": count
    }
