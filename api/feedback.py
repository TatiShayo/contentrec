from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from starlette.concurrency import run_in_threadpool
from data.feedback import add_feedback

router = APIRouter()

class FeedbackCreate(BaseModel):
    user_id: str
    item_id: str
    event_type: str
    timestamp: Optional[str] = None

VALID_EVENT_TYPES = {"view", "like", "purchase", "watch", "click"}

@router.post("/feedback")
async def post_feedback(feedback: FeedbackCreate):
    if feedback.event_type not in VALID_EVENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid event_type. Must be one of {VALID_EVENT_TYPES}")
    
    feedback_id = await run_in_threadpool(
        add_feedback, feedback.user_id, feedback.item_id,
        feedback.event_type, feedback.timestamp
    )
    return {"status": "ok", "feedback_id": feedback_id}
