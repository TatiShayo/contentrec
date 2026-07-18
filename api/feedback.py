from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
from starlette.concurrency import run_in_threadpool
from data.feedback import add_feedback
import config

router = APIRouter()

class FeedbackCreate(BaseModel):
    model_config = {"extra": "forbid"}

    user_id: str = Field(..., min_length=1, max_length=256, description="Unique user identifier", json_schema_extra={"example": "user_alice"})
    item_id: str = Field(..., min_length=1, max_length=256, description="Unique item identifier", json_schema_extra={"example": "item_12"})
    event_type: str = Field(..., max_length=32, description="Type of interaction event (view, like, purchase, watch, click)", json_schema_extra={"example": "click"})
    timestamp: Optional[str] = Field(None, max_length=64, description="ISO-8601 formatted timestamp", json_schema_extra={"example": "2026-06-11T23:22:34"})
    dwell_time: Optional[float] = Field(0.0, ge=0.0, le=86400.0, description="Time in seconds spent viewing/interacting with the item", json_schema_extra={"example": 15.4})

VALID_EVENT_TYPES = {"view", "like", "purchase", "watch", "click"}

@router.post("/feedback")
async def post_feedback(feedback: FeedbackCreate, request: Request, background_tasks: BackgroundTasks):
    """Log user interaction feedback (click, view, purchase, etc.) and check for automatic model retraining."""
    if feedback.event_type not in VALID_EVENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid event_type. Must be one of {VALID_EVENT_TYPES}")
    
    feedback_id = await run_in_threadpool(
        add_feedback, feedback.user_id, feedback.item_id,
        feedback.event_type, feedback.timestamp, feedback.dwell_time or 0.0
    )
    
    # Process conversion tracking & auto-retraining triggers
    engine = getattr(request.app.state, 'engine', None)
    if engine is not None:
        # Check and record cohort conversion
        engine.metrics.record_conversion(feedback.user_id, feedback.item_id, engine=engine)
        
        # Push training sample to online sequential trainer queue
        if engine.seq_online_trainer is not None and engine.seq_item_to_idx:
            try:
                from sessions.session_builder import get_user_sequence
                sequence = await run_in_threadpool(get_user_sequence, feedback.user_id, 50)
                if len(sequence) >= 2:
                    idx_seq = [engine.seq_item_to_idx.get(iid, 0) for iid in sequence]
                    seq_indices = idx_seq[:-1]
                    next_idx = idx_seq[-1]
                    dwell_time = feedback.dwell_time or 0.0
                    engine.seq_online_trainer.add_sample(seq_indices, next_idx, dwell_time)
            except Exception as e:
                import logging
                logging.warning(f"Error adding online training sample for user {feedback.user_id}: {e}")

        # Increment retraining counter
        engine.new_feedback_counter += 1
        
        # Trigger background training if threshold is crossed
        threshold = getattr(config, 'RETRAIN_THRESHOLD_FEEDBACK', 50)
        if engine.new_feedback_counter >= threshold:
            def train_and_sync():
                engine.train()
                request.app.state.faiss_index = engine.faiss_index
            background_tasks.add_task(train_and_sync)
            
    return {"status": "ok", "feedback_id": feedback_id}
