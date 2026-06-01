from fastapi import APIRouter, Query, Request
from typing import Optional
from starlette.concurrency import run_in_threadpool
import config

router = APIRouter()

@router.get("/recommend/{user_id}")
async def get_recommendations(user_id: str, request: Request, n: int = config.DEFAULT_N_RECOMMENDATIONS, features: Optional[str] = None):
    engine = request.app.state.engine
    recs = await run_in_threadpool(engine.recommend, user_id, n, features)
    return {"user_id": user_id, "recommendations": recs}

@router.get("/similar/{item_id}")
async def get_similar(item_id: str, request: Request, n: int = 5):
    engine = request.app.state.engine
    similar = await run_in_threadpool(engine.similar_items, item_id, n)
    return {"item_id": item_id, "similar": similar}
