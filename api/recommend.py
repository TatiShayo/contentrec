from fastapi import APIRouter, Query, Request
from typing import Optional
import time
from starlette.concurrency import run_in_threadpool
import config

router = APIRouter()

@router.get("/recommend/{user_id}")
async def get_recommendations(
    user_id: str, 
    request: Request, 
    n: int = Query(config.DEFAULT_N_RECOMMENDATIONS, description="Number of recommendations to return"), 
    features: Optional[str] = Query(None, description="Cold-start context feature tags/categories to match"),
    diversity: float = Query(0.8, ge=0.0, le=1.0, description="MMR diversity parameter (0.0 for max diversity, 1.0 for max relevance)"),
    exclude_categories: Optional[str] = Query(None, description="Comma-separated categories to exclude from recommendations", json_schema_extra={"example": "music,articles"}),
    exclude_items: Optional[str] = Query(None, description="Comma-separated item IDs to exclude from recommendations", json_schema_extra={"example": "item_1,item_2"}),
    device: Optional[str] = Query(None, description="Client device type context", json_schema_extra={"example": "mobile"}),
    location: Optional[str] = Query(None, description="Client location context", json_schema_extra={"example": "london"}),
    time_of_day: Optional[str] = Query(None, description="Time of day context (morning, afternoon, evening, night)", json_schema_extra={"example": "evening"}),
    w_relevance: Optional[float] = Query(None, description="Weight for relevance score"),
    w_freshness: Optional[float] = Query(None, description="Weight for freshness boost"),
    w_fatigue: Optional[float] = Query(None, description="Weight for category fatigue penalty"),
    w_context: Optional[float] = Query(None, description="Weight for context matching"),
    query: Optional[str] = Query(None, description="Natural language search query to steer recommendations"),
    w_ssl: Optional[float] = Query(None, description="Weight for self-supervised universal representation matching")
):
    """Retrieve personalized blended recommendations for a user, supporting A/B test routing, MMR diversity, context context, and exclusions."""
    engine = request.app.state.engine
    cache_key = f"user:{user_id}:n:{n}:features:{features}:diversity:{diversity}:ex_cats:{exclude_categories}:ex_items:{exclude_items}:dev:{device}:loc:{location}:tod:{time_of_day}:wr:{w_relevance}:wf:{w_freshness}:wfat:{w_fatigue}:wc:{w_context}:q:{query}:wssl:{w_ssl}"
    
    # Check cache
    cached = engine.cache.get(cache_key)
    if cached is not None:
        engine.metrics.record_cache_hit()
        return {"user_id": user_id, "recommendations": cached, "cached": True}
        
    engine.metrics.record_cache_miss()
    
    start_time = time.time()
    recs = await run_in_threadpool(
        engine.recommend,
        user_id=user_id,
        n=n,
        features=features,
        diversity=diversity,
        exclude_categories=exclude_categories,
        exclude_items=exclude_items,
        context_device=device,
        context_location=location,
        context_time=time_of_day,
        w_relevance=w_relevance,
        w_freshness=w_freshness,
        w_fatigue=w_fatigue,
        w_context=w_context,
        query=query,
        w_ssl=w_ssl
    )
    duration = time.time() - start_time
    
    # Store in cache
    engine.cache.set(cache_key, recs)
    
    return {
        "user_id": user_id, 
        "recommendations": recs, 
        "cached": False,
        "latency_seconds": round(duration, 4)
    }

@router.get("/similar/{item_id}")
async def get_similar(
    item_id: str, 
    request: Request, 
    n: int = Query(5, description="Number of similar items to return"),
    exclude_categories: Optional[str] = Query(None, description="Comma-separated categories to exclude from similar items", json_schema_extra={"example": "music"}),
    exclude_items: Optional[str] = Query(None, description="Comma-separated item IDs to exclude from similar items", json_schema_extra={"example": "item_5"})
):
    """Find items similar to a target item, supporting content similarity (FAISS) or collaborative embedding similarity."""
    engine = request.app.state.engine
    similar = await run_in_threadpool(engine.similar_items, item_id, n, exclude_categories, exclude_items)
    return {"item_id": item_id, "similar": similar}
