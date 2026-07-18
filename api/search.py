from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import time
from typing import Optional
import config

router = APIRouter(tags=["search"])

class SearchRequest(BaseModel):
    model_config = {"extra": "forbid"}

    query: str = Field(..., max_length=1024, description="Semantic text query to search for", json_schema_extra={"example": "machine learning"})
    n: int = Field(10, ge=1, le=config.MAX_N_RECOMMENDATIONS, description="Number of results to return", json_schema_extra={"example": 5})

@router.post("/search")
async def search_post(req: SearchRequest, request: Request):
    """Semantic search via POST body."""
    engine = getattr(request.app.state, 'engine', None)
    faiss_index = getattr(request.app.state, 'faiss_index', None)
    if faiss_index is None or faiss_index.index is None or faiss_index.index.ntotal == 0:
        raise HTTPException(status_code=503, detail="Search index not built. POST /train first.")

    if engine is None:
        results = await run_in_threadpool(faiss_index.search_by_text, req.query, req.n)
        return {"query": req.query, "results": results}

    cache_key = f"search:query:{req.query}:n:{req.n}"
    
    # Check cache
    cached = engine.cache.get(cache_key)
    if cached is not None:
        engine.metrics.record_cache_hit()
        return {"query": req.query, "results": cached, "cached": True}
        
    engine.metrics.record_cache_miss()
    
    start_time = time.time()
    results = await run_in_threadpool(faiss_index.search_by_text, req.query, req.n)
    duration = time.time() - start_time
    
    # Record search latency
    engine.metrics.record_latency("search", duration)
    
    # Store in cache
    engine.cache.set(cache_key, results)
    
    return {
        "query": req.query, 
        "results": results, 
        "cached": False,
        "latency_seconds": round(duration, 4)
    }

@router.get("/search")
async def search_get(
    request: Request,
    q: str = Query(..., max_length=1024),
    n: int = Query(10, ge=1, le=config.MAX_N_RECOMMENDATIONS),
):
    """Semantic search via query params."""
    engine = getattr(request.app.state, 'engine', None)
    faiss_index = getattr(request.app.state, 'faiss_index', None)
    if faiss_index is None or faiss_index.index is None or faiss_index.index.ntotal == 0:
        raise HTTPException(status_code=503, detail="Search index not built. POST /train first.")

    if engine is None:
        results = await run_in_threadpool(faiss_index.search_by_text, q, n)
        return {"query": q, "results": results}

    cache_key = f"search:query:{q}:n:{n}"
    
    # Check cache
    cached = engine.cache.get(cache_key)
    if cached is not None:
        engine.metrics.record_cache_hit()
        return {"query": q, "results": cached, "cached": True}
        
    engine.metrics.record_cache_miss()
    
    start_time = time.time()
    results = await run_in_threadpool(faiss_index.search_by_text, q, n)
    duration = time.time() - start_time
    
    # Record search latency
    engine.metrics.record_latency("search", duration)
    
    # Store in cache
    engine.cache.set(cache_key, results)
    
    return {
        "query": q, 
        "results": results, 
        "cached": False,
        "latency_seconds": round(duration, 4)
    }
