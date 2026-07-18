"""Sequential recommendation API endpoints."""
from fastapi import APIRouter, Request, Query
from typing import Optional
from starlette.concurrency import run_in_threadpool
import time
import config

router = APIRouter(tags=["sequential"])


def _get_sequential_recs(request, user_id, n, exclude_categories=None, exclude_items=None,
                         device=None, location=None, time_of_day=None):
    """Get sequential recommendations for a user (blocking), applying exclusions and context re-ranking."""
    engine = request.app.state.engine
    seq_model = getattr(engine, 'seq_model', None)
    seq_item_to_idx = getattr(engine, 'seq_item_to_idx', None)
    seq_idx_to_item = getattr(engine, 'seq_idx_to_item', None)

    if seq_model is None or seq_item_to_idx is None:
        return []

    try:
        from sessions.session_builder import get_user_sequence
        sequence = get_user_sequence(user_id, max_len=50)

        if not sequence:
            return []

        # Convert to indices
        idx_seq = [seq_item_to_idx.get(iid, 0) for iid in sequence]
        idx_seq = [i for i in idx_seq if i != 0]  # remove unknowns

        if not idx_seq:
            return []

        # Request more items to account for exclusions
        top_indices = seq_model.predict_next(idx_seq, n=n + 50)

        # Normalize exclusions
        ex_items = set()
        if exclude_items:
            if isinstance(exclude_items, str):
                ex_items = {i.strip() for i in exclude_items.split(",") if i.strip()}
            else:
                ex_items = set(exclude_items)

        ex_categories = set()
        if exclude_categories:
            if isinstance(exclude_categories, str):
                ex_categories = {c.strip().lower() for c in exclude_categories.split(",") if c.strip()}
            else:
                ex_categories = {c.lower() for c in exclude_categories}

        candidate_ids = []
        for idx in top_indices:
            item_id = seq_idx_to_item.get(idx, None)
            if item_id:
                candidate_ids.append(item_id)
        
        from data.items import get_items_by_ids
        item_details = get_items_by_ids(candidate_ids) if candidate_ids else {}

        results = []
        rank = 0
        for idx in top_indices:
            item_id = seq_idx_to_item.get(idx, None)
            if not item_id:
                continue

            if item_id in ex_items:
                continue

            if ex_categories:
                details = item_details.get(item_id)
                if details:
                    cat = (details.get("category") or "").lower()
                    if cat in ex_categories:
                        continue

            results.append({
                "item_id": item_id,
                "score": round(1.0 - rank * 0.05, 3),
                "source": "sequential",
                "explanation": "Based on your recent interaction sequence."
            })
            rank += 1

        # Multi-Objective re-ranking for sequential results too!
        from models.engine import get_freshness_score, get_context_match_score
        re_ranked = []
        for r in results:
            item_id = r["item_id"]
            details = item_details.get(item_id) or {}
            
            relevance = r["score"]
            freshness = get_freshness_score(details)
            context_match = get_context_match_score(details, device, location, time_of_day)
            
            final_score = relevance + (freshness * 0.2) + (context_match * 0.4)
            
            exp = r["explanation"]
            if device or time_of_day or location:
                reasons = []
                if device and context_match > 0:
                    reasons.append(f"optimized for {device}")
                if time_of_day and context_match > 0:
                    reasons.append(f"suited for your {time_of_day} session")
                if reasons:
                    exp = f"Sequential recommendation, " + " and ".join(reasons) + "."
            
            r_copy = r.copy()
            r_copy["score"] = round(final_score, 4)
            r_copy["explanation"] = exp
            re_ranked.append(r_copy)
            
        return sorted(re_ranked, key=lambda x: x["score"], reverse=True)[:n]
    except Exception as e:
        print(f"Sequential recommendation error: {e}")
        return []


@router.get("/sequential/{user_id}")
async def get_sequential_recommendations(
    user_id: str, 
    request: Request, 
    n: int = Query(10, ge=1, le=config.MAX_N_RECOMMENDATIONS, description="Number of recommendations to return"),
    exclude_categories: Optional[str] = Query(None, description="Comma-separated categories to exclude from recommendations", json_schema_extra={"example": "music"}),
    exclude_items: Optional[str] = Query(None, description="Comma-separated item IDs to exclude from recommendations", json_schema_extra={"example": "item_1"}),
    device: Optional[str] = Query(None, description="Client device type context", json_schema_extra={"example": "mobile"}),
    location: Optional[str] = Query(None, description="Client location context", json_schema_extra={"example": "london"}),
    time_of_day: Optional[str] = Query(None, description="Time of day context (morning, afternoon, evening, night)", json_schema_extra={"example": "evening"})
):
    """Get next-item recommendations based on user's interaction sequence, supporting exclusions and context-aware scoring."""
    engine = request.app.state.engine
    cache_key = f"sequential:user:{user_id}:n:{n}:ex_cats:{exclude_categories}:ex_items:{exclude_items}:dev:{device}:loc:{location}:tod:{time_of_day}"
    
    # Check cache
    cached = engine.cache.get(cache_key)
    if cached is not None:
        engine.metrics.record_cache_hit()
        return {"user_id": user_id, "recommendations": cached, "cached": True}
        
    engine.metrics.record_cache_miss()
    
    start_time = time.time()
    results = await run_in_threadpool(
        _get_sequential_recs, request, user_id, n,
        exclude_categories, exclude_items,
        device, location, time_of_day
    )
    duration = time.time() - start_time
    
    # Record sequential latency
    engine.metrics.record_latency("sequential", duration)
    
    # Store in cache
    engine.cache.set(cache_key, results)
    
    return {
        "user_id": user_id, 
        "recommendations": results, 
        "cached": False,
        "latency_seconds": round(duration, 4)
    }
