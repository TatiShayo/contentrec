from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from api.feedback import router as feedback_router
from api.items import router as items_router
from api.recommend import router as recommend_router
from api.search import router as search_router
from api.sequential import router as sequential_router
from api.onboarding import router as onboarding_router
from models.engine import RecommendationEngine
from data.database import init_db
from data.feedback import get_feedback_count
from data.items import get_all_items
from starlette.concurrency import run_in_threadpool
import config
from utils.logging import setup_logging, LoggingMiddleware
from utils.rate_limiter import RateLimitMiddleware

import asyncio
import logging
from contextlib import asynccontextmanager

# Initialize system-wide structured logging configuration
setup_logging()

async def check_retraining_loop(app: FastAPI):
    """Periodically checks if the model should be retrained based on the time elapsed since last training."""
    try:
        while True:
            await asyncio.sleep(60)  # Check every 60 seconds
            engine = getattr(app.state, 'engine', None)
            if engine is not None:
                import time
                elapsed = time.time() - engine.last_trained_time
                threshold_seconds = getattr(config, 'RETRAIN_INTERVAL_SECONDS', 86400)
                if elapsed >= threshold_seconds and engine.new_feedback_counter > 0:
                    logging.info("Auto-retraining triggered by time-based scheduler.")
                    try:
                        await run_in_threadpool(engine.train)
                        app.state.faiss_index = engine.faiss_index
                    except Exception as e:
                        logging.error(f"Error in time-based auto-retraining: {e}")
    except asyncio.CancelledError:
        logging.info("check_retraining_loop task cancelled.")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup events
    await run_in_threadpool(init_db)
    app.state.engine = RecommendationEngine()
    app.state.faiss_index = app.state.engine.faiss_index
    # Initial train if data exists
    await run_in_threadpool(app.state.engine.train)
    # Refresh reference after training in case it got rebuilt
    app.state.faiss_index = app.state.engine.faiss_index
    # Start time-based background scheduler check loop
    retrain_task = asyncio.create_task(check_retraining_loop(app))
    
    yield
    
    # Shutdown events: Cleanly cancel the retraining loop task
    retrain_task.cancel()
    try:
        await retrain_task
    except asyncio.CancelledError:
        logging.info("Retraining loop task successfully cancelled.")
    except Exception as e:
        logging.error(f"Error when cancelling retraining loop task: {e}")
        
    # Shutdown events: Stop/join the SASRec online training background thread
    engine = getattr(app.state, 'engine', None)
    if engine is not None:
        if getattr(engine, 'seq_online_stop_event', None) is not None:
            logging.info("Stopping SASRec online training background thread.")
            engine.seq_online_stop_event.set()
            if engine.seq_online_thread is not None:
                await run_in_threadpool(engine.seq_online_thread.join, 5.0)

app = FastAPI(title="Content Recommendation Engine", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom sliding-window rate limiting middleware (e.g., 60 requests/min per IP)
app.add_middleware(RateLimitMiddleware, requests_limit=60, window_sec=60)

# Custom performance metrics logging middleware
app.add_middleware(LoggingMiddleware)


# Mount routers
app.include_router(feedback_router, tags=["feedback"])
app.include_router(items_router, tags=["items"])
app.include_router(recommend_router, tags=["recommend"])
app.include_router(search_router, tags=["search"])
app.include_router(sequential_router, tags=["sequential"])
app.include_router(onboarding_router, tags=["onboarding"])

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    def train_and_sync():
        app.state.engine.train()
        app.state.faiss_index = app.state.engine.faiss_index
        
    background_tasks.add_task(train_and_sync)
    return {"status": "training started in background"}

@app.get("/metrics")
async def get_metrics(request: Request):
    """Diagnose service operational performance, caching efficiency, and database statistics."""
    engine = getattr(request.app.state, 'engine', None)
    if engine is None:
        return {"status": "engine not initialized"}
        
    feedback_count = await run_in_threadpool(get_feedback_count)
    items = await run_in_threadpool(get_all_items)
    
    metrics = engine.metrics.get_metrics()
    metrics.update({
        "database": {
            "item_count": len(items),
            "feedback_count": feedback_count
        },
        "cache_size": engine.cache.size()
    })
    return metrics

@app.get("/stats")
async def get_stats():
    feedback_count = await run_in_threadpool(get_feedback_count)
    items = await run_in_threadpool(get_all_items)
    item_count = len(items)

    # Get unique users from feedback
    def get_user_count():
        from data.database import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM feedback")
            return cursor.fetchone()[0]

    user_count = await run_in_threadpool(get_user_count)

    return {
        "user_count": user_count,
        "item_count": item_count,
        "feedback_count": feedback_count
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
