from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from api.feedback import router as feedback_router
from api.items import router as items_router
from api.recommend import router as recommend_router
from models.engine import RecommendationEngine
from data.database import init_db
from data.feedback import get_feedback_count
from data.items import get_all_items
from starlette.concurrency import run_in_threadpool
import sqlite3
import config

app = FastAPI(title="Content Recommendation Engine")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(feedback_router, tags=["feedback"])
app.include_router(items_router, tags=["items"])
app.include_router(recommend_router, tags=["recommend"])

@app.on_event("startup")
async def startup_event():
    await run_in_threadpool(init_db)
    app.state.engine = RecommendationEngine()
    # Initial train if data exists
    await run_in_threadpool(app.state.engine.train)

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(app.state.engine.train)
    return {"status": "training started in background"}

@app.get("/stats")
async def get_stats():
    feedback_count = get_feedback_count()
    items = get_all_items()
    item_count = len(items)
    
    # Get unique users from feedback
    def get_user_count():
        import sqlite3
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
