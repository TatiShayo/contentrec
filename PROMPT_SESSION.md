You are a Python backend engineer. Build a content recommendation engine server.

## Rules
- Use only free, open-source libraries
- Do NOT use any paid API (OpenAI, OpenRouter, etc.)
- All models run locally on CPU
- Keep dependencies minimal
- IMPORTANT: All SQLite and file operations MUST use run_in_threadpool or aiosqlite to avoid blocking the async event loop

## Tech Stack
- FastAPI + Uvicorn for the web server
- LightFM for the recommendation model
- SQLite3 with thread-safe access (run_in_threadpool or aiosqlite)
- NumPy, SciPy for numerical operations
- threading.Lock for model operations (model is not thread-safe)

## What to build

### Storage Layer

1. Create `data/database.py`:
   - Initialize SQLite database
   - Create tables: feedback (id, user_id, item_id, event_type, timestamp), items (id, item_id TEXT UNIQUE, title TEXT, tags TEXT, category TEXT, metadata_json TEXT)
   - Connection management with context manager
   - Table creation on first import

2. Create `data/feedback.py`:
   - add_feedback(user_id, item_id, event_type, timestamp=None) — insert into feedback table
   - get_user_feedback(user_id, limit=100) — return list of feedback for user
   - get_all_feedback() — return all feedback for training
   - get_feedback_count() — total feedback count

3. Create `data/items.py`:
   - add_item(item_id, title, tags="", category="", metadata=None) — insert or update item
   - get_item(item_id) — return item dict
   - get_all_items() — return all items
   - search_by_tags(query) — search items by tag substring

### Model Layer

4. Create `models/engine.py` — LightFM recommendation engine:
   - Class `RecommendationEngine` initialized with config
   - Uses `threading.Lock` around ALL train/predict/partial_fit operations (LightFM is not thread-safe)
   - train() method:
     - Build user-item interaction matrix from SQLite feedback
     - Build item_features matrix from item metadata (tags as features)
     - Fit LightFM model with loss='warp' (Weighted Approximate-Rank Pairwise)
     - Save model state to disk
   - recommend(user_id, n=10) method:
     - Get user's feedback history
     - If user has < COLD_START_THRESHOLD interactions → call cold_start_recommend()
     - Else: use LightFM's predict() on all items, return top-N
     - Return list of {item_id, score} dicts
   - similar_items(item_id, n=5) method:
     - Get item embedding from LightFM
     - Find nearest items by cosine similarity
     - Return list of {item_id, score}
   - cold_start_recommend(tags_or_category=None, n=10) method:
     - Use content-based: find items whose tags match the provided tags
     - Fallback: return most popular items (most feedback)
   - partial_fit() method:
     - Incremental update without full retrain (LightFM supports this)

### API Layer

5. Create `api/feedback.py`:
   - POST /feedback — accepts {"user_id": str, "item_id": str, "event_type": str, "timestamp": str (optional)}
   - Returns {"status": "ok", "feedback_id": int}
   - Validates that event_type is one of: view, like, purchase, watch, click

6. Create `api/items.py`:
   - POST /items — accepts {"item_id": str, "title": str, "tags": str (comma-separated), "category": str}
   - GET /items/{item_id} — returns item metadata
   - GET /items — returns all items with pagination (query params: offset, limit)

7. Create `api/recommend.py`:
   - GET /recommend/{user_id} — returns {"user_id": str, "recommendations": [{item_id, score}, ...]}
   - Query params: n (default 10), features (optional, comma-separated tags for cold-start)
   - GET /similar/{item_id} — returns {"item_id": str, "similar": [{item_id, score}, ...]}
   - Query params: n (default 5)

### Entry Point

8. Create `main.py`:
   - FastAPI application with title="Content Recommendation Engine"
   - Import and mount all API routers
   - On startup: init database, create engine, train model
   - POST /train endpoint: uses FastAPI BackgroundTasks to run training in background (non-blocking)
   - GET /stats endpoint returning counts of users, items, feedback
   - GET /health endpoint returning {"status": "ok"}
   - CORS middleware (allow all origins for dev)
   - Run with: uvicorn main:app --host 0.0.0.0 --port 8000

9. Create `config.py`:
   - DATABASE_PATH = "data/recommender.db"
   - MODEL_PATH = "data/model.pkl"
   - COLD_START_THRESHOLD = 3
   - DEFAULT_N_RECOMMENDATIONS = 10

10. Create `requirements.txt`:
    ```
    fastapi
    uvicorn
    lightfm
    numpy
    scipy
    pydantic
    ```

## Testing
Create a `tests/` directory with pytest test files.

11. Create `tests/test_data.py`:
    - Test add_item, get_item, search_by_tags
    - Test add_feedback, get_user_feedback, get_all_feedback
    - Test empty database behavior
    - Test duplicate item insert

12. Create `tests/test_engine.py`:
    - Test RecommendationEngine initialization
    - Test train() with sample data
    - Test recommend() returns valid results
    - Test cold_start_recommend() for new user (no interactions)
    - Test similar_items() returns items
    - Test partial_fit() incremental update

13. Create `tests/test_api.py`:
    - Use FastAPI TestClient (no server needed)
    - Test all endpoints: POST /items, POST /feedback, GET /recommend/{id}, GET /similar/{id}, POST /train, GET /stats, GET /health
    - Test validation (missing fields, bad types)
    - Test cold-start user flow
    - Test warm user flow (with interactions)

14. Create `tests/test_integration.py`:
    - Full pipeline: seed items → add feedback → train → get recommendations → verify
    - Cold start: new user with no history → gets content-based recs
    - Warm user: user with 10+ interactions → gets personalized recs
    - Stats endpoint returns correct counts

15. Create `tests/conftest.py`:
    - Test fixtures for app, database, sample items, sample feedback

16. Update `requirements.txt` to add pytest:
    ```
    fastapi
    uvicorn
    lightfm
    numpy
    scipy
    pydantic
    pytest
    httpx
    ```

## Manual Verification
After building and running tests, verify:
1. `pytest tests/ -v` passes all tests
2. uvicorn main:app --reload starts without errors
3. POST /items with sample items via curl returns 200
4. POST /feedback with interactions returns 200
5. GET /recommend/test_user returns a list of item IDs
6. GET /similar/some_item returns similar items
7. POST /train returns training status
8. GET /stats returns correct counts
9. Cold start: GET /recommend/new_user (no history) returns content-based recs
