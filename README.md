# 🎯 Content Recommendation Engine

A **local-first, CPU-friendly content recommendation engine** built with FastAPI and Python. It combines collaborative filtering (LightFM) with content-based fallbacks to deliver personalised recommendations — all running on your own machine, no paid APIs required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI App                        │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │  /items   │  │/feedback │  │  /recommend/{uid}  │    │
│  │  (CRUD)   │  │ (ingest) │  │  /similar/{iid}    │    │
│  └────┬─────┘  └────┬─────┘  └─────────┬──────────┘    │
│       │              │                  │               │
│       ▼              ▼                  ▼               │
│  ┌─────────────────────────────────────────────────┐    │
│  │              SQLite Database                     │    │
│  │   items (id, title, tags, category, metadata)    │    │
│  │   feedback (user_id, item_id, event_type, ts)    │    │
│  └──────────────────┬──────────────────────────────┘    │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐    │
│  │          Recommendation Engine                   │    │
│  │                                                  │    │
│  │  ┌──────────────────┐  ┌─────────────────────┐  │    │
│  │  │ LightFM (collab) │  │ Content-based / Pop │  │    │
│  │  │ (Linux/Docker)   │  │ (always available)  │  │    │
│  │  └──────────────────┘  └─────────────────────┘  │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

> **Note:** LightFM is automatically disabled on Windows due to a known Cython crash. The engine falls back to content-based + popularity recommendations seamlessly.

---

## Features

- **Hybrid recommendations** — collaborative filtering via LightFM + content-based tag matching + popularity fallback
- **Cold-start handling** — new users get tag-based or popularity-based recommendations until enough feedback accumulates
- **Real-time feedback ingestion** — `view`, `like`, `purchase`, `watch`, `click` events
- **Similar items** — find items similar to a given item via embedding cosine similarity
- **Background training** — trigger model retraining without blocking the API
- **Zero external services** — SQLite for storage, all models run locally on CPU
- **Docker-ready** — multi-stage Dockerfile + Docker Compose included
- **Demo seeder** — `seed_data.py` populates 54 items and 129 feedback events

---

## Quick Start

### 1. Clone & set up a virtual environment

```bash
git clone <repo-url> contentrec
cd contentrec

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> On Windows, LightFM may fail to install (requires C compiler). This is fine — the engine works without it.

### 3. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Seed demo data

With the server running in another terminal:

```bash
python seed_data.py
```

Or, without the server:

```bash
python seed_data.py --direct
```

### 5. Try it out

```bash
# Get recommendations for a user
curl http://localhost:8000/recommend/user_alice

# Get similar items
curl http://localhost:8000/similar/movie_002

# Check system stats
curl http://localhost:8000/stats
```

---

## Docker

### Build & run with Docker Compose

```bash
docker compose up --build -d
```

The API will be available at `http://localhost:8000`.

### Seed data in the container

```bash
docker compose exec app python seed_data.py --direct
```

### Stop

```bash
docker compose down
```

### Build image only

```bash
docker build -t contentrec .
docker run -p 8000:8000 -v ./data:/app/data contentrec
```

---

## API Reference

### Health Check

```
GET /health
```

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

### System Stats

```
GET /stats
```

```bash
curl http://localhost:8000/stats
# → {"user_count": 10, "item_count": 54, "feedback_count": 129}
```

---

### Items

#### Create / Update an Item

```
POST /items
Content-Type: application/json
```

```bash
curl -X POST http://localhost:8000/items \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "movie_999",
    "title": "My Movie",
    "tags": "action,thriller",
    "category": "movies"
  }'
# → {"status": "ok"}
```

#### Get a Single Item

```
GET /items/{item_id}
```

```bash
curl http://localhost:8000/items/movie_001
# → {"id": 1, "item_id": "movie_001", "title": "The Shawshank Redemption", ...}
```

#### List All Items

```
GET /items?offset=0&limit=100
```

```bash
curl "http://localhost:8000/items?limit=10"
```

---

### Feedback

#### Submit Feedback

```
POST /feedback
Content-Type: application/json
```

Valid `event_type` values: `view`, `like`, `purchase`, `watch`, `click`

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_alice",
    "item_id": "movie_002",
    "event_type": "like"
  }'
# → {"status": "ok", "feedback_id": 42}
```

---

### Recommendations

#### Get Recommendations for a User

```
GET /recommend/{user_id}?n=10&features=sci-fi,action
```

| Parameter | Type   | Default | Description                          |
|-----------|--------|---------|--------------------------------------|
| `n`       | int    | 10      | Number of recommendations to return  |
| `features`| string | null    | Comma-separated tags for cold start  |

```bash
curl "http://localhost:8000/recommend/user_alice?n=5"
# → {"user_id": "user_alice", "recommendations": [{"item_id": "movie_008", "score": 3.14}, ...]}
```

#### Get Similar Items

```
GET /similar/{item_id}?n=5
```

```bash
curl "http://localhost:8000/similar/movie_002?n=3"
# → {"item_id": "movie_002", "similar": [{"item_id": "movie_008", "score": 0.92}, ...]}
```

---

### Training

#### Trigger Model Training

```
POST /train
```

```bash
curl -X POST http://localhost:8000/train
# → {"status": "training started in background"}
```

---

## Configuration

`config.py` reads every setting from environment variables (with safe defaults),
so secrets and deployment specifics never need to be hard-coded:

| Variable                     | Default                | Description                                    |
|------------------------------|------------------------|------------------------------------------------|
| `DATABASE_PATH`              | `data/recommender.db`  | Path to the SQLite database file               |
| `MODEL_PATH`                 | `data/model.pkl`       | Path to save/load the trained model            |
| `COLD_START_THRESHOLD`       | `3`                    | Min feedback events before collaborative recs  |
| `DEFAULT_N_RECOMMENDATIONS`  | `10`                   | Default number of recommendations returned     |
| `MAX_N_RECOMMENDATIONS`      | `100`                  | Hard cap on `n` per request (DoS guard)        |
| `MAX_PAGE_LIMIT`             | `500`                  | Hard cap on list-endpoint pagination           |
| `API_KEY`                    | *(unset)*              | If set, all non-public endpoints require `X-API-Key` |
| `CORS_ALLOW_ORIGINS`         | *(empty)*              | Comma-separated allowed origins (wildcard disables credentials) |

## Security

- **Authentication** is opt-in: set `API_KEY` to require an `X-API-Key` header on
  every write/recommend endpoint (`/health`, `/docs`, `/openapi.json` stay
  public). Unset by default for frictionless local use. Note there is still no
  *per-user* identity — `user_id` is caller-supplied (see `REVIEW_FINDINGS.md`).
- **CORS** never combines a wildcard origin with credentials.
- **Rate limiting** is per client IP (sliding window, proxy-header aware).
- **Input validation**: all request bodies reject unknown fields and enforce
  length/range bounds; `n` and pagination are capped.
- Unhandled errors return a generic `500` — no stack traces are leaked.
- Run `python -m pip_audit` to check dependency advisories.

See `ARCHITECTURE.md`, `REVIEW_FINDINGS.md`, and `AUDIT_LOG.md` for the full
security posture.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_api.py -v

# Run with coverage (install pytest-cov first)
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Project Structure

```
contentrec/
├── main.py                 # FastAPI application entry point
├── config.py               # Configuration constants
├── seed_data.py            # Demo data seeder script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker Compose orchestration
├── .dockerignore           # Docker build context exclusions
│
├── api/                    # API route handlers
│   ├── __init__.py
│   ├── items.py            # Item CRUD endpoints
│   ├── feedback.py         # Feedback ingestion endpoint
│   └── recommend.py        # Recommendation & similarity endpoints
│
├── data/                   # Data layer (SQLite)
│   ├── __init__.py
│   ├── database.py         # DB connection & schema init
│   ├── items.py            # Item data operations
│   ├── feedback.py         # Feedback data operations
│   └── recommender.db      # SQLite database (auto-created)
│
├── models/                 # ML engine
│   ├── __init__.py
│   └── engine.py           # Recommendation engine (LightFM + fallbacks)
│
└── tests/                  # Test suite
    ├── __init__.py
    ├── conftest.py          # Shared test fixtures
    ├── test_api.py          # API endpoint tests
    ├── test_data.py         # Data layer tests
    ├── test_engine.py       # Engine unit tests
    └── test_integration.py  # Integration tests
```

---

## Technology Stack

| Component        | Technology                              |
|------------------|-----------------------------------------|
| Web Framework    | FastAPI + Uvicorn                       |
| ML Engine        | LightFM (collaborative filtering)      |
| Fallback Engine  | Content-based (tag matching) + popularity |
| Database         | SQLite (via stdlib `sqlite3`)           |
| Numerics         | NumPy, SciPy                            |
| Validation       | Pydantic                                |
| Testing          | pytest, httpx                           |
| Containerisation | Docker, Docker Compose                  |

---

## Phase Roadmap

| Phase | Focus                            | Status      |
|-------|----------------------------------|-------------|
| 0     | Research & design                | ✅ Complete |
| 1     | Core data layer (SQLite, models) | ✅ Complete |
| 2     | API layer (FastAPI endpoints)    | ✅ Complete |
| 3     | Test suite                       | ✅ Complete |
| 4     | Production touches (Docker, docs, seed data) | ✅ Complete |
| 5     | Advanced features (FAISS, A/B testing, caching) | 🔜 Planned |

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
