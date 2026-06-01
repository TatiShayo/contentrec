# Content Recommendation Engine — Phase Plan
**Project:** contentrec
**Budget:** $0 (all free/open-source)
**Builder:** Gemini CLI
**Date:** 2026-06-01

---

## Architecture Overview

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Client Apps  │────>│  FastAPI Server   │<────│  SQLite / Disk   │
│ (POST/GET)   │     │  (port 8000)      │     │  (persistence)   │
└──────────────┘     └──────────────────┘     └──────────────────┘
                            │
                            ▼
                     ┌──────────────────┐
                     │   LightFM Model   │──── Hybrid CF + Content
                     │   (WARP loss)      │
                     └──────────────────┘
                            │
                            ▼
                     ┌──────────────────┐
                     │  Sentence-BERT    │──── Content Embeddings
                     │  (free, local)     │     (CPU, no GPU)
                     └──────────────────┘
                            │
                            ▼
                     ┌──────────────────┐
                     │   FAISS Index     │──── ANN Search
                     │  (Meta, open)      │     (CPU version)
                     └──────────────────┘
```

## No LLM Layer (Reasons)
- $0 budget — OpenRouter costs $, even cheap models add up
- Engine works without it — LLM was additive (explanations, reranking)
- Can add later when credits exist

## Phase Breakdown

### Phase 1: Core Hybrid Engine
**Files created:** ~10
**Gemini sessions:** 2 (PROMPT_SESSION.md → build, then test/debug)

**What builds:**
```
contentrec/
├── main.py                  # FastAPI app entry
├── models/
│   ├── __init__.py
│   └── engine.py            # LightFM wrapper (train, predict, recommend)
├── data/
│   ├── __init__.py
│   ├── feedback.py          # Feedback ingestion + storage (JSON/SQLite)
│   ├── items.py             # Item metadata storage
│   └── users.py             # User management (optional)
├── api/
│   ├── __init__.py
│   ├── feedback.py          # POST /feedback
│   ├── recommend.py         # GET /recommend/{user_id}
│   └── items.py             # POST /items, GET /items/{id}
├── cold_start.py            # Content-based fallback when user is new
├── config.py                # Configuration
├── requirements.txt
└── README.md
```

**Algorithms inside:**
- LightFM with WARP loss (hybrid CF + content features)
- Item metadata features → LightFM's `item_features` matrix
- Cold-start: content similarity (cosine on item features) when user has <3 interactions
- Implicit feedback only (no explicit ratings needed)

**Storage:**
- SQLite (Python built-in, no install needed) for feedback + items
- or simple JSON files for minimal setup

**Training:**
- Trigger: `POST /train` or automatic after N new feedbacks
- Incremental: LightFM supports `partial_fit` for online learning
- Full retrain on schedule

**API endpoints:**
```
POST /feedback          {user_id, item_id, event_type, timestamp}
POST /items             {item_id, metadata: {title, tags, category, ...}}
GET  /recommend/{user_id}?n=10    → [item_id, score]
GET  /similar/{item_id}?n=5       → [item_id, score]
POST /train                        → {"status": "training", "duration": "2s"}
GET  /stats                        → {"users": N, "items": M, "feedbacks": K}
```

**Dependencies (all free, all CPU-compatible):**
- fastapi, uvicorn (web server)
- lightfm (hybrid CF + content)
- numpy, scipy (numerical)
- SQLite3 (Python stdlib)

---

### Phase 2: Content Embeddings + FAISS
**Files added:** ~5

**What builds:**
```
├── embeddings/
│   ├── __init__.py
│   ├── text.py              # Sentence-BERT embeddings (all-MiniLM-L6-v2)
│   └── multimodal.py        # CLIP embeddings (optional, for images)
├── search/
│   ├── __init__.py
│   └── faiss_index.py       # FAISS ANN index
├── api/search.py            # POST /search?q=... (semantic search)
```

**What changes:**
- `cold_start.py` updated: content similarity now uses real embeddings, not metadata only
- `engine.py` updated: candidate generation fetches from both LightFM AND FAISS
- `GET /recommend` returns merged results from hybrid approach

**Dependencies added:**
- sentence-transformers (free, Apache 2.0, CPU)
- faiss-cpu (free, MIT license, CPU)
- torch (free, runs CPU mode)

**Important:** sentence-transformers `all-MiniLM-L6-v2` is 80MB download, runs on CPU in ~50ms per text. No GPU needed. FAISS CPU version handles up to 1M vectors easily.

**How it merges:**
```
def get_recommendations(user_id, n=10):
    cf_candidates = lightfm_recommend(user_id, n=20)     # from LightFM
    content_candidates = faiss_search(user_embedding, n=20)  # from embeddings
    # Merge: interleave or weighted blend
    return merge(cf_candidates, content_candidates, n=n)
```

---

### Phase 3: Sequential Model (SASRec)
**Files added:** ~6

**What builds:**
```
├── models/
│   ├── sasrec.py            # SASRec implementation (PyTorch)
│   └── sequential_train.py  # Training loop for sequential model
├── sessions/
│   ├── __init__.py
│   └── session_builder.py   # Build session sequences from feedback
├── api/sequential.py        # Sequential recommendation endpoint
```

**What changes:**
- `get_recommendations` gets a third source: sequential predictions
- Session tracking: user's last N interactions as sequence
- Adaptive blending weights shift based on user history density

**Dependencies added:**
- torch (already from Phase 2)
- No new dependencies

**Note:** SASRec is ~200 lines of PyTorch, trains in minutes on CPU for MovieLens-scale data. Gemini CLI can write this if given a clear spec.

---

### Phase 4: Production Touches
**Files added:** ~4

```
├── Dockerfile               # Containerize everything
├── docker-compose.yml       # Optional orchestration
├── test_api.py              # Integration tests
└── seed_data.py             # Demo dataset (MovieLens small)
```

---

## Gemini CLI Instructions (for each phase)

### Phase 1 PROMPT_SESSION.md strategy:

The build prompt should be structured as **multiple Gemini sessions** (2-3), each with a focused scope. Gemini works better with clear, bounded tasks than one giant prompt.

**Session 1: Scaffold + Data Layer**
Prompt Gemini to create: main.py structure, data layer (SQLite), item/feedback storage, config

**Session 2: Engine + API**
Prompt Gemini to create: LightFM engine wrapper, API endpoints, cold-start fallback

**Session 3: Test + Verify**
Prompt Gemini to: run the server, test with curl/requests, fix bugs

---

## File: Phase 1 PROMPT_SESSION.md

```
You are a Python backend engineer. Build a content recommendation engine server.

## Rules
- Use only free, open-source libraries
- Do NOT use any paid API (OpenAI, OpenRouter, etc.)
- All embeddings and models run locally on CPU
- Keep dependencies minimal

## Tech Stack
- FastAPI + Uvicorn for the web server
- LightFM for the recommendation model (pip install lightfm)
- SQLite3 (Python standard library) for storage
- NumPy, SciPy for numerical operations

## What to build

### Storage Layer (data/)
1. `data/feedback.py` — Store user-item interactions in SQLite
   - Table: feedback (id, user_id, item_id, event_type, timestamp)
   - Methods: add_feedback(), get_user_feedback(user_id, limit=N), get_all_feedback()
   - Events: "view", "like", "purchase", "watch", "click" — configurable

2. `data/items.py` — Store item metadata in SQLite
   - Table: items (id, item_id, title, tags, category, metadata_json)
   - Methods: add_item(), get_item(item_id), get_all_items(), search_by_tags()

3. `data/database.py` — Initialize SQLite, create tables, connection management

### Model Layer (models/)
4. `models/engine.py` — LightFM recommendation engine
   - Implements LightFM with WARP loss (loss='warp')
   - Supports item_features matrix for content-based cold-start
   - Methods:
     - train() — Build user-item matrix, fit LightFM model
     - recommend(user_id, n=10) — Return top-N item IDs with scores
     - similar_items(item_id, n=5) — Return similar items via embedding similarity
     - get_user_embedding(user_id) — Get user latent vector
     - cold_start_recommend(features_or_tags, n=10) — Content-based recs for new users
   - Auto-detects cold users (<3 interactions) and falls back to content-based

### API Layer (api/)
5. `api/feedback.py` — POST /feedback endpoint
   - Accepts JSON: {user_id, item_id, event_type, timestamp}
   - Stores to SQLite
   - Optionally triggers partial_fit on the model

6. `api/items.py` — POST /items and GET /items/{item_id} endpoints
   - Accepts: {item_id, title, tags, category}
   - Stores metadata for content-based features

7. `api/recommend.py` — GET /recommend/{user_id}?n=10
   - Returns: {"user_id": "...", "recommendations": [{"item_id": "...", "score": 0.85}, ...]}
   - Supports optional ?features=tag1,tag2 for cold-start users (no user_id history)

### Entry Point
8. `main.py` — FastAPI application
   - Import and mount all API routes
   - Startup event: load data, initialize LightFM, train
   - POST /train endpoint for manual retraining
   - GET /stats endpoint for system status
   - CORS middleware enabled

9. `config.py` — Configuration constants
   - DATABASE_PATH
   - MODEL_PATH
   - COLD_START_THRESHOLD (default: 3 interactions)
   - DEFAULT_N_RECOMMENDATIONS

10. `requirements.txt` — Dependencies (fastapi, uvicorn, lightfm, numpy, scipy, pydantic)

## Testing
After building, verify:
1. `uvicorn main:app --reload` starts without errors
2. POST /items with sample items returns 200
3. POST /feedback with some interactions returns 200
4. GET /recommend/test_user returns a list of item IDs
5. Train endpoint returns status
```

---

## Timeline (All Night)

| Block | What | Time |
|-------|------|------|
| **Block 1** | Phase 1: Scaffold + Data Layer (Gemini Session 1) | 30 min |
| **Block 2** | Phase 1: Engine + API (Gemini Session 2) | 45 min |
| **Block 3** | Phase 1: Test + Debug | 30 min |
| **Block 4** | Phase 2: Sentence-BERT + FAISS (Gemini Session 3) | 45 min |
| **Block 5** | Phase 2: Integrate FAISS into recommendation pipeline | 30 min |
| **Block 6** | Phase 3: SASRec sequential model (Gemini Session 4) | 45 min |
| **Block 7** | Phase 3: Adaptive blending + integration | 30 min |
| **Block 8** | Phase 4: Docker + Tests + Seed data | 30 min |
| **Buffer** | Bug fixes, debugging | 45 min |

**Total:** ~5-6 hours of builder time. Gemini does the code, you direct.

---

## Directory Structure (Full)

```
C:\Users\TATI\Desktop\Clients\contentrec\
├── PROMPT_SESSION.md             # Phase-by-phase Gemini instructions
├── main.py                       # FastAPI server entry
├── config.py                     # Config constants
├── requirements.txt              # Python dependencies
├── data/
│   ├── __init__.py
│   ├── database.py               # SQLite connection + table creation
│   ├── feedback.py               # Feedback storage
│   └── items.py                  # Item metadata storage
├── models/
│   ├── __init__.py
│   ├── engine.py                 # LightFM wrapper
│   └── sasrec.py                 # SASRec sequential (Phase 3)
├── embeddings/
│   ├── __init__.py
│   ├── text.py                   # Sentence-BERT embeddings (Phase 2)
│   └── multimodal.py             # CLIP embeddings (Phase 2, optional)
├── search/
│   ├── __init__.py
│   └── faiss_index.py            # FAISS ANN index (Phase 2)
├── api/
│   ├── __init__.py
│   ├── feedback.py               # POST /feedback
│   ├── items.py                  # POST/GET /items
│   ├── recommend.py              # GET /recommend/{user_id}
│   └── sequential.py             # Sequential recs (Phase 3)
├── cold_start.py                 # Cold-start fallback logic
├── test_api.py                   # Test script
├── seed_data.py                  # Demo dataset
├── Dockerfile                    # Container (Phase 4)
└── README.md                     # Documentation
```

## Key Design Decisions

1. **SQLite not PostgreSQL** — Zero setup, Python stdlib, portable. Good for prototype.
2. **LightFM WARP loss** — Optimizes for top-N ranking (which is what recommendations are). BPR loss is alternative but WARP converges faster.
3. **sentence-transformers over API embeddings** — Free, local, 80MB model runs on CPU. all-MiniLM-L6-v2 gives 384-dim vectors, good enough for content similarity.
4. **FAISS CPU not GPU** — IndexFlatIP (brute force inner product) works for up to 1M items on CPU in ~10ms. With IVF indexing, scales to 10M+.
5. **Cold-start via content features** — LightFM accepts item_features matrix natively. New items with metadata get embeddings immediately, no history needed.
6. **No user auth** — Users are string IDs from the consuming app. The engine is stateless with respect to auth.
7. **No LLM** — Cut for budget. Can add later by increasing cold_start_threshold or adding rules-based explanations.
8. **"feedback" not "ratings"** — Implicit signals only (view, like, purchase, watch). No 5-star system needed. LightFM handles implicit with WARP loss.
