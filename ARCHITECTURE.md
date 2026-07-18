# ARCHITECTURE — ContentRec

A local-first, CPU-friendly content recommendation service built on **FastAPI**.
It layers a hybrid recommendation engine (collaborative, sequential, graph,
content, and RL signals) over a single-file **SQLite** store, with **FAISS** for
semantic retrieval. No external network services are required — everything runs
in-process.

## 1. Request / process topology

```
                      ┌──────────────────────── FastAPI (main.py) ────────────────────────┐
   HTTP client  ─────▶│ Middleware chain (outer→inner):                                    │
                      │   RateLimitMiddleware ─▶ APIKeyMiddleware ─▶ LoggingMiddleware ─▶ CORS
                      │ Exception handler: generic 500 (no traceback leak)                 │
                      │                                                                    │
                      │  Routers:  /items  /feedback  /recommend /similar                  │
                      │            /search  /sequential  /onboarding                       │
                      │  App-level: /train /metrics /stats /health                         │
                      └───────────────┬───────────────────────────────┬───────────────────┘
                                      │                                │
                         app.state.engine (RecommendationEngine)   app.state.faiss_index
                                      │
        ┌─────────────────────────────┼──────────────────────────────────────────┐
        ▼                             ▼                            ▼               ▼
   SQLite (data/*.db)         FAISS HNSW index            Torch models       In-mem cache
   items / feedback /         (semantic retrieval)        SASRec / LightGCN   (TTL + maxsize)
   impressions / bandit                                   BCQ / BEST-Rec
```

Blocking work (DB, training, embedding, FAISS) is dispatched off the event loop
via `starlette.concurrency.run_in_threadpool`. A background `asyncio` task
(`check_retraining_loop`) plus a daemon thread (SASRec online trainer) run for
the app lifetime and are cancelled/joined in the `lifespan` shutdown.

## 2. Modules

| Layer | Path | Responsibility |
|-------|------|----------------|
| Entry / wiring | `main.py` | App factory, middleware, lifespan, `/train`, `/metrics`, `/stats`, `/health` |
| Config | `config.py` | Env-driven settings, paths, caps, secrets |
| API | `api/*.py` | Request models (pydantic) + routing per resource |
| Engine | `models/engine.py` | `RecommendationEngine`: load/train/blend/rerank orchestration |
| Models | `models/{sasrec,lightgcn,bcq,best_rec,causal}.py` | Torch/graph/RL model definitions + trainers |
| Retrieval | `search/faiss_index.py` | HNSW inner-product index, persistence |
| Embeddings | `embeddings/{text,vision}.py` | SBERT text + image embedding (late fusion) |
| Ranking utils | `utils/{diversity,dpp,fairness,surprise,bandit,explain,ab_test}.py` | MMR, DPP onboarding, fairness PID, bandit, explainability |
| Cross-cutting | `utils/{cache,rate_limiter,auth,logging,metrics}.py` | TTL cache, rate limiting, API-key auth, structured logging, metrics |
| Data | `data/{database,items,feedback}.py` | SQLite schema + parameterized queries |
| Sessions | `sessions/session_builder.py` | User interaction sequence construction |

## 3. Data model (SQLite)

- **items** `(id, item_id UNIQUE, title, tags, category, metadata_json, image_embedding)`
- **feedback** `(id, user_id, item_id, event_type, timestamp, dwell_time)`
- **impressions** `(id, user_id, item_id, cohort, context_json, timestamp)` — causal propensity logging
- **bandit_states** `(arm_id PK, state_json)` — neural-linear bandit persistence

All queries are parameterized (`?` placeholders) — no string interpolation into
SQL (confirmed via fuzz test). Connections are per-operation via a
`get_db_connection()` context manager.

## 4. Recommendation data flow (`GET /recommend/{user_id}`)

1. **Cache** lookup (`RecommendationCache`, TTL 300s, `maxsize=10000` w/ eviction).
2. **Context vector** built for the neural-linear bandit → arm/weights selected
   (unless caller overrides weights).
3. **A/B cohort** routing (deterministic hash of `user_id`).
4. **Candidate generation**: sequential (SASRec) + graph CF (LightGCN) +
   collaborative (LightFM, Linux only) + content/intent (FAISS), fused by
   **Reciprocal Rank Fusion**.
5. **Filtering**: exclusions + query-negation parsing.
6. **Multi-objective re-rank**: relevance, freshness, fatigue, context, BCQ
   Q-value, BEST-Rec SSL similarity → linear fusion; fairness PID adjustment;
   Bayesian-surprise diversity λ; **MMR** diversity re-rank.
7. **Explainability**: Shapley (cohort A) / LIME (cohort B).
8. **Side effects**: impression logging, CTR/bandit bookkeeping, latency metric.

Cold-start (no history) falls back to FAISS content match → tag search →
popularity.

## 5. External dependencies

There are **no outbound network calls at runtime**. The only "external" pieces
are local model artifacts loaded from `data/` (LightFM/SASRec pickles, FAISS
index, Torch `.pth` weights) and the SBERT model (downloaded once by
`sentence-transformers`, mocked in tests). This is why there is no retry/circuit
-breaker layer — failure domains are local disk and CPU, handled by
try/except-with-logging around every model load and per-model training block.

## 6. Security posture (post-audit)

- **Auth**: optional shared-secret `X-API-Key` (`APIKeyMiddleware`), enabled by
  setting `API_KEY`. Off by default for local/dev; there is still **no per-user
  identity** (caller-supplied `user_id` is trusted) — documented in AUDIT_LOG.
- **CORS**: origins from `CORS_ALLOW_ORIGINS`; wildcard never combined with
  credentials.
- **Rate limiting**: sliding-window per client IP (proxy-header aware).
- **Input validation**: pydantic models with `extra="forbid"`, `min/max_length`,
  numeric bounds; query `n`/pagination capped (`MAX_N_RECOMMENDATIONS`,
  `MAX_PAGE_LIMIT`) to prevent candidate-blow-up DoS.
- **Deserialization**: FAISS id-map persisted as JSON (not pickle). Remaining
  model pickles/Torch checkpoints are treated as **trusted local artifacts only**
  (see AUDIT_LOG deferred item).
- **Error handling**: global handler returns generic 500 — no traceback leakage.
- **Container**: runs as non-root `app` user.

## 7. Known trade-offs / dead-weight

- LightFM is disabled on Windows (Cython crash) — the engine degrades to
  content/popularity seamlessly.
- SQLite + threaded writes: fine at demo scale; not a high-concurrency store.
- The re-rank loop re-embeds candidate items per request for BCQ/SSL scoring —
  the dominant per-request cost; acceptable at demo scale, flagged for batching.
