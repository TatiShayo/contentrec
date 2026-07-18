# REVIEW FINDINGS — ContentRec

Consolidated audit findings (Phases 2–3, 7) with severity, status, and evidence.
Companion to `AUDIT_LOG.md` (chronological) and `ARCHITECTURE.md` (system map).

Status legend: ✅ fixed · 🟡 mitigated / safe-default · ⛔ deferred (documented)

---

## CRITICAL

| # | Finding | Status | Fix / note |
|---|---------|--------|------------|
| C1 | **CORS `allow_origins=["*"]` + `allow_credentials=True`** in `main.py`. Browser-spec violation and cross-origin exfiltration risk (compounded by no auth). The code still shipped the vulnerable combo despite an earlier log claiming otherwise. | ✅ | Origins now from `config.CORS_ALLOW_ORIGINS`; wildcard forces `allow_credentials=False`. Default = no cross-origin access. |
| C2 | **No authentication on any endpoint.** Anyone can write items/feedback, trigger `/train`. | 🟡 | Added env-gated `APIKeyMiddleware` (`X-API-Key`, constant-time compare). Off by default to preserve local/test UX; a real deploy sets `API_KEY`. Per-user identity still absent → C3. |
| C3 | **No per-user identity** — caller-supplied `user_id` is trusted (impersonation). | ⛔ | Needs real session/JWT layer; out of scope for a local demo. Documented; API-key gate is the interim control. |
| C4 | **Unsafe `pickle.load` of model artifacts** (`engine.py`, FAISS map). Arbitrary code execution if an attacker can plant a file. | 🟡 | FAISS id-map migrated to **JSON** (was pickle, and the prior checkpoint left it half-broken/un-importable — fixed). LightFM/SASRec/BCQ/Torch checkpoints remain pickle/`torch.load` and are now explicitly scoped as **trusted local artifacts only** (not attacker-reachable via any endpoint). |

## HIGH

| # | Finding | Status | Fix / note |
|---|---------|--------|------------|
| H1 | **Unbounded `n` DoS.** `GET /recommend/{u}?n=1e9` → candidate pool `max(n*3,50)`, per-candidate re-embedding/scoring, and a unique cache key per `n` (cache pollution). | ✅ | `n` capped by `config.MAX_N_RECOMMENDATIONS` (default 100) on `/recommend`, `/similar`, `/sequential`, `/search`; pagination capped by `MAX_PAGE_LIMIT`. Proven + regression-tested (`test_n_dos_cap_boundary`). |
| H2 | **Mass assignment** — pydantic silently ignored unknown fields (`is_admin`, `role`, …). | ✅ | `model_config = {"extra": "forbid"}` on every request model → 422 on unexpected fields. |
| H3 | **No input length bounds** — 100 KB strings accepted, empty IDs accepted as junk. | ✅ | `min_length`/`max_length` on IDs/strings; numeric `ge/le` bounds; metadata/ratings dict-size caps. |
| H4 | **Traceback leakage** on unhandled errors. | ✅ | Global exception handler returns generic `500 {"detail": "Internal server error"}` and logs server-side. |
| H5 | **Rate limiter X-Forwarded-For bypass** (prior round). | ✅ | Reads `x-forwarded-for` → `x-real-ip` → `client.host`; uvicorn `proxy_headers=True`. |
| H6 | **Dependency advisories** (pip-audit): `starlette < 1.3.1` (DoS), `torch < 2.13.0`. | 🟡 | Secure version floors pinned in `requirements.txt`. Cannot force-upgrade an already-installed ambient env from here; floors apply on next clean install. |

## MEDIUM

| # | Finding | Status | Fix / note |
|---|---------|--------|------------|
| M1 | **Docker runs as root.** | ✅ | Added non-root `app` user + ownership in `Dockerfile`. |
| M2 | **Unbounded cache growth** (prior round). | ✅ | `maxsize=10000` w/ oldest-entry eviction. |
| M3 | **Server bound to `0.0.0.0`** in the `__main__` runner. | ✅ | Changed to `127.0.0.1` for the local dev runner (containers override host explicitly). |
| M4 | **No secrets in env.** No hardcoded secrets existed, but config was static. | ✅ | `config.py` now reads all paths/limits/secrets from env with safe defaults. |
| M5 | **`/items/{id}` returns 200 with `{"error": ...}`** on miss (soft 404). | ⛔ | Left as-is to avoid breaking existing clients/tests; noted. |

## PERFORMANCE

| # | Finding | Status | Note |
|---|---------|--------|------|
| P1 | Per-request re-embedding of every candidate for BCQ Q-value + BEST-Rec SSL scoring — the dominant recommend-path cost. | ⛔ | Flagged for batch-embedding; correctness-preserving refactor deferred (touches hot RL/SSL loop). |
| P2 | `RecommendationCache` (TTL + maxsize) already covers repeated recommend/search calls; invalidated per-user on feedback and globally on retrain. | ✅ | Verified present and bounded. |
| P3 | FAISS uses **HNSW** (`efConstruction=200`, `efSearch=64`) with incremental `add_item` — avoids full rebuilds on single-item inserts. Full rebuild only on `/train`. | ✅ | Build cost is amortized; acceptable at demo scale. |
| P4 | DPP onboarding selector is `O(pool² · n_quiz)` pure-Python. | ⛔ | Fine for `pool_size≤50`; documented, not hot-path. |

## RELIABILITY

| # | Finding | Status | Note |
|---|---------|--------|------|
| R1 | **No external network calls** at runtime — all model/data access is local disk/CPU. | ✅ | Documented in ARCHITECTURE §5; retry/circuit-breaker layer intentionally absent. |
| R2 | Every model load/train block is wrapped in try/except with logging — a single failing sub-model does not take down `/recommend`. | ✅ | Graceful degradation verified (BEST-Rec checkpoint size-mismatch logs an error and continues). |
| R3 | Recommend-path test coverage. | ✅ | `test_api`, `test_engine`, `test_integration`, `test_round3_live_attacks` exercise the recommend/feedback/search flows; 92 tests green. |
| R4 | Background tasks (retrain loop, SASRec online thread) leak on shutdown. | ✅ | `lifespan` cancels the asyncio task and joins the daemon thread with a timeout. |

---

## Root-cause themes

- **Boundary validation was assumed, not enforced.** Every write model lacked
  `extra="forbid"` and length/range bounds, and query caps — one shared pattern
  fixed across all routers.
- **"Fixed" ≠ verified.** CORS and the FAISS-map JSON migration were both logged
  as done but shipped broken/reverted. Re-proving each finding against the actual
  code surfaced them.
