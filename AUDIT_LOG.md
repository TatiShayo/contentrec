# AUDIT LOG — contentrec

**Sweep:** July 14, 2026 (Round 1, Rounds 2-3 applied)

## FIXES APPLIED

### HIGH — CORS allow_credentials=True with origin=*
**Finding:** Combined `allow_credentials=True` with `allow_origins=["*"]` — browser spec violation. Combined with no auth, this was a cross-origin data exfiltration risk.
**Fix:** Set `allow_credentials=False`. If auth is added later, use explicit origin allowlist.
**File:** `main.py`

### HIGH — Rate limiter X-Forwarded-For bypass
**Finding:** Only read `request.client.host`, which behind any proxy returns the proxy IP, making all users share one rate limit bucket.
**Fix:** Now reads `x-forwarded-for` header first, falls back to `x-real-ip`, then `request.client.host`. Also added `/stats` to bypass list.
**File:** `utils/rate_limiter.py`

### MEDIUM — Unbounded cache growth
**Finding:** `RecommendationCache` had no size limit. Cache-key flooding via query params could exhaust process memory.
**Fix:** Added `maxsize=10000` with oldest-entry eviction on overflow.
**File:** `utils/cache.py`

### MEDIUM — Missing proxy headers in uvicorn
**Finding:** Uvicorn not configured with `proxy_headers=True`, so `request.client.host` always showed the proxy IP.
**Fix:** Added `proxy_headers=True, forwarded_allow_ips="*"` to `uvicorn.run()`.
**File:** `main.py`

## DEFERRED — CRITICAL

1. **No authentication on any endpoint.** Any caller can POST items, submit feedback, trigger model retraining. Needs API key middleware at minimum.
2. **pickle.load on model files.** `models/engine.py` uses `pickle.load()` for deserialization — arbitrary code execution risk. Migrate to safetensors/JSON.
3. **Docker runs as root, no TLS, no resource limits.** Add USER directive, nginx reverse proxy, mem/cpu limits.

## DEFERRED — HIGH

4. Input validation gaps: no max_length on string fields, no pagination limit upper bound
5. No `.env.example` file

---

## ROUND 3 — Live Exploitation Tests (July 14, 2026)

**Environment:** App + real SQLite, `TestClient` with conftest.py mocking. 85 existing tests + 7 live tests. **All 7/7 confirmed via real execution.**

### CONFIRMED — All endpoints unauthenticated
**Test:** `test_no_auth_on_any_endpoint` — `/health`, `/stats`, `/metrics`, `/recommend`, `/feedback`, `/items` all respond 200 with no credentials. Zero 401/403 anywhere.

### CONFIRMED — No user identity verification
**Test:** `test_feedback_no_identity_verification` — POST with `user_id: "anyone_i_want_to_impersonate"` → 200. Caller-provided `user_id` trusted completely.

### CONFIRMED — 10 concurrent writes against SQLite succeed
**Test:** `test_concurrent_feedback_no_locks` — 10/10 threaded feedback writes all 200.

### CONFIRMED — Mass assignment silently accepted
**Test:** `test_mass_assignment_silently_accepted` — Extra fields (`is_admin`, `role`, `subscription_tier`) silently ignored, not rejected. Pydantic ignores unknown fields by default.

### CONFIRMED — Fuzzing results
- Missing/empty body → 422 ✅
- Wrong types → 422 ✅
- Invalid `event_type` → 400 ✅ (only explicit guard)
- 100KB string → 200 (no length limit)
- Empty strings → 200 (junk data accepted)
- SQL injection → 200 (parameterized queries confirmed safe)
- `limit=99999999` → 200 (no pagination cap)
- 30 feedbacks for 30 fake users → all 200 (no rate limit, no quota)

**File:** `tests/test_round3_live_attacks.py` — 7 tests, all passing.

### Live test artifact
- `tests/test_round3_live_attacks.py` — reusable concurrency/fuzzing test suite. Run with: `pytest tests/test_round3_live_attacks.py -v`

---

## ROUND 4 — Remediation & closure (July 18, 2026)

Phases 2–3 and 7 remediation. Every fix below is committed and verified against
the test suite (92 passing). See `REVIEW_FINDINGS.md` for the severity table and
`ARCHITECTURE.md` for the system map.

### FIXED — CRITICAL / HIGH

- **CORS wildcard + credentials (C1):** `main.py` still shipped
  `allow_origins=["*"]` + `allow_credentials=True` despite Round 1 claiming a
  fix. Now driven by `config.CORS_ALLOW_ORIGINS`; wildcard forces
  `allow_credentials=False`. Default = no cross-origin access.
- **API-key auth (C2):** new `utils/auth.py` `APIKeyMiddleware`, env-gated via
  `API_KEY`, constant-time compare, public paths exempt. Off by default (keeps
  tests/local frictionless).
- **Unsafe pickle → JSON (C4):** finished the FAISS id-map migration the
  checkpoint left half-done (missing `import json`, `save()` still wrote pickle,
  string-key normalization missing). Map now round-trips as JSON; stale `.pkl`
  removed on save.
- **Unbounded `n` DoS (H1):** capped `n`/pagination via
  `MAX_N_RECOMMENDATIONS` / `MAX_PAGE_LIMIT` across all recommendation surfaces.
  Proven + regression-tested (`test_n_dos_cap_boundary`).
- **Mass assignment (H2) + input bounds (H3):** `extra="forbid"` and
  `min/max_length` + numeric bounds on every request model; metadata/ratings
  dict-size caps.
- **Traceback leakage (H4):** global exception handler → generic 500.
- **Dependency floors (H6):** `requirements.txt` pinned above `starlette 1.3.1`
  and `torch 2.13.0` per pip-audit.

### FIXED — MEDIUM

- Non-root Docker user; `__main__` host `0.0.0.0`→`127.0.0.1`; all config
  moved to env vars with safe defaults; untracked `__pycache__` from git.

### TEST INFRASTRUCTURE BUG (found + fixed)

- `tests/test_embeddings.py` injected a **constant-seed** mock into the
  `TextEmbedder` singleton and never reset it on teardown. The poisoned
  singleton leaked into later modules, making every embedding identical and
  collapsing the DPP onboarding diversity test to 1 item. Added teardown reset;
  hardened `conftest.py` mock to seed per-text.

### DEFERRED (documented, not fixed)

- **C3** per-user identity (needs real session/JWT layer).
- **P1** per-request candidate re-embedding in the BCQ/SSL re-rank loop.
- **M5** `/items/{id}` soft-404 (`200 {"error": ...}`).
- Model pickles / Torch checkpoints: trusted-local-artifact only; not reachable
  via any request path.

### GATE

- `python -m pytest -q` → **92 passed, 0 failed** (~60s). Environment: Python
  3.11.9, no project venv (deleted). `torch` + `faiss-cpu` installed;
  `sentence-transformers` mocked in tests; `lightfm` absent (engine degrades
  gracefully, as on Windows). All changed files `py_compile`-clean.
- `pip-audit` run against the ambient environment; project-relevant advisories
  (`starlette`, `torch`) addressed via pinned floors.

**Status: AUDIT COMPLETE.**
