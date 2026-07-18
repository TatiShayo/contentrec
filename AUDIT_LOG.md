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
