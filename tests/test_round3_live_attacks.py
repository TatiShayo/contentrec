"""
Round 3 - Live Exploitation Tests against contentrec
Uses the conftest.py fixture pattern (mocked sentence_transformers, temp DB).
Tests real API security: auth bypass, fuzzing, concurrency, business logic abuse.
"""
import concurrent.futures
from fastapi.testclient import TestClient
from main import app


def test_no_auth_on_any_endpoint(clean_db):
    """CONFIRMED: Zero authentication on all endpoints."""
    with TestClient(app) as client:
        # GET endpoints
        for path in ["/health", "/stats", "/metrics"]:
            r = client.get(path)
            assert r.status_code == 200, f"{path} returned {r.status_code}"

        # Seed an item so recommend works
        client.post("/items", json={"item_id": "test1", "title": "Test Item"})
        app.state.engine.train()
        app.state.faiss_index = app.state.engine.faiss_index

        # Recommend - no auth
        r = client.get("/recommend/test_user?n=5")
        assert r.status_code == 200

        # Feedback - no auth, trusts caller-provided user_id
        r = client.post("/feedback", json={
            "user_id": "someone_elses_account",
            "item_id": "test1",
            "event_type": "view",
        })
        assert r.status_code == 200


def test_feedback_no_identity_verification(clean_db):
    """CONFIRMED: API trusts caller-provided user_id. No session/auth."""
    with TestClient(app) as client:
        r = client.post("/feedback", json={
            "user_id": "anyone_i_want_to_impersonate",
            "item_id": "test1",
            "event_type": "click",
        })
        assert r.status_code == 200


def test_concurrent_feedback_no_locks(clean_db):
    """10 concurrent writes against real SQLite backend."""
    with TestClient(app) as client:
        client.post("/items", json={"item_id": "concur_test", "title": "Test"})

        def post(i):
            return client.post("/feedback", json={
                "user_id": f"racer_{i}",
                "item_id": "concur_test",
                "event_type": "view",
            })

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            results = [f.result() for f in [pool.submit(post, i) for i in range(10)]]

        ok = sum(1 for r in results if r.status_code == 200)
        assert ok == 10


def test_fuzzing_malformed_inputs(clean_db):
    """Fuzzing: wrong types, missing fields, empty body, extreme lengths."""
    with TestClient(app) as client:
        # Missing required fields
        r = client.post("/feedback", json={})
        assert r.status_code == 422

        # Missing body entirely
        r = client.post("/feedback", content=b"")
        assert r.status_code == 422

        # Wrong types
        r = client.post("/feedback", json={
            "user_id": None, "item_id": 42, "event_type": "view",
        })
        assert r.status_code == 422

        # Invalid event_type (only guard that exists)
        r = client.post("/feedback", json={
            "user_id": "test", "item_id": "test", "event_type": "malicious",
        })
        assert r.status_code == 400  # explicit guard

        # Extremely long strings - now rejected by max_length (was: accepted)
        r = client.post("/feedback", json={
            "user_id": "A" * 100000, "item_id": "test", "event_type": "view",
        })
        assert r.status_code == 422  # HARDENED: max_length caps user_id

        # Empty strings now rejected by min_length (was: accepted as junk)
        r = client.post("/feedback", json={
            "user_id": "", "item_id": "", "event_type": "view",
        })
        assert r.status_code == 422  # HARDENED: min_length=1

        # SQL metacharacters - parameterized queries prevent injection
        r = client.post("/feedback", json={
            "user_id": "'; DROP TABLE feedback; --",
            "item_id": "test",
            "event_type": "view",
        })
        assert r.status_code == 200


def test_mass_assignment_rejected(clean_db):
    """HARDENED: extra fields now rejected via model_config extra='forbid'."""
    with TestClient(app) as client:
        r = client.post("/feedback", json={
            "user_id": "attacker",
            "item_id": "test1",
            "event_type": "view",
            "is_admin": True,
            "subscription_tier": "enterprise",
            "role": "superuser",
            "internal_credit": 999999,
        })
        assert r.status_code == 422


def test_unbounded_limits(clean_db):
    """HARDENED: pagination + n now capped; feedback still unlimited (needs auth/quota)."""
    with TestClient(app) as client:
        # Items: pagination now capped (was: 99999999 accepted)
        r = client.get("/items?limit=99999999")
        assert r.status_code == 422  # HARDENED: le=MAX_PAGE_LIMIT

        # Recommend: unbounded n was a candidate-blowup DoS; now capped
        r = client.get("/recommend/dos_user?n=99999999")
        assert r.status_code == 422  # HARDENED: le=MAX_N_RECOMMENDATIONS

        # Feedback: can create unlimited for unlimited fake users
        for i in range(30):
            r = client.post("/feedback", json={
                "user_id": f"fake_bot_{i}",
                "item_id": f"item_{i % 5}",
                "event_type": "view",
            })
            assert r.status_code == 200

        # Stats confirm unbounded growth
        stats = client.get("/stats").json()
        assert stats["feedback_count"] >= 30


def test_concurrent_read_write(clean_db):
    """Mixed concurrent reads and writes."""
    with TestClient(app) as client:
        client.post("/items", json={"item_id": "rw_test", "title": "Concurrent RW"})

        def get_items():
            return client.get("/items")

        def post_feedback():
            return client.post("/feedback", json={
                "user_id": "rw_user", "item_id": "rw_test", "event_type": "click",
            })

        ops = [get_items, get_items, post_feedback, get_items, post_feedback, post_feedback]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = [f.result() for f in [pool.submit(op) for op in ops]]

        ok = sum(1 for r in results if r.status_code == 200)
        assert ok == 6
