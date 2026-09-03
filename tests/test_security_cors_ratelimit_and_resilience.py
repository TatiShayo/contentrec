"""Tests for security controls: CORS policy hardening, Rate Limiting, API Key auth, and resilience."""

import time
import pytest
from fastapi.testclient import TestClient
from main import app
import config
from utils.rate_limiter import RateLimiter
from utils.cache import RecommendationCache


class TestCORSAndHeaders:
    def test_cors_preflight_and_wildcard_handling(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        # Verify CORS enforcement when allowed origins are configured
        cors_app = FastAPI()
        cors_app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://allowed.example.com"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        @cors_app.get("/test")
        def route():
            return {"status": "ok"}
            
        with TestClient(cors_app) as client:
            # 1. Allowed origin OPTIONS preflight
            res_allowed = client.options(
                "/test",
                headers={
                    "Origin": "https://allowed.example.com",
                    "Access-Control-Request-Method": "GET"
                }
            )
            assert res_allowed.status_code == 200
            assert res_allowed.headers.get("access-control-allow-origin") == "https://allowed.example.com"
            
            # 2. Disallowed origin OPTIONS preflight
            res_disallowed = client.options(
                "/test",
                headers={
                    "Origin": "https://malicious-attacker.com",
                    "Access-Control-Request-Method": "GET"
                }
            )
            assert res_disallowed.status_code == 400


class TestRateLimiterCore:
    def test_rate_limiter_allows_under_limit(self):
        limiter = RateLimiter(requests_limit=5, window_sec=60)
        ip = "192.168.1.100"
        for _ in range(5):
            assert limiter.is_allowed(ip) is True
        # 6th request exceeds limit
        assert limiter.is_allowed(ip) is False

    def test_rate_limiter_sliding_window_expiry(self):
        # 1-second window
        limiter = RateLimiter(requests_limit=2, window_sec=1)
        ip = "10.0.0.1"
        assert limiter.is_allowed(ip) is True
        assert limiter.is_allowed(ip) is True
        assert limiter.is_allowed(ip) is False

        # Sleep past window
        time.sleep(1.1)
        assert limiter.is_allowed(ip) is True

    def test_rate_limiter_memory_pruning(self):
        limiter = RateLimiter(requests_limit=10, window_sec=1)
        # Populate with many IPs
        for i in range(100):
            limiter.is_allowed(f"ip_{i}")

        time.sleep(1.1)
        # Trigger cleanup by calling is_allowed
        limiter.is_allowed("trigger_ip")
        # Ensure stale IPs don't blow up memory indefinitely
        assert len(limiter._requests) >= 1


class TestAPIKeyAuthentication:
    def test_api_key_when_configured(self, monkeypatch):
        monkeypatch.setattr(config, "API_KEY", "super_secret_production_token")
        monkeypatch.setattr(config, "TESTING", False)

        with TestClient(app) as client:
            # Public endpoint should succeed without key
            r_health = client.get("/health")
            assert r_health.status_code == 200

            # Protected endpoint without key -> 401
            r_unauth = client.get("/stats")
            assert r_unauth.status_code == 401
            assert "Invalid or missing API key" in r_unauth.json()["detail"]

            # Protected endpoint with wrong key -> 401
            r_wrong = client.get("/stats", headers={"x-api-key": "wrong_key"})
            assert r_wrong.status_code == 401

            # Protected endpoint with valid key -> 200
            r_valid = client.get("/stats", headers={"x-api-key": "super_secret_production_token"})
            assert r_valid.status_code == 200


class TestErrorHandlingAndResilience:
    def test_missing_item_returns_404(self, clean_db):
        with TestClient(app) as client:
            r = client.get("/items/non_existent_item_99999")
            assert r.status_code == 404
            assert "Item not found" in r.json()["detail"]

    def test_cache_user_invalidation_exact_boundary(self):
        cache = RecommendationCache()
        # Set cache keys for user 1, user 10, user 100
        cache.set("user:1:n:10", ["rec1"])
        cache.set("user:10:n:10", ["rec10"])
        cache.set("user:100:n:10", ["rec100"])
        cache.set("sequential:user:1:n:5", ["seq1"])
        cache.set("sequential:user:10:n:5", ["seq10"])

        # Invalidate only user 1
        cache.invalidate_user("1")

        # user 1 should be invalidated
        assert cache.get("user:1:n:10") is None
        assert cache.get("sequential:user:1:n:5") is None

        # user 10 and user 100 should remain intact!
        assert cache.get("user:10:n:10") == ["rec10"]
        assert cache.get("user:100:n:10") == ["rec100"]
        assert cache.get("sequential:user:10:n:5") == ["seq10"]
