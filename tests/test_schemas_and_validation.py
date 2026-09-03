"""Tests for Pydantic schema validation, bounds checking, and input sanitization across all models and endpoints."""

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from main import app
import config
from api.feedback import FeedbackCreate, VALID_EVENT_TYPES
from api.items import ItemCreate
from api.onboarding import QuizRequest, OnboardingSubmit
from api.search import SearchRequest


class TestPydanticSchemas:
    def test_feedback_create_valid(self):
        fb = FeedbackCreate(
            user_id="user_123",
            item_id="item_456",
            event_type="click",
            timestamp="2026-08-14T10:00:00",
            dwell_time=45.5
        )
        assert fb.user_id == "user_123"
        assert fb.dwell_time == 45.5

    def test_feedback_create_invalid_events_and_bounds(self):
        # Empty user_id
        with pytest.raises(ValidationError):
            FeedbackCreate(user_id="", item_id="i1", event_type="click")

        # Empty item_id
        with pytest.raises(ValidationError):
            FeedbackCreate(user_id="u1", item_id="", event_type="click")

        # Negative dwell_time
        with pytest.raises(ValidationError):
            FeedbackCreate(user_id="u1", item_id="i1", event_type="click", dwell_time=-5.0)

        # Excessive dwell_time (> 86400s / 1 day)
        with pytest.raises(ValidationError):
            FeedbackCreate(user_id="u1", item_id="i1", event_type="click", dwell_time=100000.0)

        # Oversized user_id (> 256 chars)
        with pytest.raises(ValidationError):
            FeedbackCreate(user_id="u" * 300, item_id="i1", event_type="click")

        # Extra forbidden fields (Mass assignment protection)
        with pytest.raises(ValidationError):
            FeedbackCreate(user_id="u1", item_id="i1", event_type="click", is_admin=True)

    def test_item_create_valid_and_bounds(self):
        it = ItemCreate(
            item_id="item_1",
            title="Clean Code",
            tags="programming,craftsmanship",
            category="books",
            metadata={"author": "Robert C. Martin", "pages": 464}
        )
        assert it.item_id == "item_1"
        assert it.metadata["pages"] == 464

        # Empty title
        with pytest.raises(ValidationError):
            ItemCreate(item_id="i1", title="")

        # Oversized metadata (>100 keys)
        large_metadata = {f"k_{i}": f"v_{i}" for i in range(101)}
        with pytest.raises(ValidationError, match="metadata may not contain more than 100 keys"):
            ItemCreate(item_id="i1", title="Title", metadata=large_metadata)

        # Extra forbidden fields
        with pytest.raises(ValidationError):
            ItemCreate(item_id="i1", title="Title", admin_override="yes")

    def test_quiz_request_bounds(self):
        q = QuizRequest(n_quiz=8)
        assert q.n_quiz == 8

        # Below min (ge=3)
        with pytest.raises(ValidationError):
            QuizRequest(n_quiz=2)

        # Above max (le=15)
        with pytest.raises(ValidationError):
            QuizRequest(n_quiz=16)

        # Extra fields
        with pytest.raises(ValidationError):
            QuizRequest(n_quiz=5, extra="bad")

    def test_onboarding_submit_bounds(self):
        submit = OnboardingSubmit(
            user_id="new_user_1",
            ratings={"i1": 1.0, "i2": -1.0}
        )
        assert len(submit.ratings) == 2

        # Oversized ratings (>200 items)
        oversized_ratings = {f"item_{i}": 1.0 for i in range(201)}
        with pytest.raises(ValidationError, match="ratings may not contain more than 200 entries"):
            OnboardingSubmit(user_id="u1", ratings=oversized_ratings)

        # Empty user_id
        with pytest.raises(ValidationError):
            OnboardingSubmit(user_id="", ratings={"i1": 1.0})

    def test_search_request_bounds(self):
        req = SearchRequest(query="recommendation engines", n=10)
        assert req.n == 10

        # Empty query or missing query
        with pytest.raises(ValidationError):
            SearchRequest(n=5)

        # n <= 0
        with pytest.raises(ValidationError):
            SearchRequest(query="test", n=0)

        # n > MAX_N_RECOMMENDATIONS
        with pytest.raises(ValidationError):
            SearchRequest(query="test", n=config.MAX_N_RECOMMENDATIONS + 1)


class TestEndpointInputValidationViaHTTP:
    def test_endpoint_n_recommendations_bounds(self, clean_db):
        with TestClient(app) as client:
            client.post("/items", json={"item_id": "item1", "title": "Test Item"})

            # Valid n
            r = client.get("/recommend/user1?n=5")
            assert r.status_code == 200

            # n = 0 should be rejected with 422
            r = client.get("/recommend/user1?n=0")
            assert r.status_code == 422

            # n > MAX_N_RECOMMENDATIONS should be rejected with 422
            r = client.get(f"/recommend/user1?n={config.MAX_N_RECOMMENDATIONS + 1}")
            assert r.status_code == 422

    def test_endpoint_items_pagination_bounds(self, clean_db):
        with TestClient(app) as client:
            # offset < 0 rejected
            r = client.get("/items?offset=-1")
            assert r.status_code == 422

            # limit < 1 rejected
            r = client.get("/items?limit=0")
            assert r.status_code == 422

            # limit > MAX_PAGE_LIMIT rejected
            r = client.get(f"/items?limit={config.MAX_PAGE_LIMIT + 1}")
            assert r.status_code == 422

            # Valid pagination
            r = client.get("/items?offset=0&limit=50")
            assert r.status_code == 200
