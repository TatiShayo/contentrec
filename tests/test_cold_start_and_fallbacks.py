"""Tests for cold-start recommendation logic, DPP onboarding selection, and model fallbacks.

Verifies:
- Cold-start with empty database, tag-matching, popularity fallback, and catalog fallback
- DPP diversity-maximizing onboarding item selection
- Cold-start user profile bootstrap from onboarding ratings
- Graceful degradation when models are missing or uninitialized
"""

import pytest
import numpy as np
from data.items import add_item, get_all_items, get_item
from data.feedback import add_feedback, get_all_feedback
from models.engine import RecommendationEngine
from utils.dpp import DPPSelector, ColdStartProfileBuilder


class TestColdStartRecommendationFallbacks:
    def test_cold_start_empty_database(self, clean_db):
        engine = RecommendationEngine()
        recs = engine.cold_start_recommend(tags_or_category="tech", n=5)
        assert recs == []
        recs_rec = engine.recommend("brand_new_user", n=5)
        assert recs_rec == []

    def test_cold_start_catalog_fallback_no_feedback(self, clean_db):
        # Database has items, but no tags match and no feedback exists
        for i in range(5):
            add_item(f"cat_item_{i}", f"Catalog Item {i}", tags="general", category="misc")

        engine = RecommendationEngine()
        recs = engine.cold_start_recommend(tags_or_category="non_existent_tag", n=3)
        assert len(recs) == 3
        assert all("item_id" in r for r in recs)

    def test_cold_start_tag_matching(self, clean_db):
        add_item("tech_1", "Intro to AI", tags="ai,machine learning", category="articles")
        add_item("tech_2", "Python Deep Dive", tags="python,coding", category="articles")
        add_item("cooking_1", "Italian Pasta", tags="food,cooking", category="videos")

        engine = RecommendationEngine()
        recs = engine.cold_start_recommend(tags_or_category="python", n=2)
        assert len(recs) > 0
        assert recs[0]["item_id"] == "tech_2"
        assert recs[0]["source"] in ["cold_start_content", "cold_start_tag"]

    def test_cold_start_popularity_ranking_fallback(self, clean_db):
        # Seed items
        for i in range(4):
            add_item(f"item_pop_{i}", f"Title {i}", tags="random")

        # Feedback: item_pop_2 has 4 views, item_pop_1 has 2 views
        for _ in range(4):
            add_feedback("u1", "item_pop_2", "view")
        for _ in range(2):
            add_feedback("u2", "item_pop_1", "view")
        add_feedback("u3", "item_pop_0", "view")

        engine = RecommendationEngine()
        # With tags_or_category=None, should fall back to popularity ranking
        recs = engine.cold_start_recommend(tags_or_category=None, n=3)
        assert len(recs) == 3
        # item_pop_2 should be ranked first due to popularity (4 views)
        assert recs[0]["item_id"] == "item_pop_2"
        assert recs[0]["source"] == "popularity"
        assert recs[1]["item_id"] == "item_pop_1"


class TestDPPOnboardingAndProfileBootstrap:
    def test_dpp_select_diverse_items_empty(self):
        selected = DPPSelector.select_diverse_items([], pool_size=10, n_quiz=5)
        assert selected == []

    def test_dpp_select_diverse_items_from_catalog(self, clean_db):
        items = [
            {"item_id": "m1", "title": "Avengers Action Movie", "tags": "action,hero", "category": "movies"},
            {"item_id": "m2", "title": "Batman Dark Hero", "tags": "action,hero", "category": "movies"},
            {"item_id": "b1", "title": "Clean Architecture Book", "tags": "software,architecture", "category": "books"},
            {"item_id": "s1", "title": "Rock Music Song", "tags": "rock,guitar", "category": "music"},
            {"item_id": "n1", "title": "World Economy News", "tags": "news,finance", "category": "news"},
        ]
        for it in items:
            add_item(it["item_id"], it["title"], it["tags"], it["category"])

        catalog = get_all_items()
        selected = DPPSelector.select_diverse_items(catalog, pool_size=10, n_quiz=3)
        assert len(selected) == 3
        selected_ids = [it["item_id"] for it in selected]
        assert len(set(selected_ids)) == 3  # No duplicates

    def test_cold_start_profile_bootstrap(self):
        ratings = {"item_1": 1.0, "item_2": 1.0, "item_3": -1.0}
        item_details = {
            "item_1": {"title": "Python Programming", "tags": "python", "category": "books"},
            "item_2": {"title": "FastAPI Web Apps", "tags": "python,web", "category": "articles"},
            "item_3": {"title": "Disliked Movie", "tags": "drama", "category": "movies"},
        }
        gcn_embs = np.random.randn(2, 64).astype(np.float32)
        item_to_gcn = {"item_1": 0, "item_2": 1}

        gcn_profile, sbert_profile = ColdStartProfileBuilder.bootstrap_user_profile(
            ratings, item_details, gcn_embs, item_to_gcn
        )
        assert gcn_profile.shape == (64,)
        assert sbert_profile.shape == (384,)
        assert np.isclose(np.linalg.norm(gcn_profile), 1.0, atol=1e-4)
        assert np.isclose(np.linalg.norm(sbert_profile), 1.0, atol=1e-4)

    def test_cold_start_profile_bootstrap_empty_ratings(self):
        ratings = {}
        item_details = {}
        gcn_embs = np.ones((2, 64), dtype=np.float32)
        item_to_gcn = {}

        gcn_profile, sbert_profile = ColdStartProfileBuilder.bootstrap_user_profile(
            ratings, item_details, gcn_embs, item_to_gcn
        )
        assert gcn_profile.shape == (64,)
        assert sbert_profile.shape == (384,)


class TestModelDegradationAndExclusionFilters:
    def test_recommend_with_exclusion_filters(self, clean_db):
        add_item("m1", "Movie 1", "action", "movies")
        add_item("m2", "Movie 2", "comedy", "movies")
        add_item("b1", "Book 1", "science", "books")

        add_feedback("u1", "m1", "view")
        add_feedback("u1", "m2", "view")
        add_feedback("u1", "b1", "view")

        engine = RecommendationEngine()
        engine.train()

        # Exclude movies category
        recs = engine.recommend("u1", n=5, exclude_categories="movies")
        for r in recs:
            item = get_item(r["item_id"])
            if item:
                assert item.get("category") != "movies"

        # Exclude specific item ID
        recs_item_ex = engine.recommend("u1", n=5, exclude_items="m1")
        rec_ids = [r["item_id"] for r in recs_item_ex]
        assert "m1" not in rec_ids

    def test_similar_items_with_exclusion(self, clean_db):
        add_item("t1", "Target Item", "tech", "articles")
        add_item("s1", "Similar Item 1", "tech", "articles")
        add_item("s2", "Similar Item 2", "tech", "articles")
        add_item("s3", "Similar Item 3", "tech", "books")

        engine = RecommendationEngine()
        engine.train()

        sim = engine.similar_items("t1", n=5, exclude_items="s1", exclude_categories="books")
        sim_ids = [s["item_id"] for s in sim]
        assert "t1" not in sim_ids  # target itself excluded
        assert "s1" not in sim_ids  # excluded by item ID
        assert "s3" not in sim_ids  # excluded by category
