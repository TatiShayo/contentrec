import pytest
import time
from fastapi.testclient import TestClient
from main import app
from utils.cache import RecommendationCache
from utils.metrics import MetricsTracker
from utils.diversity import mmr_rerank


def test_cache_ttl_and_invalidation():
    cache = RecommendationCache(default_ttl=1)
    cache.set("key1", "val1")
    assert cache.get("key1") == "val1"
    
    # Wait for TTL expiration
    time.sleep(1.1)
    assert cache.get("key1") is None
    
    # User invalidation test
    cache.set("user:u1:rec", "recs_u1")
    cache.set("user:u2:rec", "recs_u2")
    cache.invalidate_user("u1")
    assert cache.get("user:u1:rec") is None
    assert cache.get("user:u2:rec") == "recs_u2"


def test_metrics_tracker():
    tracker = MetricsTracker()
    tracker.record_cache_hit()
    tracker.record_cache_hit()
    tracker.record_cache_miss()
    tracker.record_latency("recommend", 0.05)
    tracker.record_latency("recommend", 0.15)
    
    metrics = tracker.get_metrics()
    assert metrics["cache"]["hits"] == 2
    assert metrics["cache"]["misses"] == 1
    assert abs(metrics["cache"]["hit_rate"] - (2.0 / 3.0)) < 1e-4
    assert abs(metrics["average_latency_seconds"]["recommend"] - 0.10) < 1e-4


def test_mmr_diversity():
    candidates = [
        {"item_id": "i1", "score": 1.0},
        {"item_id": "i2", "score": 0.9},
        {"item_id": "i3", "score": 0.8},
    ]
    item_details = {
        "i1": {"item_id": "i1", "title": "Action Movie A", "tags": "action,adventure", "category": "movies"},
        # i2 is identical in category and tags to i1
        "i2": {"item_id": "i2", "title": "Action Movie B", "tags": "action,adventure", "category": "movies"},
        # i3 is completely different in tags and category
        "i3": {"item_id": "i3", "title": "Romance Book", "tags": "romance,love", "category": "books"},
    }

    
    from unittest.mock import patch
    import numpy as np
    
    # Pre-defined orthogonal vectors
    # i1 and i2 are highly similar/identical
    # i3 is completely different (orthogonal to i1/i2)
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0], dtype=np.float32)
    
    embeddings_map = {"i1": v1, "i2": v2, "i3": v3}
    
    with patch('embeddings.text.TextEmbedder.embed_item') as mock_embed:
        mock_embed.side_effect = lambda item: embeddings_map[item["item_id"]]
        
        # Pure relevance (lambda=1.0) should pick i1 then i2
        pure_rel = mmr_rerank(candidates, item_details, n=2, diversity_lambda=1.0)
        assert [c["item_id"] for c in pure_rel] == ["i1", "i2"]
        
        # Pure diversity (lambda=0.0) should penalize similarity and select different item i3 over i2
        diverse = mmr_rerank(candidates, item_details, n=2, diversity_lambda=0.0)
        assert [c["item_id"] for c in diverse] == ["i1", "i3"]



def test_metrics_endpoint(clean_db):
    with TestClient(app) as client:
        # Seed items
        client.post("/items", json={"item_id": "i1", "title": "T1"})
        client.post("/feedback", json={"user_id": "u1", "item_id": "i1", "event_type": "view"})
        
        # Trigger train to ensure everything initialized
        client.post("/train")
        
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "cache" in data
        assert "database" in data
        assert data["database"]["item_count"] == 1
        assert data["database"]["feedback_count"] == 1


def test_rate_limiting():
    from utils.rate_limiter import RateLimiter
    limiter = RateLimiter(requests_limit=2, window_sec=10)
    assert limiter.is_allowed("127.0.0.1") is True
    assert limiter.is_allowed("127.0.0.1") is True
    # Third request exceeds limit
    assert limiter.is_allowed("127.0.0.1") is False
    # Separate IP remains unaffected
    assert limiter.is_allowed("192.168.1.1") is True


def test_ab_test_manager():
    from utils.ab_test import ABTestManager
    c1 = ABTestManager.get_cohort("user_alice")
    c2 = ABTestManager.get_cohort("user_alice")
    assert c1 == c2
    
    # Cohorts should distribute across a population
    cohorts = {ABTestManager.get_cohort(f"user_{i}") for i in range(100)}
    assert "A" in cohorts
    assert "B" in cohorts


def test_recommend_ab_routing(clean_db):
    with TestClient(app) as client:
        # Seed items
        client.post("/items", json={"item_id": "i1", "title": "T1"})
        client.post("/feedback", json={"user_id": "user_alice", "item_id": "i1", "event_type": "view"})
        client.post("/train")
        
        from utils.ab_test import ABTestManager
        cohort = ABTestManager.get_cohort("user_alice")
        
        resp = client.get("/recommend/user_alice")
        assert resp.status_code == 200
        
        # Verify from metrics endpoint that cohort serving count was recorded
        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        metrics = metrics_resp.json()
        assert metrics["cohort_recommendations_served"][cohort] >= 1


def test_recommendation_filtering(clean_db):
    with TestClient(app) as client:
        # Seed items with categories
        client.post("/items", json={"item_id": "item_1", "title": "SciFi Book", "category": "books"})
        client.post("/items", json={"item_id": "item_2", "title": "Action Movie", "category": "movies"})
        client.post("/items", json={"item_id": "item_3", "title": "Historical Novel", "category": "books"})
        
        # User Alice liked SciFi Book
        client.post("/feedback", json={"user_id": "user_alice", "item_id": "item_1", "event_type": "like"})
        client.post("/train")
        
        # Get recommendations excluding category "books"
        resp = client.get("/recommend/user_alice?exclude_categories=books")
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        item_ids = [r["item_id"] for r in recs]
        assert "item_2" in item_ids or len(item_ids) == 0
        assert "item_3" not in item_ids
        assert "item_1" not in item_ids
        
        # Get recommendations excluding item_2
        resp = client.get("/recommend/user_alice?exclude_items=item_2")
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        item_ids = [r["item_id"] for r in recs]
        assert "item_2" not in item_ids


def test_conversion_tracking_and_ctr(clean_db):
    with TestClient(app) as client:
        client.post("/items", json={"item_id": "i1", "title": "T1"})
        client.post("/items", json={"item_id": "i2", "title": "T2"})
        client.post("/feedback", json={"user_id": "user_alice", "item_id": "i1", "event_type": "view"})
        client.post("/train")
        
        # Request recommendations (records impression for user_alice's cohort)
        rec_resp = client.get("/recommend/user_alice")
        assert rec_resp.status_code == 200
        recs = rec_resp.json()["recommendations"]
        assert len(recs) > 0
        rec_item = recs[0]["item_id"]
        
        # Post feedback for the recommended item (records conversion)
        fb_resp = client.post("/feedback", json={"user_id": "user_alice", "item_id": rec_item, "event_type": "click"})
        assert fb_resp.status_code == 200
        
        # Verify conversions and CTR under metrics
        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        metrics = metrics_resp.json()
        
        from utils.ab_test import ABTestManager
        cohort = ABTestManager.get_cohort("user_alice")
        assert metrics["cohort_conversions"][cohort] >= 1
        assert metrics["cohort_click_through_rate"][cohort] > 0.0


def test_automatic_retraining_trigger(clean_db):
    import config
    # Temporarily set retraining threshold to 2 for testing
    old_threshold = config.RETRAIN_THRESHOLD_FEEDBACK
    config.RETRAIN_THRESHOLD_FEEDBACK = 2
    try:
        with TestClient(app) as client:
            client.post("/items", json={"item_id": "i1", "title": "T1"})
            client.post("/items", json={"item_id": "i2", "title": "T2"})
            client.post("/train")
            
            # Reset engine retraining variables to verify changes
            app.state.engine.new_feedback_counter = 0
            
            # Post first feedback event
            client.post("/feedback", json={"user_id": "user_alice", "item_id": "i1", "event_type": "view"})
            assert app.state.engine.new_feedback_counter == 1
            
            # Post second feedback event (reaches threshold of 2, triggers train)
            client.post("/feedback", json={"user_id": "user_alice", "item_id": "i2", "event_type": "click"})
            
            # Wait a brief moment for background thread to execute training
            time.sleep(0.5)
            # Counter should be reset to 0 upon retraining
            assert app.state.engine.new_feedback_counter == 0
    finally:
        config.RETRAIN_THRESHOLD_FEEDBACK = old_threshold


def test_contextual_recommendations(clean_db):
    with TestClient(app) as client:
        # Seed items: movies vs books
        client.post("/items", json={"item_id": "m1", "title": "SciFi Movie", "category": "movies"})
        client.post("/items", json={"item_id": "b1", "title": "History Book", "category": "books"})
        client.post("/feedback", json={"user_id": "user_bob", "item_id": "m1", "event_type": "view"})
        client.post("/feedback", json={"user_id": "user_bob", "item_id": "b1", "event_type": "view"})
        client.post("/train")
        
        # Query recommendations during evening. Movies should get a contextual boost.
        resp = client.get("/recommend/user_alice?time_of_day=evening&w_relevance=1.0&w_context=1.0")
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        assert len(recs) > 0
        # Movies item should be at the top or highly scored
        top_item = recs[0]
        assert top_item.get("item_id") == "m1"
        assert "evening" in top_item.get("explanation", "").lower()


def test_category_fatigue_and_freshness(clean_db):
    with TestClient(app) as client:
        # Seed items: old book vs fresh book
        client.post("/items", json={"item_id": "old_b", "title": "Old Book", "category": "books", "metadata": {"published_year": 1990}})
        client.post("/items", json={"item_id": "new_b", "title": "New Book", "category": "books", "metadata": {"published_year": 2026}})
        client.post("/feedback", json={"user_id": "user_alice", "item_id": "old_b", "event_type": "like"})
        client.post("/train")
        
        # Query recommendations. The fresh book should get a freshness boost.
        resp = client.get("/recommend/user_alice?w_relevance=0.0&w_freshness=1.0")
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        assert len(recs) > 0
        item_ids = [r["item_id"] for r in recs]
        # new_b should rank higher or equal to old_b due to freshness
        if "new_b" in item_ids and "old_b" in item_ids:
            assert item_ids.index("new_b") <= item_ids.index("old_b")


def test_latency_sla_telemetry(clean_db):
    with TestClient(app) as client:
        # Get metrics initially
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "p95_latency_seconds" in data
        assert "sla_telemetry" in data
        assert "estimated_memory_bytes" in data


def test_lightgcn_embeddings():
    from models.lightgcn import LightGCNTrainer
    trainer = LightGCNTrainer(emb_dim=16, epochs=2, batch_size=4)
    
    # Mock data
    feedback = [
        {"user_id": "u1", "item_id": "i1"},
        {"user_id": "u1", "item_id": "i2"},
        {"user_id": "u2", "item_id": "i2"},
    ]
    items = [
        {"item_id": "i1", "title": "Item 1"},
        {"item_id": "i2", "title": "Item 2"},
        {"item_id": "i3", "title": "Item 3"},
    ]
    
    user_emb, item_emb = trainer.train_model(feedback, items)
    assert user_emb is not None
    assert item_emb is not None
    assert user_emb.shape == (2, 16)
    assert item_emb.shape == (3, 16)


def test_neural_bandit_sampling(clean_db):
    from utils.bandit import NeuralLinearBandit
    import numpy as np
    bandit = NeuralLinearBandit(context_dim=73, feature_dim=8)
    
    context = np.random.randn(73).astype(np.float32)
    arm_id, weights = bandit.select_action(context)
    assert arm_id in range(5)
    assert len(weights) == 6  # w_rel, w_fresh, w_fatigue, w_context, w_rl, w_ssl
    
    # Perform update and verify covariance change
    old_B_trace = np.trace(bandit.B[arm_id])
    bandit.update(arm_id, context, reward=1.0)
    new_B_trace = np.trace(bandit.B[arm_id])
    assert new_B_trace > old_B_trace  # Precision matrix should gain magnitude


def test_hnsw_incremental_updates(clean_db):
    from search.faiss_index import FAISSIndex
    import numpy as np
    
    index = FAISSIndex(dimension=8)
    items = [
        {"item_id": "item_a", "title": "A"},
        {"item_id": "item_b", "title": "B"},
    ]
    
    from unittest.mock import patch
    with patch('embeddings.text.TextEmbedder.encode_batch') as mock_batch, \
         patch('embeddings.text.TextEmbedder.embed_item') as mock_item:
        mock_batch.return_value = np.random.randn(2, 8).astype(np.float32)
        mock_item.return_value = np.random.randn(8).astype(np.float32)
        
        index.build_index(items)
        assert index.index.ntotal == 2
        
        # Add a new item dynamically
        new_item = {"item_id": "item_c", "title": "C"}
        index.add_item(new_item)
        assert index.index.ntotal == 3


def test_multi_task_dwell_time():
    import torch
    from models.sasrec import SASRec
    
    # 5 items total, max seq length 10
    model = SASRec(num_items=5, hidden_dim=16, max_seq_len=10)
    seq = torch.randint(0, 5, (2, 8)) # batch size 2, length 8
    
    logits, mu, sigma = model(seq, return_dwell=True)
    assert logits.shape == (2, 8, 5)
    assert mu.shape == (2, 8, 5)
    assert sigma.shape == (2, 8, 5)
    assert torch.all(sigma > 0)


def test_fairness_pid_controller():
    from utils.fairness import FairnessAuditor
    
    auditor = FairnessAuditor(target_di=1.0, threshold=0.8, window_size=10)
    auditor.cat_minority = 2.0
    auditor.cat_majority = 8.0
    
    # Initial DI with empty impressions
    assert auditor.compute_di() == 1.0
    
    # Add only majority recommendations (underrepresenting minority category)
    # Minority category is 'books'
    auditor.record_recommendations([{"category": "movies"}] * 10)
    
    # Auditing triggers PID calculation
    di = auditor.audit_and_update_pid()
    assert di == 0.0
    assert auditor.lambda_fair > 0.0 # PID controller should produce a boost value


def test_shapley_explanations():
    from utils.explain import LinearShapleyExplainer
    
    metrics = {"relevance": 0.8, "freshness": 0.5, "fatigue": 0.2, "context": 0.9}
    baseline_metrics = {"relevance": 0.5, "freshness": 0.5, "fatigue": 0.1, "context": 0.5}
    weights = {"w_relevance": 1.0, "w_freshness": 0.2, "w_fatigue": 0.3, "w_context": 0.4}
    
    diff, phi, explanation = LinearShapleyExplainer.explain_recommendation("i1", metrics, baseline_metrics, weights)
    
    # Score diff = 1.0*(0.3) + 0.2*(0.0) - 0.3*(0.1) + 0.4*(0.4) = 0.3 - 0.03 + 0.16 = 0.43
    assert abs(diff - 0.43) < 1e-4
    assert len(explanation) > 0
    assert "relevance contributes" in explanation


def test_bcq_reinforcement_learning():
    import torch
    from models.bcq import BCQQNetwork
    
    # State dim 390, item emb dim 384, latent dim 16
    model = BCQQNetwork(state_dim=390, item_emb_dim=384, latent_dim=16)
    
    state = torch.randn(4, 390)
    item_emb = torch.randn(4, 384)
    q_vals = model(state, item_emb)
    assert q_vals.shape == (4,)
    
    # Batch action selection mode
    item_emb_batch = torch.randn(4, 5, 384)
    q_vals_batch = model(state, item_emb_batch)
    assert q_vals_batch.shape == (4, 5)


