import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient
from main import app
import config
from data.database import get_db_connection, init_db, add_impression
from models.causal import PropensityEstimator
from utils.dpp import DPPSelector
from embeddings.vision import VisionEmbedder
from search.faiss_index import FAISSIndex
from models.best_rec import BESTRec, BESTRecTrainer
from utils.surprise import SurpriseController
from models.sequential_train import OnlineSequentialTrainer

@pytest.fixture
def test_client(clean_db):
    with TestClient(app) as c:
        yield c

def test_causal_propensity_estimator(clean_db):
    """1. Test causal propensity model training, prediction, and IPS weight calculation."""
    estimator = PropensityEstimator()
    
    # Check encoding
    vec = estimator._encode_features(cohort="B", device="mobile", time_of_day="evening", category="movies")
    assert vec.shape == (13,)
    assert vec[0] == 1.0  # Cohort B
    assert vec[1] == 1.0  # Mobile
    assert vec[6] == 1.0  # Evening
    assert vec[8] == 1.0  # Movies idx is 0
    
    # Try prediction without training (should return default sigmoid logits)
    prop = estimator.predict_propensity(cohort="A", device="desktop", time_of_day="morning", category="music")
    assert 0.05 <= prop <= 0.95
    
    # IPS weights should be clipped
    w_click = estimator.get_ips_weight(cohort="A", device="desktop", time_of_day="morning", category="music", clicked=True)
    w_non_click = estimator.get_ips_weight(cohort="A", device="desktop", time_of_day="morning", category="music", clicked=False)
    assert 0.1 <= w_click <= 10.0
    assert 0.1 <= w_non_click <= 10.0

    # Seed impressions to test training
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO items (item_id, title, category) VALUES ('i1', 'Item 1', 'movies')")
        cursor.execute("INSERT INTO items (item_id, title, category) VALUES ('i2', 'Item 2', 'music')")
        cursor.execute("INSERT INTO items (item_id, title, category) VALUES ('i3', 'Item 3', 'books')")
        conn.commit()
        
    add_impression("u1", "i1", "B", '{"device": "mobile", "time_of_day": "evening"}')
    add_impression("u2", "i2", "A", '{"device": "desktop", "time_of_day": "morning"}')
    
    loss = estimator.train_model(epochs=2, batch_size=2)
    assert loss >= 0.0
    
    # Predict should utilize cache
    prop_cached = estimator.predict_propensity(cohort="B", device="mobile", time_of_day="evening", category="movies")
    assert (cohort := "B", device := "mobile", time_of_day := "evening", category := "movies") in estimator.propensity_cache

def test_dpp_onboarding_quiz_and_submit(test_client):
    """2. Test interactive DPP onboarding endpoints and synthetic rating submission."""
    # Seed some items to create a pool
    categories = ["movies", "music", "books", "articles", "news"]
    for i in range(20):
        test_client.post("/items", json={
            "item_id": f"item_{i}",
            "title": f"Title {i}",
            "category": categories[i % 5],
            "tags": f"tag_{i}"
        })
        
    # Trigger /onboarding/quiz
    resp = test_client.post("/onboarding/quiz", json={"n_quiz": 6})
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) == 6
    
    # Submit onboarding quiz answers
    submit_payload = {
        "user_id": "cold_user_123",
        "ratings": {
            "item_0": 1.0,
            "item_1": -1.0,
            "item_2": 0.0
        }
    }
    resp_submit = test_client.post("/onboarding/submit", json=submit_payload)
    assert resp_submit.status_code == 200
    submit_data = resp_submit.json()
    assert submit_data["status"] == "ok"
    assert submit_data["processed_ratings"] == 3
    
    # Check that interactions were saved in DB
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE user_id='cold_user_123'")
        cnt = cursor.fetchone()[0]
        assert cnt == 3

def test_query_intent_parsing_negations(test_client):
    """3. Test query steering parsing with negation and category exclusion."""
    # Seed items with different categories
    test_client.post("/items", json={"item_id": "movie_1", "title": "Funny Movie", "category": "movies"})
    test_client.post("/items", json={"item_id": "music_1", "title": "Classical Music", "category": "music"})
    test_client.post("/items", json={"item_id": "book_1", "title": "SciFi Novel", "category": "books"})
    
    test_client.post("/feedback", json={"user_id": "user_q", "item_id": "movie_1", "event_type": "view"})
    test_client.post("/feedback", json={"user_id": "user_q", "item_id": "music_1", "event_type": "view"})
    test_client.post("/feedback", json={"user_id": "user_q", "item_id": "book_1", "event_type": "view"})
    
    test_client.post("/train")
    
    # Request recommendations with query excluding music: "funny stuff but no music"
    resp = test_client.get("/recommend/user_q?query=funny+stuff+but+no+music&w_relevance=1.0")
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]
    
    # Verify that the recommended list has NO items from music category
    item_ids = [r["item_id"] for r in recs]
    # Check categories of recommended items
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT category FROM items WHERE item_id IN ({','.join(['?']*len(item_ids))})", item_ids)
        cats = [row[0].lower() for row in cursor.fetchall()]
        assert "music" not in cats

def test_online_sequential_sgd():
    """4. Test online real-time sequential learning queue consumer."""
    # Build a simple SASRec model
    from models.sasrec import SASRec
    model_train = SASRec(num_items=10, hidden_dim=8, max_seq_len=5, num_heads=2, num_blocks=1)
    model_serve = SASRec(num_items=10, hidden_dim=8, max_seq_len=5, num_heads=2, num_blocks=1)
    
    trainer = OnlineSequentialTrainer(model_train, model_serve, lr=1e-3)
    
    # Add samples to queue
    trainer.add_sample(seq_indices=[1, 2, 3], next_idx=4, dwell_time=20.0)
    trainer.add_sample(seq_indices=[2, 3, 4], next_idx=5, dwell_time=15.0)
    
    # Process queue and perform step
    assert trainer.queue.qsize() == 2
    loss = trainer.process_queue_and_step()
    
    # Buffer should consume and loss should be returned
    assert trainer.queue.qsize() == 0
    assert len(trainer.replay_buffer) == 2
    assert loss > 0.0

def test_late_fusion_clip_faiss(clean_db):
    """5. Test CLIP-like late-fusion FAISS indexing."""
    # Seed items
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO items (item_id, title, category, tags) VALUES ('i1', 'Cover Art Movie', 'movies', 'action')")
        cursor.execute("INSERT INTO items (item_id, title, category, tags) VALUES ('i2', 'Spectrogram Song', 'music', 'pop')")
        conn.commit()
        
    embedder = VisionEmbedder()
    # Check vision embedder output shape (dim 384 SBERT baseline)
    img_emb = embedder.embed_item_image({"item_id": "i1", "title": "Cover Art Movie"})
    assert img_emb.shape == (384,)
    
    # Check late-fusion build in FAISSIndex
    index = FAISSIndex(dimension=384)
    items = [
        {"item_id": "i1", "title": "Cover Art Movie", "category": "movies", "tags": "action"},
        {"item_id": "i2", "title": "Spectrogram Song", "category": "music", "tags": "pop"}
    ]
    index.build_index(items)
    
    # Verify search executes
    query_emb = np.random.randn(384).astype(np.float32)
    recs = index.search(query_emb, n=2)
    assert len(recs) > 0

def test_best_rec_autoencoder_pretraining():
    """6. Test BEST-Rec universal autoencoder pretraining tasks."""
    model = BESTRec(num_items=20, hidden_dim=16)
    trainer = BESTRecTrainer(model, lr=1e-3)
    
    # Create synthetic user item sequences
    sequences = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6, 7],
        [1, 5, 3]
    ]
    
    # Mapping of item index to category index (0-4)
    item_to_cat_idx = {i: i % 5 for i in range(20)}
    
    # Train one epoch and check loss decreases or is calculated
    loss = trainer.train_epoch(sequences, item_to_cat_idx)
    assert loss > 0.0
    
    # Test state dict saving and loading
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        trainer.save(tmp_name)
        assert os.path.exists(tmp_name)
        
        # Load in new trainer
        new_model = BESTRec(num_items=20, hidden_dim=16)
        new_trainer = BESTRecTrainer(new_model)
        new_trainer.load(tmp_name)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def test_dirichlet_pid_surprise():
    """7. Test Bayesian category surprise Dirichlet preference updates and PID lambda adjustments."""
    controller = SurpriseController(target_kl=0.5, kp=0.5)
    
    user_feedback = [
        {"item_id": "i1"},
        {"item_id": "i1"},
        {"item_id": "i2"}
    ]
    item_details = {
        "i1": {"category": "movies"},
        "i2": {"category": "books"},
        "i3": {"category": "music"}
    }
    
    # Check Dirichlet counts (alpha: movie=3, book=2, music=1, articles=1, news=1)
    alpha = controller.get_user_dirichlet_prior(user_feedback, item_details)
    assert alpha[0] == 3.0  # movies idx 0: 1 (prior) + 2 feedback
    assert alpha[2] == 2.0  # books idx 2: 1 (prior) + 1 feedback
    
    # Base diversity lambda is 0.8
    # Candidates are all movies (highly predictable, low surprise -> should increase diversity (lower lambda))
    candidates = [
        {"item_id": "i1"},
        {"item_id": "i1"},
        {"item_id": "i1"}
    ]
    
    adj_lambda, kl, err = controller.adjust_diversity_lambda(
        base_lambda=0.8,
        user_feedback=user_feedback,
        candidates=candidates,
        item_details=item_details
    )
    
    assert kl >= 0.0
    assert 0.0 <= adj_lambda <= 1.0
