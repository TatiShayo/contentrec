import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client(clean_db):
    with TestClient(app) as c:
        yield c

def test_full_pipeline(client):
    # 1. Seed items
    items = [
        {"item_id": "movie1", "title": "Action Movie", "tags": "action,adventure"},
        {"item_id": "movie2", "title": "Romance Movie", "tags": "romance,drama"},
        {"item_id": "movie3", "title": "Action Sequel", "tags": "action,thriller"},
    ]
    for item in items:
        client.post("/items", json=item)
        
    # 2. Add feedback (User 1 likes action)
    client.post("/feedback", json={"user_id": "u1", "item_id": "movie1", "event_type": "like"})
    client.post("/feedback", json={"user_id": "u1", "item_id": "movie3", "event_type": "view"})
    
    # 3. Train
    # POST to /train triggers training in the background. Since background tasks run 
    # after the response is sent, we should train directly or trigger background tasks 
    # execution if TestClient allows, or simply run app.state.engine.train() directly 
    # to guarantee synchronous execution for the test.
    app.state.engine.train()
    # Synchronize the index reference
    app.state.faiss_index = app.state.engine.faiss_index
    
    # 4. Verify stats
    stats = client.get("/stats").json()
    assert stats["item_count"] == 3
    assert stats["feedback_count"] == 2
    
    # 5. Get recommendations for u1
    resp = client.get("/recommend/u1")
    recs = resp.json()["recommendations"]
    assert len(recs) > 0
    
    # 6. Cold start for u2
    resp = client.get("/recommend/u2?features=romance")
    recs = resp.json()["recommendations"]
    assert any(r["item_id"] == "movie2" for r in recs)

    # 7. Semantic Search (FAISS)
    search_resp = client.post("/search", json={"query": "action adventure", "n": 2})
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert len(search_data["results"]) > 0
    
    search_get_resp = client.get("/search", params={"q": "romance", "n": 2})
    assert search_get_resp.status_code == 200
    assert len(search_get_resp.json()["results"]) > 0

    # 8. Sequential Recommendation (SASRec)
    seq_resp = client.get("/sequential/u1?n=2")
    assert seq_resp.status_code == 200
    seq_data = seq_resp.json()
    assert seq_data["user_id"] == "u1"
    assert len(seq_data["recommendations"]) > 0
