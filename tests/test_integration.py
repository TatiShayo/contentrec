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
    client.post("/train")
    
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
