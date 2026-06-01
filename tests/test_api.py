import pytest
from fastapi.testclient import TestClient
from main import app
from data.database import init_db
import os
import config

@pytest.fixture
def client(clean_db):
    with TestClient(app) as c:
        yield c

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_items_api(client):
    resp = client.post("/items", json={
        "item_id": "i1",
        "title": "Title 1",
        "tags": "tag1",
        "category": "cat1"
    })
    assert resp.status_code == 200
    
    resp = client.get("/items/i1")
    assert resp.status_code == 200
    assert resp.json()["item_id"] == "i1"

def test_feedback_api(client):
    client.post("/items", json={"item_id": "i1", "title": "T1"})
    resp = client.post("/feedback", json={
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "view"
    })
    assert resp.status_code == 200
    
    # Test invalid event type
    resp = client.post("/feedback", json={
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "invalid"
    })
    assert resp.status_code == 400

def test_recommend_api(client):
    client.post("/items", json={"item_id": "i1", "title": "T1", "tags": "tech"})
    client.post("/feedback", json={"user_id": "u1", "item_id": "i1", "event_type": "view"})
    
    # Test cold start rec
    resp = client.get("/recommend/u2?features=tech")
    assert resp.status_code == 200
    assert len(resp.json()["recommendations"]) > 0

def test_stats_api(client):
    client.post("/items", json={"item_id": "i1", "title": "T1"})
    client.post("/feedback", json={"user_id": "u1", "item_id": "i1", "event_type": "view"})
    
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["item_count"] == 1
    assert data["feedback_count"] == 1
    assert data["user_count"] == 1
