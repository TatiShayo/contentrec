import pytest
import os
from models.engine import RecommendationEngine
from data.items import add_item
from data.feedback import add_feedback
import config

def test_engine_init(clean_db):
    from models.engine import LIGHTFM_AVAILABLE
    engine = RecommendationEngine()
    if LIGHTFM_AVAILABLE:
        assert engine.model is not None
    else:
        assert engine.model is None

def test_train_and_recommend(clean_db):
    # Seed data
    for i in range(10):
        add_item(f"item{i}", f"Title {i}", "tag1,tag2" if i % 2 == 0 else "tag3")
        
    for i in range(5):
        add_feedback("user1", f"item{i}", "view")
        
    engine = RecommendationEngine()
    engine.train()
    
    recs = engine.recommend("user1", n=3)
    assert len(recs) == 3
    assert "item_id" in recs[0]
    assert "score" in recs[0]

def test_cold_start(clean_db):
    add_item("item1", "T1", "tech")
    add_item("item2", "T2", "sports")
    
    engine = RecommendationEngine()
    # New user with no feedback
    recs = engine.recommend("new_user", n=5, features="tech")
    assert len(recs) > 0
    assert recs[0]["item_id"] == "item1"

def test_similar_items(clean_db):
    from models.engine import LIGHTFM_AVAILABLE
    add_item("item1", "T1", "a,b")
    add_item("item2", "T2", "a,b")
    add_item("item3", "T3", "x,y")
    add_feedback("u1", "item1", "v")
    add_feedback("u2", "item2", "v")
    
    engine = RecommendationEngine()
    engine.train()
    
    similar = engine.similar_items("item1", n=2)
    if LIGHTFM_AVAILABLE:
        assert len(similar) > 0
    else:
        assert len(similar) == 0
