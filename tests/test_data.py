import pytest
from data.items import add_item, get_item, get_all_items, search_by_tags
from data.feedback import add_feedback, get_user_feedback, get_all_feedback, get_feedback_count

def test_items_crud(clean_db):
    add_item("item1", "Title 1", "tag1,tag2", "cat1", {"key": "val"})
    item = get_item("item1")
    assert item["item_id"] == "item1"
    assert item["title"] == "Title 1"
    assert item["tags"] == "tag1,tag2"
    assert item["metadata"]["key"] == "val"

    add_item("item1", "Updated Title")
    item = get_item("item1")
    assert item["title"] == "Updated Title"

    items = get_all_items()
    assert len(items) == 1

def test_search_by_tags(clean_db):
    add_item("item1", "T1", "apple,banana")
    add_item("item2", "T2", "banana,cherry")
    
    results = search_by_tags("banana")
    assert len(results) == 2
    
    results = search_by_tags("apple")
    assert len(results) == 1
    assert results[0]["item_id"] == "item1"

def test_feedback(clean_db):
    add_feedback("user1", "item1", "view")
    add_feedback("user1", "item2", "like")
    add_feedback("user2", "item1", "click")
    
    assert get_feedback_count() == 3
    
    user1_feedback = get_user_feedback("user1")
    assert len(user1_feedback) == 2
    
    all_feedback = get_all_feedback()
    assert len(all_feedback) == 3

def test_empty_db(clean_db):
    assert get_item("nonexistent") is None
    assert get_all_items() == []
    assert get_user_feedback("user") == []
    assert get_feedback_count() == 0
