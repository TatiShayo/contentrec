import pytest
import numpy as np
from embeddings.cache import EmbeddingCache

def test_embedding_cache_put_get():
    cache = EmbeddingCache(max_size=3)
    emb1 = np.random.randn(384).astype(np.float32)
    cache.put("query 1", emb1)
    
    retrieved = cache.get("query 1")
    assert retrieved is not None
    assert np.allclose(retrieved, emb1)
    assert cache.hits == 1
    assert cache.misses == 0

def test_embedding_cache_lru_eviction():
    cache = EmbeddingCache(max_size=2)
    e1 = np.ones(384, dtype=np.float32)
    e2 = np.ones(384, dtype=np.float32) * 2
    e3 = np.ones(384, dtype=np.float32) * 3
    
    cache.put("a", e1)
    cache.put("b", e2)
    # Access "a" so "b" becomes least recently used
    _ = cache.get("a")
    cache.put("c", e3)
    
    assert cache.get("b") is None  # evicted
    assert cache.get("a") is not None
    assert cache.get("c") is not None
    assert cache.evictions == 1

def test_embedding_cache_stats():
    cache = EmbeddingCache(max_size=10)
    cache.put("hello", np.zeros(384))
    _ = cache.get("hello")
    _ = cache.get("world")
    
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert stats["hit_rate"] == 0.5
