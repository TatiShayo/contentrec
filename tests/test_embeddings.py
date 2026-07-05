"""
Tests for Phase 2: embeddings and FAISS search.

Tests are designed to work even when sentence-transformers is not installed
by mocking the model with deterministic random embeddings.
"""

import os
import sys
import tempfile
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock sentence-transformers *before* importing our modules so that
# tests pass on CI / environments where the package is not installed.
# ---------------------------------------------------------------------------

_MOCK_DIM = 384


def _make_mock_model():
    """Return a mock SentenceTransformer whose encode() returns random vectors."""
    model = MagicMock()

    def _encode(text_or_texts, normalize_embeddings=True, show_progress_bar=False):
        rng = np.random.RandomState(42)
        if isinstance(text_or_texts, str):
            vec = rng.randn(_MOCK_DIM).astype(np.float32)
            if normalize_embeddings:
                vec /= np.linalg.norm(vec) + 1e-10
            return vec
        vecs = rng.randn(len(text_or_texts), _MOCK_DIM).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            vecs /= norms
        return vecs

    model.encode = _encode
    return model


# Patch at module level so every TextEmbedder instance uses the mock.
_mock_st_module = MagicMock()
_mock_st_module.SentenceTransformer = _make_mock_model


@pytest.fixture(autouse=True)
def _patch_sentence_transformers():
    """Ensure sentence_transformers is mocked for every test."""
    with patch.dict(sys.modules, {"sentence_transformers": _mock_st_module}):
        # Reset TextEmbedder singleton so each test starts fresh
        from embeddings.text import TextEmbedder

        with TextEmbedder._instance_lock:
            TextEmbedder._instance = None

        embedder = TextEmbedder()
        # Force-inject the mock model so lazy loading is bypassed
        embedder._model = _make_mock_model()
        yield


# ===================================================================
# TextEmbedder tests
# ===================================================================


class TestTextEmbedder:
    """Tests for embeddings.text.TextEmbedder."""

    def test_singleton(self):
        """Two calls to TextEmbedder() return the same object."""
        from embeddings.text import TextEmbedder

        a = TextEmbedder()
        b = TextEmbedder()
        assert a is b

    def test_encode_shape(self):
        """encode() returns a 1-D vector of the expected dimension."""
        from embeddings.text import TextEmbedder

        vec = TextEmbedder().encode("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (_MOCK_DIM,)

    def test_encode_batch_shape(self):
        """encode_batch() returns a 2-D array with correct shape."""
        from embeddings.text import TextEmbedder

        texts = ["hello", "world", "foo"]
        vecs = TextEmbedder().encode_batch(texts)
        assert vecs.shape == (3, _MOCK_DIM)

    def test_encode_batch_empty(self):
        """encode_batch([]) returns an empty array with correct columns."""
        from embeddings.text import TextEmbedder

        vecs = TextEmbedder().encode_batch([])
        assert vecs.shape == (0, _MOCK_DIM)

    def test_embed_item(self):
        """embed_item() works with a typical item dict."""
        from embeddings.text import TextEmbedder

        item = {"item_id": "1", "title": "ML Guide", "tags": "ai,ml", "category": "tech"}
        vec = TextEmbedder().embed_item(item)
        assert vec.shape == (_MOCK_DIM,)

    def test_embed_item_missing_fields(self):
        """embed_item() handles items with missing optional fields."""
        from embeddings.text import TextEmbedder

        item = {"item_id": "2", "title": "Untitled"}
        vec = TextEmbedder().embed_item(item)
        assert vec.shape == (_MOCK_DIM,)

    def test_thread_safety(self):
        """Multiple threads can call encode() concurrently without error."""
        from embeddings.text import TextEmbedder

        embedder = TextEmbedder()
        results = {}
        errors = []

        def worker(thread_id):
            try:
                results[thread_id] = embedder.encode(f"thread {thread_id}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 8


# ===================================================================
# FAISSIndex tests
# ===================================================================


class TestFAISSIndex:
    """Tests for search.faiss_index.FAISSIndex."""

    @pytest.fixture()
    def sample_items(self):
        return [
            {"item_id": "a1", "title": "Introduction to Python", "tags": "python,beginner", "category": "programming"},
            {"item_id": "a2", "title": "Advanced Machine Learning", "tags": "ml,ai", "category": "data-science"},
            {"item_id": "a3", "title": "Web Development with React", "tags": "react,javascript", "category": "web"},
            {"item_id": "a4", "title": "Deep Learning Fundamentals", "tags": "deep-learning,ai", "category": "data-science"},
            {"item_id": "a5", "title": "Database Design Patterns", "tags": "sql,databases", "category": "backend"},
        ]

    def _make_index(self):
        from search.faiss_index import FAISSIndex

        return FAISSIndex()

    def test_build_index(self, sample_items):
        """build_index populates the FAISS index with the correct count."""
        idx = self._make_index()
        idx.build_index(sample_items)
        assert idx.index.ntotal == len(sample_items)

    def test_build_index_empty(self):
        """build_index with no items leaves the index empty."""
        idx = self._make_index()
        idx.build_index([])
        assert idx.index.ntotal == 0

    def test_search_returns_results(self, sample_items):
        """search() returns results with item_id and score keys."""
        idx = self._make_index()
        idx.build_index(sample_items)
        query_vec = np.random.randn(_MOCK_DIM).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)
        results = idx.search(query_vec, n=3)
        assert len(results) <= 3
        for r in results:
            assert "item_id" in r
            assert "score" in r

    def test_search_n_capped(self, sample_items):
        """Requesting more results than items returns at most ntotal items."""
        idx = self._make_index()
        idx.build_index(sample_items)
        query_vec = np.random.randn(_MOCK_DIM).astype(np.float32)
        results = idx.search(query_vec, n=100)
        assert len(results) <= len(sample_items)

    def test_search_empty_index(self):
        """search() on an empty index returns an empty list."""
        idx = self._make_index()
        results = idx.search(np.zeros(_MOCK_DIM, dtype=np.float32))
        assert results == []

    def test_search_by_text(self, sample_items):
        """search_by_text() encodes the query and returns results."""
        idx = self._make_index()
        idx.build_index(sample_items)
        results = idx.search_by_text("python programming", n=2)
        assert len(results) <= 2

    def test_add_item(self, sample_items):
        """add_item() grows the index by one."""
        idx = self._make_index()
        idx.build_index(sample_items)
        before = idx.index.ntotal
        idx.add_item({"item_id": "new1", "title": "New Article", "tags": "new", "category": "misc"})
        assert idx.index.ntotal == before + 1

    def test_save_and_load(self, sample_items):
        """save() + load() round-trips the index and mapping."""
        idx = self._make_index()
        idx.build_index(sample_items)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.index")
            idx.save(path)

            idx2 = self._make_index()
            idx2.load(path)

            assert idx2.index.ntotal == idx.index.ntotal
            assert idx2._row_to_id == idx._row_to_id
            assert idx2._id_to_row == idx._id_to_row

    def test_thread_safety(self, sample_items):
        """Concurrent search calls do not crash."""
        idx = self._make_index()
        idx.build_index(sample_items)
        errors = []

        def worker():
            try:
                idx.search_by_text("test query", n=2)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ===================================================================
# Search API tests
# ===================================================================


class TestSearchAPI:
    """Tests for api.search endpoints."""

    @pytest.fixture()
    def client(self):
        """Create a test client with a mocked FAISS index on app.state."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.search import router

        app = FastAPI()
        app.include_router(router)

        # Create a mock FAISS index
        mock_index = MagicMock()
        mock_index.search_by_text.return_value = [
            {"item_id": "x1", "score": 0.95},
            {"item_id": "x2", "score": 0.88},
        ]
        app.state.faiss_index = mock_index

        return TestClient(app)

    def test_post_search(self, client):
        """POST /search returns expected structure."""
        resp = client.post("/search", json={"query": "machine learning", "n": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "machine learning"
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 2
        assert data["results"][0]["item_id"] == "x1"

    def test_post_search_default_n(self, client):
        """POST /search works with default n."""
        resp = client.post("/search", json={"query": "test"})
        assert resp.status_code == 200

    def test_get_search(self, client):
        """GET /search?q=... returns expected structure."""
        resp = client.get("/search", params={"q": "python", "n": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "python"
        assert isinstance(data["results"], list)

    def test_get_search_missing_query(self, client):
        """GET /search without q= returns 422."""
        resp = client.get("/search")
        assert resp.status_code == 422

    def test_post_search_empty_query(self, client):
        """POST /search with empty query still succeeds."""
        resp = client.post("/search", json={"query": ""})
        assert resp.status_code == 200
