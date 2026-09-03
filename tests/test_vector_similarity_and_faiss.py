"""Tests for vector similarity mathematics, TextEmbedder, VisionEmbedder, and FAISSIndex.

Verifies:
- Cosine similarity mathematical invariants
- Dimension checking and error handling
- Zero-vector, NaN, and Inf normalization and sanitization
- FAISS insertion, update, query top-k, and persistence
"""

import numpy as np
import pytest
import os
import tempfile
from embeddings.text import TextEmbedder, EMBEDDING_DIM
from embeddings.vision import VisionEmbedder
from search.faiss_index import FAISSIndex


class TestVectorMathAndEmbeddings:
    def test_text_embedder_singleton(self):
        e1 = TextEmbedder()
        e2 = TextEmbedder()
        assert e1 is e2
        assert hasattr(e1, "dimension")
        assert e1.dimension == EMBEDDING_DIM

    def test_text_embedder_encode_shape_and_norm(self):
        embedder = TextEmbedder()
        vec = embedder.encode("Machine learning and recommendation algorithms")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (EMBEDDING_DIM,)
        assert vec.dtype == np.float32
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-4)

    def test_text_embedder_encode_empty_text(self):
        embedder = TextEmbedder()
        vec = embedder.encode("")
        assert vec.shape == (EMBEDDING_DIM,)
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-4)
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()

    def test_text_embedder_encode_batch(self):
        embedder = TextEmbedder()
        texts = ["Artificial intelligence", "Database systems", "Data science"]
        vecs = embedder.encode_batch(texts)
        assert vecs.shape == (3, EMBEDDING_DIM)
        for i in range(3):
            assert np.isclose(np.linalg.norm(vecs[i]), 1.0, atol=1e-4)

    def test_text_embedder_encode_batch_empty(self):
        embedder = TextEmbedder()
        vecs = embedder.encode_batch([])
        assert vecs.shape == (0, EMBEDDING_DIM)

    def test_text_embedder_embed_item(self):
        embedder = TextEmbedder()
        item = {
            "item_id": "i100",
            "title": "Quantum Computing",
            "tags": "physics,tech",
            "category": "articles"
        }
        vec = embedder.embed_item(item)
        assert vec.shape == (EMBEDDING_DIM,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    def test_vision_embedder_singleton_and_properties(self):
        v1 = VisionEmbedder()
        v2 = VisionEmbedder()
        assert v1 is v2

    def test_vision_embedder_embed_item_image(self):
        vision = VisionEmbedder()
        item = {"item_id": "item_vis_1", "title": "Movie Poster A", "tags": "action", "category": "movies"}
        img_emb = vision.embed_item_image(item)
        assert img_emb.shape == (EMBEDDING_DIM,)
        assert np.isclose(np.linalg.norm(img_emb), 1.0, atol=1e-4)
        assert not np.isnan(img_emb).any()

    def test_vision_embedder_determinism(self):
        vision = VisionEmbedder()
        item = {"item_id": "deterministic_item", "title": "SciFi Book", "tags": "space"}
        emb1 = vision.embed_item_image(item)
        emb2 = vision.embed_item_image(item)
        np.testing.assert_array_almost_equal(emb1, emb2)


class TestFAISSIndexCore:
    def test_faiss_init_empty(self):
        idx = FAISSIndex()
        assert idx.dimension == EMBEDDING_DIM
        assert idx.index.ntotal == 0
        assert idx.search(np.zeros(EMBEDDING_DIM), n=5) == []

    def test_faiss_build_and_search(self):
        idx = FAISSIndex()
        items = [
            {"item_id": "item_a", "title": "Python for Data Analysis", "tags": "python,data", "category": "books"},
            {"item_id": "item_b", "title": "Cooking Masterclass", "tags": "food,cooking", "category": "videos"},
            {"item_id": "item_c", "title": "Deep Learning with PyTorch", "tags": "python,ai,deep learning", "category": "books"},
        ]
        idx.build_index(items)
        assert idx.index.ntotal == 3

        # Search for Python books
        results = idx.search_by_text("python machine learning", n=2)
        assert len(results) == 2
        assert all("item_id" in r and "score" in r for r in results)
        # Scores should be in valid cosine similarity range [-1.0, 1.0]
        for r in results:
            assert -1.0 <= r["score"] <= 1.01

    def test_faiss_dimension_mismatch_validation(self):
        idx = FAISSIndex()
        items = [{"item_id": "item_1", "title": "Test 1", "tags": "tag1"}]
        idx.build_index(items)

        # 1D wrong dimension
        wrong_1d = np.zeros(128, dtype=np.float32)
        with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
            idx.search(wrong_1d, n=5)

        # 2D wrong dimension
        wrong_2d = np.zeros((1, 512), dtype=np.float32)
        with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
            idx.search(wrong_2d, n=5)

    def test_faiss_nan_and_inf_query_sanitization(self):
        idx = FAISSIndex()
        items = [
            {"item_id": "i1", "title": "First Item", "tags": "tagA"},
            {"item_id": "i2", "title": "Second Item", "tags": "tagB"},
        ]
        idx.build_index(items)

        nan_query = np.full(EMBEDDING_DIM, np.nan, dtype=np.float32)
        results = idx.search(nan_query, n=2)
        assert len(results) == 2
        assert not any(np.isnan(r["score"]) for r in results)

        inf_query = np.full(EMBEDDING_DIM, np.inf, dtype=np.float32)
        results = idx.search(inf_query, n=2)
        assert len(results) == 2
        assert not any(np.isinf(r["score"]) for r in results)

    def test_faiss_zero_vector_query(self):
        idx = FAISSIndex()
        items = [{"item_id": "i1", "title": "First Item", "tags": "tagA"}]
        idx.build_index(items)

        zero_query = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        results = idx.search(zero_query, n=1)
        assert len(results) == 1
        assert not np.isnan(results[0]["score"])

    def test_faiss_top_k_bounds(self):
        idx = FAISSIndex()
        items = [
            {"item_id": f"item_{i}", "title": f"Title {i}", "tags": f"tag_{i}"}
            for i in range(5)
        ]
        idx.build_index(items)

        # n <= 0 returns empty list
        assert idx.search(np.ones(EMBEDDING_DIM), n=0) == []
        assert idx.search(np.ones(EMBEDDING_DIM), n=-5) == []
        assert idx.search_by_text("test", n=0) == []

        # n larger than total items
        results = idx.search(np.ones(EMBEDDING_DIM), n=100)
        assert len(results) == 5

    def test_faiss_add_and_update_item(self):
        idx = FAISSIndex()
        item1 = {"item_id": "up_1", "title": "Initial Title", "tags": "science"}
        idx.add_item(item1)
        assert idx.index.ntotal == 1

        res = idx.search_by_text("Initial Title", n=1)
        assert len(res) == 1
        assert res[0]["item_id"] == "up_1"

        # Update same item ID with new content
        item1_updated = {"item_id": "up_1", "title": "Updated Title History", "tags": "history"}
        idx.add_item(item1_updated)
        assert idx.index.ntotal >= 1

        res2 = idx.search_by_text("Updated Title History", n=1)
        assert len(res2) == 1
        assert res2[0]["item_id"] == "up_1"

    def test_faiss_save_and_load_persistence(self, tmp_path):
        idx_path = str(tmp_path / "test_faiss.index")
        idx = FAISSIndex()
        items = [
            {"item_id": "p1", "title": "Persistence Test 1", "tags": "tag1"},
            {"item_id": "p2", "title": "Persistence Test 2", "tags": "tag2"},
        ]
        idx.build_index(items)
        idx.save(idx_path)

        assert os.path.exists(idx_path)
        json_map = str(tmp_path / "test_faiss_map.json")
        assert os.path.exists(json_map)

        # Load into new index instance
        loaded_idx = FAISSIndex()
        loaded_idx.load(idx_path)
        assert loaded_idx.index.ntotal == 2
        results = loaded_idx.search_by_text("Persistence Test 1", n=2)
        assert len(results) == 2
        assert results[0]["item_id"] in ["p1", "p2"]
