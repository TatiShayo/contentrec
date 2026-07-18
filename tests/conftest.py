import sys
from unittest.mock import MagicMock
import numpy as np

# ---------------------------------------------------------------------------
# Global Mock for sentence_transformers to avoid downloading/loading model during tests
# ---------------------------------------------------------------------------
_MOCK_DIM = 384
mock_st_model = MagicMock()

import hashlib
import re


def _encode_one(text: str, normalize_embeddings: bool) -> np.ndarray:
    """Deterministic feature-hashing embedding.

    The previous mock seeded every call with a constant (42), so *every* text
    embedded to the same vector — which silently broke any logic that depends on
    distinct/related embeddings (the DPP onboarding selector collapsed to one
    item). A per-text random seed fixes distinctness but is non-deterministic
    across processes (PYTHONHASHSEED) and carries no notion of similarity.

    This feature-hashing scheme is (a) fully deterministic via hashlib and
    (b) semantically meaningful: texts that share tokens get similar vectors,
    so a query embedding is nearest the items that share its words. That keeps
    both cold-start ranking tests and diversity tests honest.
    """
    vec = np.zeros(_MOCK_DIM, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    if not tokens:
        vec[0] = 1.0  # stable non-zero vector for empty text
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % _MOCK_DIM] += 1.0
    if normalize_embeddings:
        vec /= np.linalg.norm(vec) + 1e-10
    return vec


def _mock_encode(text_or_texts, normalize_embeddings=True, show_progress_bar=False):
    if isinstance(text_or_texts, str):
        return _encode_one(text_or_texts, normalize_embeddings)
    vecs = np.stack(
        [_encode_one(t, normalize_embeddings) for t in text_or_texts]
    ).astype(np.float32)
    return vecs

mock_st_model.encode = _mock_encode

mock_st_module = MagicMock()
mock_st_module.SentenceTransformer = lambda *args, **kwargs: mock_st_model

sys.modules["sentence_transformers"] = mock_st_module


import pytest
import os
import tempfile
import config
from data.database import init_db


@pytest.fixture(autouse=True)
def _deterministic_seeds():
    """Seed every RNG so model training/ranking is reproducible across runs.

    The torch models (SASRec/LightGCN/BCQ/BEST-Rec) are randomly initialized and
    trained inside `/train`; without seeding, their scores perturb the final
    ranking non-deterministically and make ordering-sensitive tests flaky.
    """
    import random
    random.seed(1234)
    np.random.seed(1234)
    try:
        import torch
        torch.manual_seed(1234)
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def test_env():
    # Create a fresh temp file for each test
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    model_path = db_path + ".pkl"
    faiss_path = db_path + ".index"
    sasrec_path = db_path + "_sasrec.pt"
    sasrec_map_path = db_path + "_sasrec.pkl"
    
    original_db = config.DATABASE_PATH
    original_model = config.MODEL_PATH
    original_faiss = config.FAISS_INDEX_PATH
    original_sasrec = config.SASREC_MODEL_PATH
    original_sasrec_map = config.SASREC_MAP_PATH
    original_testing = getattr(config, "TESTING", False)
    
    config.DATABASE_PATH = db_path
    config.MODEL_PATH = model_path
    config.FAISS_INDEX_PATH = faiss_path
    config.SASREC_MODEL_PATH = sasrec_path
    config.SASREC_MAP_PATH = sasrec_map_path
    config.TESTING = True
    
    # Re-initialize the DB for the new path
    init_db()
    
    yield
    
    # Cleanup
    config.DATABASE_PATH = original_db
    config.MODEL_PATH = original_model
    config.FAISS_INDEX_PATH = original_faiss
    config.SASREC_MODEL_PATH = original_sasrec
    config.SASREC_MAP_PATH = original_sasrec_map
    config.TESTING = original_testing
    
    for path in [db_path, model_path, faiss_path, sasrec_path, sasrec_map_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass
        # Also clean up the faiss map path if it got created
        map_path = os.path.splitext(path)[0] + "_map.pkl"
        if os.path.exists(map_path):
            try:
                os.remove(map_path)
            except:
                pass

@pytest.fixture
def clean_db():
    # Already handled by test_env being autouse and function scoped
    yield
