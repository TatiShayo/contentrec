import sys
from unittest.mock import MagicMock
import numpy as np

# ---------------------------------------------------------------------------
# Global Mock for sentence_transformers to avoid downloading/loading model during tests
# ---------------------------------------------------------------------------
_MOCK_DIM = 384
mock_st_model = MagicMock()

def _mock_encode(text_or_texts, normalize_embeddings=True, show_progress_bar=False):
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
