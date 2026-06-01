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
    
    original_db = config.DATABASE_PATH
    original_model = config.MODEL_PATH
    
    config.DATABASE_PATH = db_path
    config.MODEL_PATH = model_path
    
    # Re-initialize the DB for the new path
    init_db()
    
    yield
    
    # Cleanup
    config.DATABASE_PATH = original_db
    config.MODEL_PATH = original_model
    
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except:
            pass
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
        except:
            pass

@pytest.fixture
def clean_db():
    # Already handled by test_env being autouse and function scoped
    yield
