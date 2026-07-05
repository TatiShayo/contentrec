import sqlite3
import os
from contextlib import contextmanager
import config

def init_db():
    db_path = config.DATABASE_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                tags TEXT,
                category TEXT,
                metadata_json TEXT
            )
        ''')
        
        # Ensure image_embedding column exists in case db was already created
        cursor.execute("PRAGMA table_info(items)")
        item_cols = [row[1] for row in cursor.fetchall()]
        if 'image_embedding' not in item_cols:
            cursor.execute("ALTER TABLE items ADD COLUMN image_embedding TEXT")
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                dwell_time REAL DEFAULT 0.0
            )
        ''')
        
        # Ensure dwell_time column exists in case db was already created
        cursor.execute("PRAGMA table_info(feedback)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'dwell_time' not in columns:
            cursor.execute("ALTER TABLE feedback ADD COLUMN dwell_time REAL DEFAULT 0.0")

        # Create impressions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS impressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                cohort TEXT,
                context_json TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create bandit states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bandit_states (
                arm_id TEXT PRIMARY KEY,
                state_json TEXT
            )
        ''')
        conn.commit()

@contextmanager
def get_db_connection():
    db_path = config.DATABASE_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def add_impression(user_id: str, item_id: str, cohort: str, context_json: str = None) -> int:
    """Record an item impression for causal propensity estimation."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO impressions (user_id, item_id, cohort, context_json) VALUES (?, ?, ?, ?)",
            (user_id, item_id, cohort, context_json)
        )
        conn.commit()
        return cursor.lastrowid


def get_all_impressions():
    """Retrieve all recorded impressions."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM impressions ORDER BY timestamp DESC")
        return [dict(row) for row in cursor.fetchall()]

