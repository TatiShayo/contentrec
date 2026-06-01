import sqlite3
from datetime import datetime
from data.database import get_db_connection

def add_feedback(user_id, item_id, event_type, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (user_id, item_id, event_type, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, item_id, event_type, timestamp)
        )
        conn.commit()
        return cursor.lastrowid

def get_user_feedback(user_id, limit=100):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM feedback WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
        return [dict(row) for row in cursor.fetchall()]

def get_all_feedback():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback")
        return [dict(row) for row in cursor.fetchall()]

def get_feedback_count():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feedback")
        return cursor.fetchone()[0]
