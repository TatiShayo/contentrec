import json
from data.database import get_db_connection

def add_item(item_id, title, tags="", category="", metadata=None):
    metadata_json = json.dumps(metadata) if metadata else None
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO items (item_id, title, tags, category, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                title=excluded.title,
                tags=excluded.tags,
                category=excluded.category,
                metadata_json=excluded.metadata_json
        ''', (item_id, title, tags, category, metadata_json))
        conn.commit()

def get_item(item_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM items WHERE item_id = ?", (item_id,))
        row = cursor.fetchone()
        if row:
            item = dict(row)
            if item['metadata_json']:
                item['metadata'] = json.loads(item['metadata_json'])
            else:
                item['metadata'] = None
            return item
        return None

def get_all_items():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM items")
        rows = cursor.fetchall()
        items = []
        for row in rows:
            item = dict(row)
            if item['metadata_json']:
                item['metadata'] = json.loads(item['metadata_json'])
            else:
                item['metadata'] = None
            items.append(item)
        return items

def search_by_tags(query):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM items WHERE tags LIKE ?", (f'%{query}%',))
        rows = cursor.fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item['metadata'] = json.loads(item['metadata_json']) if item['metadata_json'] else None
            items.append(item)
        return items

def get_items_by_ids(item_ids):
    if not item_ids:
        return {}
    placeholders = ",".join("?" for _ in item_ids)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM items WHERE item_id IN ({placeholders})", list(item_ids))
        rows = cursor.fetchall()
        result = {}
        for row in rows:
            item = dict(row)
            item['metadata'] = json.loads(item['metadata_json']) if item['metadata_json'] else None
            result[item['item_id']] = item
        return result

