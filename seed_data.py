"""
seed_data.py - Demo Dataset Seeder

Seeds the recommendation engine database with realistic demo data.
Supports two modes:
  - API mode: POSTs data to a running server via httpx (default when run as script)
  - Direct mode: Writes directly to SQLite via seed_directly()

Usage:
    # With server running on localhost:8000:
    python seed_data.py

    # From another script (no server needed):
    from seed_data import seed_directly
    seed_directly()
"""

import random
import sys

# ---------------------------------------------------------------------------
# Demo Items: 50+ items across 5 categories
# ---------------------------------------------------------------------------

ITEMS = [
    # ── Movies (12) ──────────────────────────────────────────────────────
    {"item_id": "movie_001", "title": "The Shawshank Redemption", "tags": "drama,prison,hope,classic", "category": "movies"},
    {"item_id": "movie_002", "title": "Inception", "tags": "sci-fi,thriller,dreams,mind-bending", "category": "movies"},
    {"item_id": "movie_003", "title": "The Dark Knight", "tags": "action,superhero,crime,thriller", "category": "movies"},
    {"item_id": "movie_004", "title": "Pulp Fiction", "tags": "crime,dark-comedy,cult,classic", "category": "movies"},
    {"item_id": "movie_005", "title": "Spirited Away", "tags": "animation,fantasy,japanese,family", "category": "movies"},
    {"item_id": "movie_006", "title": "The Matrix", "tags": "sci-fi,action,cyberpunk,classic", "category": "movies"},
    {"item_id": "movie_007", "title": "Parasite", "tags": "thriller,dark-comedy,social,korean", "category": "movies"},
    {"item_id": "movie_008", "title": "Interstellar", "tags": "sci-fi,space,drama,epic", "category": "movies"},
    {"item_id": "movie_009", "title": "The Godfather", "tags": "crime,drama,mafia,classic", "category": "movies"},
    {"item_id": "movie_010", "title": "Whiplash", "tags": "drama,music,intense,indie", "category": "movies"},
    {"item_id": "movie_011", "title": "Mad Max: Fury Road", "tags": "action,post-apocalyptic,cars,intense", "category": "movies"},
    {"item_id": "movie_012", "title": "Coco", "tags": "animation,family,music,mexican", "category": "movies"},

    # ── Music (11) ───────────────────────────────────────────────────────
    {"item_id": "music_001", "title": "Bohemian Rhapsody - Queen", "tags": "rock,classic,opera,anthem", "category": "music"},
    {"item_id": "music_002", "title": "Blinding Lights - The Weeknd", "tags": "pop,synth,retro,dance", "category": "music"},
    {"item_id": "music_003", "title": "Stairway to Heaven - Led Zeppelin", "tags": "rock,classic,guitar,epic", "category": "music"},
    {"item_id": "music_004", "title": "Lose Yourself - Eminem", "tags": "hip-hop,rap,motivational,intense", "category": "music"},
    {"item_id": "music_005", "title": "Shape of You - Ed Sheeran", "tags": "pop,acoustic,dance,love", "category": "music"},
    {"item_id": "music_006", "title": "Billie Jean - Michael Jackson", "tags": "pop,dance,classic,funk", "category": "music"},
    {"item_id": "music_007", "title": "Smells Like Teen Spirit - Nirvana", "tags": "grunge,rock,alternative,90s", "category": "music"},
    {"item_id": "music_008", "title": "Take Five - Dave Brubeck", "tags": "jazz,instrumental,classic,smooth", "category": "music"},
    {"item_id": "music_009", "title": "Get Lucky - Daft Punk", "tags": "electronic,dance,funk,summer", "category": "music"},
    {"item_id": "music_010", "title": "Nuvole Bianche - Ludovico Einaudi", "tags": "classical,piano,relaxing,ambient", "category": "music"},
    {"item_id": "music_011", "title": "Runaway - Kanye West", "tags": "hip-hop,experimental,emotional,epic", "category": "music"},

    # ── Books (11) ───────────────────────────────────────────────────────
    {"item_id": "book_001", "title": "Dune by Frank Herbert", "tags": "sci-fi,epic,desert,politics", "category": "books"},
    {"item_id": "book_002", "title": "1984 by George Orwell", "tags": "dystopia,classic,political,thriller", "category": "books"},
    {"item_id": "book_003", "title": "The Hobbit by J.R.R. Tolkien", "tags": "fantasy,adventure,classic,dragons", "category": "books"},
    {"item_id": "book_004", "title": "Sapiens by Yuval Noah Harari", "tags": "non-fiction,history,science,philosophy", "category": "books"},
    {"item_id": "book_005", "title": "Project Hail Mary by Andy Weir", "tags": "sci-fi,space,humor,survival", "category": "books"},
    {"item_id": "book_006", "title": "Atomic Habits by James Clear", "tags": "non-fiction,self-help,productivity,psychology", "category": "books"},
    {"item_id": "book_007", "title": "The Name of the Wind by Patrick Rothfuss", "tags": "fantasy,magic,adventure,epic", "category": "books"},
    {"item_id": "book_008", "title": "Educated by Tara Westover", "tags": "non-fiction,memoir,education,inspiring", "category": "books"},
    {"item_id": "book_009", "title": "Neuromancer by William Gibson", "tags": "sci-fi,cyberpunk,hacker,classic", "category": "books"},
    {"item_id": "book_010", "title": "The Alchemist by Paulo Coelho", "tags": "fiction,philosophical,journey,inspiring", "category": "books"},
    {"item_id": "book_011", "title": "Clean Code by Robert C. Martin", "tags": "non-fiction,programming,software,technical", "category": "books"},

    # ── Games (10) ───────────────────────────────────────────────────────
    {"item_id": "game_001", "title": "The Witcher 3: Wild Hunt", "tags": "rpg,open-world,fantasy,story", "category": "games"},
    {"item_id": "game_002", "title": "Red Dead Redemption 2", "tags": "action,open-world,western,story", "category": "games"},
    {"item_id": "game_003", "title": "Minecraft", "tags": "sandbox,creative,survival,multiplayer", "category": "games"},
    {"item_id": "game_004", "title": "Elden Ring", "tags": "rpg,action,dark-fantasy,challenging", "category": "games"},
    {"item_id": "game_005", "title": "Hades", "tags": "roguelike,action,mythology,indie", "category": "games"},
    {"item_id": "game_006", "title": "Stardew Valley", "tags": "simulation,farming,relaxing,indie", "category": "games"},
    {"item_id": "game_007", "title": "The Legend of Zelda: TOTK", "tags": "adventure,open-world,puzzle,fantasy", "category": "games"},
    {"item_id": "game_008", "title": "Celeste", "tags": "platformer,indie,challenging,pixel-art", "category": "games"},
    {"item_id": "game_009", "title": "Portal 2", "tags": "puzzle,sci-fi,humor,co-op", "category": "games"},
    {"item_id": "game_010", "title": "Civilization VI", "tags": "strategy,turn-based,history,empire", "category": "games"},

    # ── Articles (10) ────────────────────────────────────────────────────
    {"item_id": "article_001", "title": "How Neural Networks Actually Work", "tags": "technology,ai,deep-learning,tutorial", "category": "articles"},
    {"item_id": "article_002", "title": "The Future of Renewable Energy", "tags": "science,energy,environment,future", "category": "articles"},
    {"item_id": "article_003", "title": "A Beginner's Guide to Rust Programming", "tags": "technology,programming,rust,tutorial", "category": "articles"},
    {"item_id": "article_004", "title": "The Psychology of Habit Formation", "tags": "psychology,self-improvement,science,habits", "category": "articles"},
    {"item_id": "article_005", "title": "Understanding Quantum Computing", "tags": "technology,quantum,physics,future", "category": "articles"},
    {"item_id": "article_006", "title": "Building REST APIs with FastAPI", "tags": "technology,programming,python,tutorial", "category": "articles"},
    {"item_id": "article_007", "title": "The Science of Sleep", "tags": "science,health,psychology,wellness", "category": "articles"},
    {"item_id": "article_008", "title": "Investing for Beginners: A Complete Guide", "tags": "finance,investing,beginner,money", "category": "articles"},
    {"item_id": "article_009", "title": "Climate Change: What the Data Shows", "tags": "science,environment,data,climate", "category": "articles"},
    {"item_id": "article_010", "title": "Remote Work Best Practices in 2026", "tags": "productivity,remote-work,career,tips", "category": "articles"},
]

# ---------------------------------------------------------------------------
# Demo Users with personality-driven preferences
# ---------------------------------------------------------------------------

EVENT_TYPES = ["view", "like", "purchase", "watch", "click"]

# Weight maps: event_type -> probability weight for each user profile
USER_PROFILES = [
    {
        "user_id": "user_alice",
        "preferred_categories": ["movies", "music"],
        "preferred_tags": ["sci-fi", "drama", "rock", "classic"],
        "n_events": 18,
    },
    {
        "user_id": "user_bob",
        "preferred_categories": ["games", "articles"],
        "preferred_tags": ["rpg", "technology", "programming", "open-world"],
        "n_events": 15,
    },
    {
        "user_id": "user_carol",
        "preferred_categories": ["books", "articles"],
        "preferred_tags": ["non-fiction", "science", "philosophy", "self-help"],
        "n_events": 12,
    },
    {
        "user_id": "user_dave",
        "preferred_categories": ["movies", "games"],
        "preferred_tags": ["action", "thriller", "adventure", "challenging"],
        "n_events": 20,
    },
    {
        "user_id": "user_eve",
        "preferred_categories": ["music", "books"],
        "preferred_tags": ["jazz", "classical", "fantasy", "epic"],
        "n_events": 10,
    },
    {
        "user_id": "user_frank",
        "preferred_categories": ["articles", "books"],
        "preferred_tags": ["technology", "ai", "programming", "tutorial"],
        "n_events": 14,
    },
    {
        "user_id": "user_grace",
        "preferred_categories": ["movies", "music"],
        "preferred_tags": ["animation", "family", "pop", "dance"],
        "n_events": 8,
    },
    {
        "user_id": "user_hank",
        "preferred_categories": ["games", "movies"],
        "preferred_tags": ["indie", "pixel-art", "cult", "dark-comedy"],
        "n_events": 16,
    },
    {
        "user_id": "user_iris",
        "preferred_categories": ["books", "music"],
        "preferred_tags": ["fiction", "inspiring", "emotional", "relaxing"],
        "n_events": 11,
    },
    {
        "user_id": "user_jake",
        "preferred_categories": ["games", "articles"],
        "preferred_tags": ["strategy", "sandbox", "science", "future"],
        "n_events": 5,
    },
]


def _score_item(item: dict, profile: dict) -> float:
    """Score an item for a user profile based on category and tag overlap."""
    score = 0.0
    if item["category"] in profile["preferred_categories"]:
        score += 3.0
    item_tags = {t.strip() for t in item["tags"].split(",")} if item["tags"] else set()
    overlap = item_tags & set(profile["preferred_tags"])
    score += len(overlap) * 2.0
    return score


def _generate_feedback(seed: int = 42) -> list[dict]:
    """Generate realistic feedback events for all user profiles.

    Items are chosen weighted by preference affinity so the resulting
    dataset has meaningful signal for the recommendation engine.
    """
    rng = random.Random(seed)
    feedback_events = []

    for profile in USER_PROFILES:
        # Score every item for this user
        scored = [(item, _score_item(item, profile)) for item in ITEMS]
        # Add a small random jitter so we get variety
        weights = [max(s + rng.uniform(0, 1), 0.1) for _, s in scored]
        items_only = [item for item, _ in scored]

        chosen_items = rng.choices(items_only, weights=weights, k=profile["n_events"])

        for item in chosen_items:
            # Heavier engagement (like/purchase/watch) for higher-scoring items
            item_score = _score_item(item, profile)
            if item_score >= 5:
                event = rng.choices(EVENT_TYPES, weights=[1, 3, 2, 2, 1])[0]
            elif item_score >= 2:
                event = rng.choices(EVENT_TYPES, weights=[3, 2, 1, 1, 3])[0]
            else:
                event = rng.choices(EVENT_TYPES, weights=[5, 1, 0, 0, 4])[0]

            feedback_events.append({
                "user_id": profile["user_id"],
                "item_id": item["item_id"],
                "event_type": event,
            })

    return feedback_events


# ---------------------------------------------------------------------------
# Seed via API (httpx → running server)
# ---------------------------------------------------------------------------

def seed_via_api(base_url: str = "http://localhost:8000") -> None:
    """Seed the database by POSTing to the running API server.

    Requires the server to be running on *base_url*.
    """
    import httpx

    print(f"🌱  Seeding via API at {base_url} …\n")

    # ── Items ────────────────────────────────────────────────────────────
    print(f"📦  Seeding {len(ITEMS)} items …")
    with httpx.Client(base_url=base_url, timeout=30) as client:
        for i, item in enumerate(ITEMS, 1):
            resp = client.post("/items", json=item)
            status = "✓" if resp.status_code == 200 else f"✗ {resp.status_code}"
            print(f"   [{i:>2}/{len(ITEMS)}] {status}  {item['title']}")

    # ── Feedback ─────────────────────────────────────────────────────────
    feedback_events = _generate_feedback()
    print(f"\n💬  Seeding {len(feedback_events)} feedback events …")
    with httpx.Client(base_url=base_url, timeout=30) as client:
        for i, fb in enumerate(feedback_events, 1):
            resp = client.post("/feedback", json=fb)
            status = "✓" if resp.status_code == 200 else f"✗ {resp.status_code}"
            if i % 10 == 0 or i == len(feedback_events):
                print(f"   [{i:>3}/{len(feedback_events)}] feedback events posted")

    # ── Trigger training ─────────────────────────────────────────────────
    print("\n🏋️  Triggering model training …")
    with httpx.Client(base_url=base_url, timeout=30) as client:
        resp = client.post("/train")
        print(f"   Training: {resp.json()}")

    # ── Stats ────────────────────────────────────────────────────────────
    print("\n📊  Final stats:")
    with httpx.Client(base_url=base_url, timeout=30) as client:
        stats = client.get("/stats").json()
        for key, value in stats.items():
            print(f"   {key}: {value}")

    print("\n✅  Seeding complete!")


# ---------------------------------------------------------------------------
# Seed directly into SQLite (no server required)
# ---------------------------------------------------------------------------

def seed_directly() -> None:
    """Seed the database by writing directly to SQLite.

    Useful when the server is not running. Initialises the DB tables first.
    """
    from data.database import init_db, get_db_connection
    import json

    print("🌱  Seeding directly into SQLite …\n")
    init_db()

    # ── Items ────────────────────────────────────────────────────────────
    print(f"📦  Seeding {len(ITEMS)} items …")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for i, item in enumerate(ITEMS, 1):
            metadata_json = json.dumps(item.get("metadata")) if item.get("metadata") else None
            cursor.execute(
                """
                INSERT INTO items (item_id, title, tags, category, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    title=excluded.title,
                    tags=excluded.tags,
                    category=excluded.category,
                    metadata_json=excluded.metadata_json
                """,
                (item["item_id"], item["title"], item.get("tags", ""),
                 item.get("category", ""), metadata_json),
            )
            print(f"   [{i:>2}/{len(ITEMS)}] ✓  {item['title']}")
        conn.commit()

    # ── Feedback ─────────────────────────────────────────────────────────
    feedback_events = _generate_feedback()
    print(f"\n💬  Seeding {len(feedback_events)} feedback events …")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for i, fb in enumerate(feedback_events, 1):
            cursor.execute(
                "INSERT INTO feedback (user_id, item_id, event_type) VALUES (?, ?, ?)",
                (fb["user_id"], fb["item_id"], fb["event_type"]),
            )
            if i % 10 == 0 or i == len(feedback_events):
                print(f"   [{i:>3}/{len(feedback_events)}] feedback events inserted")
        conn.commit()

    # ── Summary ──────────────────────────────────────────────────────────
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM items")
        item_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM feedback")
        feedback_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM feedback")
        user_count = cursor.fetchone()[0]

    print(f"\n📊  Final stats:")
    print(f"   items:    {item_count}")
    print(f"   users:    {user_count}")
    print(f"   feedback: {feedback_count}")
    print("\n✅  Seeding complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--direct" in sys.argv:
        seed_directly()
    else:
        seed_via_api()
