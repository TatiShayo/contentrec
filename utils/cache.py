import time
import threading
from typing import Any, Dict, Optional, Tuple


class RecommendationCache:
    """Thread-safe In-Memory TTL Cache for recommendations and search."""

    def __init__(self, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache if it hasn't expired."""
        with self._lock:
            if key not in self._cache:
                return None
            val, expiry = self._cache[key]
            if time.time() > expiry:
                del self._cache[key]
                return None
            return val

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in the cache with a specified or default TTL."""
        ttl_val = ttl if ttl is not None else self.default_ttl
        expiry = time.time() + ttl_val
        with self._lock:
            self._cache[key] = (value, expiry)

    def invalidate_user(self, user_id: str) -> None:
        """Invalidate all cache entries associated with a specific user."""
        with self._lock:
            keys_to_del = [
                key for key in self._cache.keys()
                if f"user:{user_id}" in key or f"sequential:{user_id}" in key
            ]
            for key in keys_to_del:
                del self._cache[key]

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Return the current number of cached items."""
        with self._lock:
            # Clean expired items first
            now = time.time()
            expired = [k for k, (_, exp) in self._cache.items() if now > exp]
            for k in expired:
                del self._cache[k]
            return len(self._cache)
