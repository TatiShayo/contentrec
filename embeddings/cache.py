"""
Thread-safe Bounded LRU Embedding Cache for contentrec.
Reduces redundant neural network inference for repeated text queries.
"""
import threading
from collections import OrderedDict
from typing import Optional, Tuple
import numpy as np

class EmbeddingCache:
    """Thread-safe LRU cache for high-dimensional vector embeddings."""

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, text: str) -> Optional[np.ndarray]:
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                self.hits += 1
                return self._cache[text].copy()
            self.misses += 1
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                self._cache[text] = embedding.copy()
                return

            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self.evictions += 1

            self._cache[text] = embedding.copy()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate
            }
