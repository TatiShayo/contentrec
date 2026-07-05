"""
Text embedding module using Sentence-BERT.

Provides a thread-safe, singleton TextEmbedder that lazily loads
the all-MiniLM-L6-v2 model on first use. All embeddings are
L2-normalized for cosine similarity via inner product.
"""

import threading
from typing import List, Optional

import numpy as np

# Model name and expected embedding dimension
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class TextEmbedder:
    """Thread-safe singleton for generating text embeddings.

    Uses sentence-transformers' all-MiniLM-L6-v2 model (384 dims, CPU).
    The model is lazily loaded on first encode call to avoid slow imports.
    """

    _instance: Optional["TextEmbedder"] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "TextEmbedder":
        """Singleton: return the same instance on every call."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._model = None
        self._lock = threading.Lock()
        self._initialized = True

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the sentence-transformers model (called once, under lock)."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(MODEL_NAME)

    def _ensure_model(self) -> None:
        """Ensure the model is loaded, loading it lazily if needed."""
        if self._model is None:
            with self._lock:
                self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a normalized embedding vector.

        Args:
            text: The input text to encode.

        Returns:
            A 1-D numpy array of shape ``(384,)`` with L2-normalized values.
        """
        self._ensure_model()
        with self._lock:
            embedding = self._model.encode(
                text, normalize_embeddings=True, show_progress_bar=False
            )
        return np.asarray(embedding, dtype=np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of text strings into normalized embedding vectors.

        Args:
            texts: A list of input texts.

        Returns:
            A 2-D numpy array of shape ``(len(texts), 384)``.
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self._ensure_model()
        with self._lock:
            embeddings = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_item(self, item: dict) -> np.ndarray:
        """Create an embedding from an item's title, tags, and category.

        The item dict is expected to have keys ``title``, ``tags``
        (comma-separated string or None), and ``category`` (string or None).
        These fields are concatenated into a single text passage before
        encoding.

        Args:
            item: A dict with at least a ``title`` key.

        Returns:
            A 1-D numpy array of shape ``(384,)``.
        """
        parts: List[str] = []

        title = item.get("title", "")
        if title:
            parts.append(title)

        tags = item.get("tags", "")
        if tags:
            parts.append(tags)

        category = item.get("category", "")
        if category:
            parts.append(category)

        text = " ".join(parts) if parts else ""
        return self.encode(text)
