"""
FAISS-based approximate nearest-neighbour search index using HNSW.

Stores L2-normalized item embeddings in a ``faiss.IndexHNSWFlat`` (wrapped in a
``faiss.IndexIDMap2`` for custom ID mapping and removal support) so that
inner-product search is equivalent to cosine similarity. The index and the
item-id ↔ row mapping are persisted to disk for fast restarts.
"""

import os
import pickle
import threading
from typing import Dict, List, Optional

import faiss
import numpy as np

from embeddings.text import EMBEDDING_DIM, TextEmbedder

# Default persistence paths (relative – resolved against cwd)
DEFAULT_INDEX_PATH = os.path.join("data", "faiss.index")
DEFAULT_MAP_PATH = os.path.join("data", "faiss_map.pkl")


class FAISSIndex:
    """Thread-safe wrapper around a FAISS IndexHNSWFlat inner-product index.

    The index maps custom integer IDs to item IDs so results can
    be returned as ``[{item_id, score}]`` dicts.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        """Create a new FAISS HNSW inner-product index.

        Args:
            dimension: Embedding vector dimension (default 384).
        """
        self.dimension = dimension
        # Construct HNSW Flat index with Inner Product metric
        sub_index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        sub_index.hnsw.efConstruction = 200
        sub_index.hnsw.efSearch = 64
        # Wrap in IndexIDMap2 to support custom integer IDs and removals
        self.index = faiss.IndexIDMap2(sub_index)
        self._id_to_row: Dict[str, int] = {}
        self._row_to_id: Dict[int, str] = {}
        self._lock = threading.Lock()
        self._embedder = TextEmbedder()
        self._next_id = 0

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, items: List[dict]) -> None:
        """Embed all items and rebuild the index from scratch.

        Args:
            items: List of item dicts, each containing at least
                ``item_id``, ``title``, and optionally ``tags``/``category``.
        """
        if not items:
            return

        texts = []
        item_ids = []
        for item in items:
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
            texts.append(" ".join(parts) if parts else "")
            item_ids.append(item["item_id"])

        embeddings = self._embedder.encode_batch(texts)
        # Late-fusion multi-modal embedding
        from embeddings.vision import VisionEmbedder
        vision_embedder = VisionEmbedder()
        fused_embeddings = []
        lambda_t = 0.7
        lambda_v = 0.3
        for idx, item in enumerate(items):
            t_emb = embeddings[idx]
            v_emb = vision_embedder.embed_item_image(item)
            fused = lambda_t * t_emb + lambda_v * v_emb
            fused_embeddings.append(fused)
            
        embeddings = np.array(fused_embeddings, dtype=np.float32)
        # Normalize vectors to unit length so that inner product equals cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)

        with self._lock:
            sub_index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            sub_index.hnsw.efConstruction = 200
            sub_index.hnsw.efSearch = 64
            self.index = faiss.IndexIDMap2(sub_index)
            
            self._id_to_row = {}
            self._row_to_id = {}
            ids = []
            for idx, iid in enumerate(item_ids):
                self._id_to_row[iid] = idx
                self._row_to_id[idx] = iid
                ids.append(idx)
                
            self._next_id = len(item_ids)
            ids_arr = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(embeddings.astype(np.float32), ids_arr)

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, n: int = 10) -> List[dict]:
        """Search for the *n* nearest items by cosine similarity.

        Args:
            query_embedding: A 1-D float32 array of shape ``(dimension,)``.
            n: Number of results to return.

        Returns:
            A list of dicts ``[{item_id: str, score: float}, ...]``
            ordered by descending similarity.
        """
        with self._lock:
            if self.index.ntotal == 0:
                return []

            # Normalize query vector for cosine similarity
            q_norm = np.linalg.norm(query_embedding)
            query = np.asarray(query_embedding, dtype=np.float32) / (q_norm + 1e-10)
            query = query.reshape(1, -1)
            
            k = min(n, self.index.ntotal)
            distances, indices = self.index.search(query, k)

            results: List[dict] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                item_id = self._row_to_id.get(int(idx))
                if item_id is not None:
                    results.append({"item_id": item_id, "score": float(dist)})
            return results

    def search_by_text(self, query: str, n: int = 10) -> List[dict]:
        """Encode a text query and search the index.

        Args:
            query: Natural-language search string.
            n: Number of results to return.

        Returns:
            Same format as :meth:`search`.
        """
        embedding = self._embedder.encode(query)
        return self.search(embedding, n=n)

    # ------------------------------------------------------------------
    # Incremental updates
    # ------------------------------------------------------------------

    def add_item(self, item: dict) -> None:
        """Add or update a single item in the existing index.

        Args:
            item: An item dict with ``item_id``, ``title``, etc.
        """
        from embeddings.vision import VisionEmbedder
        vision_embedder = VisionEmbedder()
        t_emb = self._embedder.embed_item(item)
        v_emb = vision_embedder.embed_item_image(item)
        
        lambda_t = 0.7
        lambda_v = 0.3
        fused = lambda_t * t_emb + lambda_v * v_emb
        
        # Normalize vector to unit length
        norm = np.linalg.norm(fused)
        embedding = fused / (norm + 1e-10)

        with self._lock:
            item_id = item["item_id"]
            if item_id in self._id_to_row:
                old_id = self._id_to_row[item_id]
                selector = faiss.IDSelectorArray(np.array([old_id], dtype=np.int64))
                self.index.remove_ids(selector)
                int_id = old_id
            else:
                int_id = self._next_id
                self._next_id += 1
                
            self._id_to_row[item_id] = int_id
            self._row_to_id[int_id] = item_id
            
            ids_arr = np.array([int_id], dtype=np.int64)
            self.index.add_with_ids(embedding.reshape(1, -1).astype(np.float32), ids_arr)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist the FAISS index and the id mapping to disk.

        Args:
            path: Path for the FAISS index file.  The mapping is saved
                alongside it with a ``.pkl`` extension.  Defaults to
                ``data/faiss.index``.
        """
        index_path = path or DEFAULT_INDEX_PATH
        map_path = os.path.splitext(index_path)[0] + "_map.pkl"

        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)

        with self._lock:
            faiss.write_index(self.index, index_path)
            with open(map_path, "wb") as f:
                pickle.dump(
                    {
                        "id_to_row": self._id_to_row,
                        "row_to_id": self._row_to_id,
                    },
                    f,
                )

    def load(self, path: Optional[str] = None) -> None:
        """Load a previously saved FAISS index and mapping from disk.

        Args:
            path: Path to the FAISS index file.  Defaults to
                ``data/faiss.index``.
        """
        index_path = path or DEFAULT_INDEX_PATH
        map_path = os.path.splitext(index_path)[0] + "_map.pkl"

        with self._lock:
            self.index = faiss.read_index(index_path)
            with open(map_path, "rb") as f:
                mapping = pickle.load(f)
                self._id_to_row = mapping["id_to_row"]
                self._row_to_id = mapping["row_to_id"]
                if self._id_to_row:
                    self._next_id = max(self._id_to_row.values()) + 1
                else:
                    self._next_id = 0
