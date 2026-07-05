"""Mock CLIP vision embedding generator.

Generates 384-dimensional image embeddings aligned to the SBERT text embedding space
using deterministic perturbations, simulating an aligned late-fusion cross-modal item representation.
"""

import hashlib
import numpy as np
from typing import List
from embeddings.text import TextEmbedder, EMBEDDING_DIM

class VisionEmbedder:
    """Singleton generating deterministic mock image embeddings aligned to text embeddings."""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._embedder = TextEmbedder()
        return cls._instance

    def _generate_deterministic_noise(self, item_id: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
        """Create a stable, reproducible noise vector of dimension dim."""
        # Use SHA-256 to hash the item_id and seed numpy's random generator
        hash_digest = hashlib.sha256(item_id.encode('utf-8')).digest()
        seed = int.from_bytes(hash_digest[:4], 'big')
        
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=0.1, size=dim)
        return noise.astype(np.float32)

    def embed_item_image(self, item: dict) -> np.ndarray:
        """Simulate obtaining a dynamically-sized image embedding from cover art/poster."""
        item_id = item.get("item_id", "default_item")
        
        # 1. Obtain SBERT text embedding as the baseline (since CLIP spaces are cross-modal aligned)
        try:
            text_emb = self._embedder.embed_item(item)
        except Exception:
            text_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            
        # 2. Add deterministic visual noise to simulate visual variance
        dim = len(text_emb)
        noise = self._generate_deterministic_noise(item_id, dim=dim)
        fused = text_emb + noise
        
        # 3. L2-normalize the result
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        else:
            fused = text_emb
            
        return fused
