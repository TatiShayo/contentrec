"""Greedy Determinantal Point Process (DPP) taste diversity and onboarding onboarding helper.

Implements the fast greedy MAP inference for DPP to select a diverse subset of items
for onboarding and bootstrap cold-start user embeddings using weighted nearest-neighbors.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple
from data.items import get_all_items
from embeddings.text import TextEmbedder

class DPPSelector:
    @staticmethod
    def select_diverse_items(items: List[dict], pool_size: int = 50, n_quiz: int = 8) -> List[dict]:
        """Greedy DPP MAP selection to find a highly diverse subset of items.
        
        Args:
            items: List of item dictionaries (with category, item_id, title, tags).
            pool_size: First select the top-N popular items as the candidate pool.
            n_quiz: The final number of items to return.
        """
        if not items:
            return []
            
        embedder = TextEmbedder()
        
        # 1. First-stage filter: get the items, and calculate simple popularity scores
        # We simulate popularity via feedback interactions or categories coverage
        # Let's count tags and title lengths or just assign a high default quality
        # to ensure good candidate pool. Let's rank by a pseudo-popularity score.
        from data.feedback import get_all_feedback
        feedback = get_all_feedback()
        counts: Dict[str, int] = {}
        for f in feedback:
            iid = f["item_id"]
            counts[iid] = counts.get(iid, 0) + 1
            
        candidate_pool = []
        for item in items:
            iid = item["item_id"]
            # Quality score: log(clicks + 2.0)
            q = float(np.log(counts.get(iid, 0) + 2.0))
            candidate_pool.append((item, q))
            
        # Sort candidates by quality (popularity) and take top pool_size
        candidate_pool.sort(key=lambda x: x[1], reverse=True)
        candidate_pool = candidate_pool[:pool_size]
        
        N = len(candidate_pool)
        if N == 0:
            return []
            
        # Get SBERT embeddings for all pool candidates
        embeddings = []
        for item, q in candidate_pool:
            try:
                emb = embedder.embed_item(item)
                embeddings.append(emb)
            except Exception:
                embeddings.append(np.zeros(384, dtype=np.float32))
                
        embeddings_arr = np.array(embeddings, dtype=np.float32)
        # Normalize embeddings
        norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
        embeddings_arr = embeddings_arr / (norms + 1e-10)
        
        # Quality scores
        q_scores = np.array([q for item, q in candidate_pool], dtype=np.float32)
        
        # Fast Greedy DPP MAP Inference (Chen et al. KDD 2018)
        # We maintain c_i and d_i to greedily compute Cholesky updates
        d = q_scores ** 2
        c = np.zeros((n_quiz, N), dtype=np.float32)
        
        selected_indices: List[int] = []
        
        for step in range(min(n_quiz, N)):
            # Find candidate with max marginal gain d
            # Mask out already selected items by setting their d to -inf
            d_masked = d.copy()
            for idx in selected_indices:
                d_masked[idx] = -float("inf")
                
            best_idx = int(np.argmax(d_masked))
            if d_masked[best_idx] <= 0:
                break
                
            selected_indices.append(best_idx)
            
            # Cholesky update step for non-selected items
            best_emb = embeddings_arr[best_idx]
            best_q = q_scores[best_idx]
            
            for i in range(N):
                if i in selected_indices:
                    continue
                # Calculate kernel value L_ij between item i and selected best_idx
                sim = float(np.dot(embeddings_arr[i], best_emb))
                L_ij = q_scores[i] * best_q * sim
                
                # Update using previous orthogonal dimensions
                sum_c = 0.0
                for k in range(step):
                    sum_c += c[k, i] * c[k, best_idx]
                    
                e = (L_ij - sum_c) / np.sqrt(d[best_idx] + 1e-10)
                c[step, i] = e
                d[i] = d[i] - e ** 2

        return [candidate_pool[idx][0] for idx in selected_indices]


class ColdStartProfileBuilder:
    @staticmethod
    def bootstrap_user_profile(
        ratings: Dict[str, float], 
        item_details: Dict[str, dict], 
        gcn_embeddings: np.ndarray, 
        item_to_gcn_idx: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bootstrap GCN (64-dim) and SBERT (384-dim) user profile embeddings from onboarding ratings.
        
        Args:
            ratings: Dict mapping item_id -> rating (1.0 for positive, -1.0 for negative, or similar).
            item_details: Dict mapping item_id -> item detail dict.
            gcn_embeddings: GCN item embeddings matrix of shape (num_items, 64).
            item_to_gcn_idx: Dict mapping item_id -> index in the GCN matrix.
            
        Returns:
            Tuple of (gcn_user_profile_embedding, sbert_user_profile_embedding).
        """
        embedder = TextEmbedder()
        
        # Filter to active ratings
        positive_items = [iid for iid, r in ratings.items() if r > 0]
        
        # 1. Bootstrap SBERT (384) User Profile
        sbert_profile = np.zeros(384, dtype=np.float32)
        sbert_count = 0
        
        for iid in positive_items:
            item = item_details.get(iid)
            if item:
                try:
                    sbert_profile += embedder.embed_item(item)
                    sbert_count += 1
                except Exception:
                    pass
                    
        if sbert_count > 0:
            sbert_profile = sbert_profile / sbert_count
            # Normalize profile
            sbert_norm = np.linalg.norm(sbert_profile)
            if sbert_norm > 0:
                sbert_profile = sbert_profile / sbert_norm

        # 2. Bootstrap GCN (64) User Profile
        gcn_profile = np.zeros(64, dtype=np.float32)
        gcn_count = 0
        
        if gcn_embeddings is not None:
            for iid in positive_items:
                if iid in item_to_gcn_idx:
                    idx = item_to_gcn_idx[iid]
                    gcn_profile += gcn_embeddings[idx]
                    gcn_count += 1
                    
        if gcn_count > 0:
            gcn_profile = gcn_profile / gcn_count
            gcn_norm = np.linalg.norm(gcn_profile)
            if gcn_norm > 0:
                gcn_profile = gcn_profile / gcn_norm
        else:
            # Fallback to mean of all item embeddings if no GCN idx matched
            if gcn_embeddings is not None and len(gcn_embeddings) > 0:
                gcn_profile = np.mean(gcn_embeddings, axis=0)

        return gcn_profile, sbert_profile
