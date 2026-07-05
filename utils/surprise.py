"""Bayesian Surprise Minimization via PID-controlled MMR Diversity.

Maintains a Dirichlet distribution over user category preferences, computes
KL-divergence between user taste profiles and candidate distributions, and dynamically
tunes MMR diversity parameters at request time.
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple

class SurpriseController:
    def __init__(self, target_kl: float = 0.5, kp: float = 0.5):
        self.target_kl = target_kl
        self.kp = kp
        self.categories = ["movies", "music", "books", "articles", "news"]
        self.num_cats = len(self.categories)
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

    def get_user_dirichlet_prior(self, user_feedback: List[dict], item_details: Dict[str, dict]) -> np.ndarray:
        """Construct Dirichlet parameter vector alpha (prior = 1.0 for Laplace smoothing)."""
        alpha = np.ones(self.num_cats, dtype=np.float32)
        
        for f in user_feedback:
            item_id = f["item_id"]
            details = item_details.get(item_id)
            if details:
                cat = (details.get("category") or "").lower()
                if cat in self.cat_to_idx:
                    # Increment count for this category
                    idx = self.cat_to_idx[cat]
                    alpha[idx] += 1.0
                    
        return alpha

    def compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Kullback-Leibler divergence D_KL(P || Q)."""
        kl = 0.0
        for i in range(len(p)):
            if p[i] > 0:
                kl += p[i] * math.log(p[i] / (q[i] + 1e-10) + 1e-10)
        return float(kl)

    def adjust_diversity_lambda(
        self, 
        base_lambda: float, 
        user_feedback: List[dict], 
        candidates: List[Dict[str, Any]], 
        item_details: Dict[str, dict]
    ) -> Tuple[float, float, float]:
        """Adjust MMR diversity lambda based on KL surprise difference.
        
        Returns:
            Tuple of:
                - adjusted_lambda: float clipped to [0.0, 1.0]
                - actual_kl: float representing the KL divergence
                - error: float representing actual_kl - target_kl
        """
        # 1. P (User Preference Distribution) from Dirichlet mean
        alpha = self.get_user_dirichlet_prior(user_feedback, item_details)
        p = alpha / np.sum(alpha)
        
        # 2. Q (Candidate Recommendations Category Distribution)
        # Count category frequencies in candidate set
        q_counts = np.ones(self.num_cats, dtype=np.float32) # Laplace smooth
        for c in candidates[:10]: # Look at top 10 candidates
            iid = c["item_id"]
            item = item_details.get(iid)
            if item:
                cat = (item.get("category") or "").lower()
                if cat in self.cat_to_idx:
                    q_counts[self.cat_to_idx[cat]] += 1.0
                    
        q = q_counts / np.sum(q_counts)
        
        # 3. Compute actual KL divergence
        actual_kl = self.compute_kl_divergence(p, q)
        
        # 4. PID controller error: actual - target
        error = actual_kl - self.target_kl
        
        # If error > 0 (too surprising / high KL), we increase lambda (reduce diversity)
        # If error < 0 (not surprising enough / low KL), we decrease lambda (increase diversity)
        adjusted_lambda = base_lambda + self.kp * error
        adjusted_lambda = max(0.0, min(1.0, adjusted_lambda))
        
        return float(adjusted_lambda), actual_kl, float(error)
