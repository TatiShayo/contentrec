"""Causal recommendation via Instrumental Variables.

Establishes a propensity model trained on SQLite impression logs to calculate
inverse propensity score (IPS) weights, debiasing collaborative and sequential
learning loops.
"""

import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from data.database import get_db_connection, get_all_impressions
from data.items import get_all_items

class PropensityModel(nn.Module):
    def __init__(self, input_dim: int = 13):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

class PropensityEstimator:
    def __init__(self, input_dim: int = 13):
        self.model = PropensityModel(input_dim=input_dim)
        self.input_dim = input_dim
        # Categories mapping
        self.categories = ["movies", "music", "books", "articles", "news"]
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # In-memory propensity lookup to avoid overhead during retrieval
        self.propensity_cache: Dict[Tuple[str, str, str, str], float] = {}

    def _encode_features(self, cohort: str, device: str, time_of_day: str, category: str) -> np.ndarray:
        """Encode cohort, device, time, and category into a 13-dimensional feature vector."""
        vec = np.zeros(self.input_dim, dtype=np.float32)
        
        # 1. Cohort: 0 for A (or others), 1 for B
        vec[0] = 1.0 if cohort == "B" else 0.0
        
        # 2. Device (3 dims)
        dev = (device or "").lower()
        if dev == "mobile":
            vec[1] = 1.0
        elif dev == "desktop":
            vec[2] = 1.0
        elif dev == "tablet":
            vec[3] = 1.0
            
        # 3. Time of Day (4 dims)
        tod = (time_of_day or "").lower()
        if tod == "morning":
            vec[4] = 1.0
        elif tod == "afternoon":
            vec[5] = 1.0
        elif tod == "evening":
            vec[6] = 1.0
        elif tod == "night":
            vec[7] = 1.0
            
        # 4. Item Category (5 dims)
        cat = (category or "").lower()
        if cat in self.cat_to_idx:
            vec[8 + self.cat_to_idx[cat]] = 1.0
            
        return vec

    def train_model(self, epochs: int = 10, batch_size: int = 32) -> float:
        """Train the propensity model on impression logs from SQLite."""
        impressions = get_all_impressions()
        if not impressions:
            logging.info("No impressions found in DB. Skipping propensity model training.")
            return 0.0

        all_items = get_all_items()
        if not all_items:
            return 0.0

        items_map = {item["item_id"]: item for item in all_items}
        
        X_data = []
        y_data = []

        # For each positive impression (shown), create a positive sample, and sample 2 negatives (not shown)
        item_ids_list = list(items_map.keys())
        
        for imp in impressions:
            user_id = imp["user_id"]
            item_id = imp["item_id"]
            cohort = imp["cohort"] or "A"
            
            # Parse context details
            context = {}
            if imp["context_json"]:
                try:
                    context = json.loads(imp["context_json"])
                except Exception:
                    pass
            
            device = context.get("device")
            time_of_day = context.get("time_of_day")
            
            item = items_map.get(item_id)
            if not item:
                continue
                
            # Positive sample
            vec_pos = self._encode_features(cohort, device, time_of_day, item.get("category", ""))
            X_data.append(vec_pos)
            y_data.append(1.0)
            
            # Sample 2 negative items (non-exposed)
            neg_sampled = 0
            attempts = 0
            while neg_sampled < 2 and attempts < 10:
                attempts += 1
                neg_id = np.random.choice(item_ids_list)
                if neg_id != item_id:
                    neg_item = items_map[neg_id]
                    vec_neg = self._encode_features(cohort, device, time_of_day, neg_item.get("category", ""))
                    X_data.append(vec_neg)
                    y_data.append(0.0)
                    neg_sampled += 1

        if not X_data:
            return 0.0

        X = torch.tensor(np.array(X_data), dtype=torch.float32)
        y = torch.tensor(np.array(y_data), dtype=torch.float32).unsqueeze(1)

        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        self.model.train()
        dataset_size = len(X)
        
        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0
            
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X[indices], y[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_x)
                
        # Re-build lookup cache for common combinations after training
        self.model.eval()
        self.propensity_cache.clear()
        
        logging.info(f"Propensity estimator trained on {dataset_size} samples. Final epoch loss: {epoch_loss/dataset_size:.4f}")
        return epoch_loss / dataset_size

    def predict_propensity(self, cohort: str, device: str, time_of_day: str, category: str) -> float:
        """Predict the exposure probability p(shown) and cache it."""
        cache_key = (cohort, device or "", time_of_day or "", category or "")
        if cache_key in self.propensity_cache:
            return self.propensity_cache[cache_key]

        vec = self._encode_features(cohort, device, time_of_day, category)
        tensor_x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            prob = float(self.model(tensor_x).item())
            
        # Guarantee a minimum propensity threshold to avoid division by zero
        prob = max(0.05, min(0.95, prob))
        self.propensity_cache[cache_key] = prob
        return prob

    def get_ips_weight(self, cohort: str, device: str, time_of_day: str, category: str, clicked: bool = True) -> float:
        """Compute the Inverse Propensity Score weight: 1/p for exposed items."""
        prob = self.predict_propensity(cohort, device, time_of_day, category)
        
        if clicked:
            # Clicked item weight: 1 / p_ui
            weight = 1.0 / prob
        else:
            # Non-clicked item weight: 1 / (1 - p_ui)
            weight = 1.0 / (1.0 - prob + 1e-10)
            
        # Clip weight to control variance in gradients [0.1, 10.0]
        return float(np.clip(weight, 0.1, 10.0))
