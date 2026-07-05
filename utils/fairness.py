import threading
from collections import deque
from typing import Dict, List, Any
import numpy as np
from data.items import get_all_items

class FairnessAuditor:
    """Audits recommendation popularity/demographic bias and applies PID-controlled re-ranking."""

    def __init__(self, target_di: float = 1.0, threshold: float = 0.8, window_size: int = 1000,
                 kp: float = 0.5, ki: float = 0.05, kd: float = 0.1):
        self._lock = threading.Lock()
        self.target_di = target_di
        self.threshold = threshold
        self.window_size = window_size
        
        # PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.error_sum = 0.0
        self.last_error = 0.0
        self.lambda_fair = 0.0
        
        # Rolling window of the last 1000 impressions
        # Stores True if the item belongs to the minority group ('books'), False otherwise
        self.impressions = deque(maxlen=window_size)
        
        # Count of items in the catalog to calculate selection probabilities
        self.cat_minority = 1.0
        self.cat_majority = 1.0
        self.refresh_catalog_counts()

    def refresh_catalog_counts(self) -> None:
        """Update catalog group counts from the database."""
        try:
            items = get_all_items()
            minority_count = sum(1 for i in items if i.get('category') == 'books')
            majority_count = sum(1 for i in items if i.get('category') != 'books')
            self.cat_minority = float(max(1, minority_count))
            self.cat_majority = float(max(1, majority_count))
        except Exception:
            # Fallbacks matching typical seed data
            self.cat_minority = 11.0
            self.cat_majority = 40.0

    def compute_di(self) -> float:
        """Compute the rolling Disparate Impact (DI) ratio."""
        if len(self.impressions) == 0:
            return 1.0
            
        rec_minority = sum(1 for x in self.impressions if x)
        rec_majority = len(self.impressions) - rec_minority
        
        p_minority = rec_minority / self.cat_minority
        p_majority = rec_majority / self.cat_majority
        
        if p_majority == 0.0:
            return 1.0
            
        return p_minority / p_majority

    def audit_and_update_pid(self) -> float:
        """Execute one step of the PID controller to tune the fairness bonus lambda."""
        with self._lock:
            di = self.compute_di()
            
            # Error is the difference between target DI (1.0) and actual DI
            error = self.target_di - di
            
            self.error_sum = float(np.clip(self.error_sum + error, -10.0, 10.0))
            derivative = error - self.last_error
            self.last_error = error
            
            # PID controller formula
            output = self.kp * error + self.ki * self.error_sum + self.kd * derivative
            self.lambda_fair = float(np.clip(output, 0.0, 2.0))
            
            return di

    def record_recommendations(self, items_details: List[dict]) -> None:
        """Add recommended items to the rolling impressions queue."""
        with self._lock:
            for item in items_details:
                is_minority = (item.get('category') == 'books')
                self.impressions.append(is_minority)

    def get_fairness_score(self, item_category: str, score_rrf: float) -> float:
        """Add fairness bonus to minority group candidate items."""
        is_minority = (item_category == 'books')
        if is_minority:
            return score_rrf + self.lambda_fair
        return score_rrf
