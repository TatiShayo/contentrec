import threading
import time
import sys
from typing import Dict, List, Any
import config


def _calculate_percentile(latencies: List[float], percentile: float) -> float:
    if not latencies:
        return 0.0
    sorted_latencies = sorted(latencies)
    idx = int(len(sorted_latencies) * percentile)
    return round(sorted_latencies[min(idx, len(sorted_latencies) - 1)], 4)


class MetricsTracker:
    """Thread-safe collector for recommendation engine operational metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cohort_counts = {"A": 0, "B": 0}
        self.cohort_conversions = {"A": 0, "B": 0}
        
        # Buffer to map (user_id, item_id) -> (cohort, timestamp)
        self.served_recs: Dict[tuple, tuple] = {}
        # Buffer to map (user_id, item_id) -> (arm_id, context_vector, timestamp)
        self.served_bandit_arms: Dict[tuple, tuple] = {}
        
        self.latencies: Dict[str, List[float]] = {
            "recommend": [],
            "sequential": [],
            "search": []
        }
        self.max_latency_records = 1000
        
        # SLA & Request tracking
        self.total_requests_counted = 0
        self.sla_violations = 0

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

    def record_recommendation_served(self, cohort: str) -> None:
        """Record recommendation serving count per cohort."""
        with self._lock:
            if cohort in self.cohort_counts:
                self.cohort_counts[cohort] += 1

    def record_served_recs(self, user_id: str, item_ids: List[str], cohort: str, engine=None) -> None:
        """Record served items to buffer to trace future conversions, pruning old entries (>1h)."""
        with self._lock:
            now = time.time()
            for item_id in item_ids:
                self.served_recs[(user_id, item_id)] = (cohort, now)
            
            # Prune entries older than 1 hour (3600 seconds)
            cutoff = now - 3600
            expired_keys = [k for k, v in self.served_recs.items() if v[1] < cutoff]
            for k in expired_keys:
                del self.served_recs[k]

            # Prune expired bandit arms and perform negative (0.0) updates
            expired_bandit_keys = [k for k, v in self.served_bandit_arms.items() if v[2] < cutoff]
            for k in expired_bandit_keys:
                arm_id, context_vector, timestamp = self.served_bandit_arms[k]
                if engine and hasattr(engine, 'bandit') and engine.bandit:
                    engine.bandit.update(arm_id, context_vector, reward=0.0)
                del self.served_bandit_arms[k]

    def record_conversion(self, user_id: str, item_id: str, engine=None) -> None:
        """If user interacts with recently recommended item, record cohort conversion and update bandit."""
        with self._lock:
            key = (user_id, item_id)
            if key in self.served_recs:
                cohort, timestamp = self.served_recs[key]
                if time.time() - timestamp <= 3600:
                    if cohort in self.cohort_conversions:
                        self.cohort_conversions[cohort] += 1
                    # Remove to prevent double counting
                    del self.served_recs[key]
                    
            if key in self.served_bandit_arms:
                arm_id, context_vector, timestamp = self.served_bandit_arms[key]
                if time.time() - timestamp <= 3600:
                    if engine and hasattr(engine, 'bandit') and engine.bandit:
                        engine.bandit.update(arm_id, context_vector, reward=1.0)
                    del self.served_bandit_arms[key]

    def record_latency(self, endpoint: str, duration_sec: float) -> None:
        with self._lock:
            if endpoint in self.latencies:
                self.latencies[endpoint].append(duration_sec)
                # Cap the list size
                if len(self.latencies[endpoint]) > self.max_latency_records:
                    self.latencies[endpoint].pop(0)
            
            self.total_requests_counted += 1
            threshold = getattr(config, 'LATENCY_SLA_ALERT_THRESHOLD_SEC', 0.15)
            if duration_sec > threshold:
                self.sla_violations += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total_cache_queries = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_cache_queries) if total_cache_queries > 0 else 0.0
            
            avg_latencies = {}
            p95_latencies = {}
            p99_latencies = {}
            for endpoint, times in self.latencies.items():
                avg_latencies[endpoint] = (sum(times) / len(times)) if times else 0.0
                p95_latencies[endpoint] = _calculate_percentile(times, 0.95)
                p99_latencies[endpoint] = _calculate_percentile(times, 0.99)

            # CTR calculations
            ctr = {}
            for c in ["A", "B"]:
                impressions = self.cohort_counts[c]
                conversions = self.cohort_conversions[c]
                ctr[c] = round(conversions / impressions, 4) if impressions > 0 else 0.0

            # SLA compliance calculation
            sla_compliance = 1.0
            if self.total_requests_counted > 0:
                sla_compliance = round(1.0 - (self.sla_violations / self.total_requests_counted), 4)

            # Memory Footprint Estimate (bytes)
            mem_footprint = sys.getsizeof(self.served_recs)
            for k, v in self.served_recs.items():
                mem_footprint += sys.getsizeof(k) + sys.getsizeof(v)

            return {
                "cache": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": round(hit_rate, 4)
                },
                "cohort_recommendations_served": self.cohort_counts.copy(),
                "cohort_conversions": self.cohort_conversions.copy(),
                "cohort_click_through_rate": ctr,
                "average_latency_seconds": {
                    k: round(v, 4) for k, v in avg_latencies.items()
                },
                "p95_latency_seconds": p95_latencies,
                "p99_latency_seconds": p99_latencies,
                "sla_telemetry": {
                    "total_requests": self.total_requests_counted,
                    "sla_violations": self.sla_violations,
                    "sla_compliance_rate": sla_compliance,
                    "latency_threshold_sec": getattr(config, 'LATENCY_SLA_ALERT_THRESHOLD_SEC', 0.15)
                },
                "estimated_memory_bytes": mem_footprint
            }

