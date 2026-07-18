"""Application configuration.

Values are read from environment variables where sensible so that secrets and
deployment-specific settings never need to be hard-coded. Import-time defaults
keep local development and the test suite working with zero setup.
"""

import os


def _get_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- Storage paths -----------------------------------------------------------
DATABASE_PATH = os.environ.get("DATABASE_PATH", "data/recommender.db")
MODEL_PATH = os.environ.get("MODEL_PATH", "data/model.pkl")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "data/faiss.index")
SASREC_MODEL_PATH = os.environ.get("SASREC_MODEL_PATH", "data/sasrec_model.pt")
SASREC_MAP_PATH = os.environ.get("SASREC_MAP_PATH", "data/sasrec_map.pkl")
LIGHTGCN_MODEL_PATH = os.environ.get("LIGHTGCN_MODEL_PATH", "data/lightgcn.pkl")
BCQ_MODEL_PATH = os.environ.get("BCQ_MODEL_PATH", "data/bcq.pth")

# --- Recommendation tuning ---------------------------------------------------
COLD_START_THRESHOLD = _get_int("COLD_START_THRESHOLD", 3)
DEFAULT_N_RECOMMENDATIONS = _get_int("DEFAULT_N_RECOMMENDATIONS", 10)
# Hard upper bound on how many recommendations/results a single request may ask
# for. Prevents an unbounded-`n` denial-of-service (candidate blow-up + cache
# pollution). See AUDIT_LOG.md "Unbounded n DoS".
MAX_N_RECOMMENDATIONS = _get_int("MAX_N_RECOMMENDATIONS", 100)
# Upper bound on pagination window for list endpoints.
MAX_PAGE_LIMIT = _get_int("MAX_PAGE_LIMIT", 500)

RETRAIN_THRESHOLD_FEEDBACK = _get_int("RETRAIN_THRESHOLD_FEEDBACK", 50)
RETRAIN_INTERVAL_SECONDS = _get_int("RETRAIN_INTERVAL_SECONDS", 86400)  # 24 hours

WEIGHT_RELEVANCE = 1.0
WEIGHT_FRESHNESS = 0.2
WEIGHT_FATIGUE = 0.3
WEIGHT_CONTEXT = 0.4
LATENCY_SLA_ALERT_THRESHOLD_SEC = 0.15

# --- Security ----------------------------------------------------------------
# Optional API key. When set (via the API_KEY env var) every non-public
# endpoint requires a matching `X-API-Key` header. Unset by default so local
# development and tests stay frictionless — set it in any real deployment.
API_KEY = os.environ.get("API_KEY", "").strip()

# CORS allowed origins. Comma-separated list from CORS_ALLOW_ORIGINS.
# Default is empty (no cross-origin browser access). A wildcard "*" is honoured
# but credentials are then force-disabled (browsers forbid `*` + credentials,
# and the combination is a data-exfiltration risk).
CORS_ALLOW_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()
]

TESTING = _get_bool("TESTING", False)
