# ============================================================================
# Content Recommendation Engine - Dockerfile
# Multi-stage build: compile LightFM (Cython/C) in builder, slim runtime image
# ============================================================================

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install build tools needed for LightFM (Cython + C extensions)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libc-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies into a prefix we can copy later
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Copy only the installed packages from builder (no compiler toolchain)
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source code
COPY config.py main.py seed_data.py ./
COPY api/ ./api/
COPY data/__init__.py data/database.py data/feedback.py data/items.py ./data/
COPY models/ ./models/
COPY tests/ ./tests/

# Create the data directory for SQLite DB and model artifacts, and run as a
# non-root user (least privilege — a container escape does not land as root).
RUN mkdir -p /app/data \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app \
    && chown -R app:app /app
USER app

# Expose the API port
EXPOSE 8000

# Health-check: hit /health every 30 s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
