# ADR 0001: Thread-Safe LRU Embedding Cache and Lifespan Lifecycle

## Context
Semantic content recommendation engines face high external LLM costs and latency when re-embedding identical content across articles.

## Decision
1. **Thread-Safe LRU Cache**: Cache embedding vectors by content SHA-256 hashes using read-write locks.
2. **Async Lifespan Management**: Warm up models during application startup and cleanly flush connections on exit.
3. **Hybrid Cold-Start Scoring**: Blend semantic similarity with recency factors.

## Consequences
- **Positive**: Massive reduction in embedding API bills and sub-10ms recommendation retrieval.
- **Negative**: Cache consumes up to 512MB of RAM per worker process.
