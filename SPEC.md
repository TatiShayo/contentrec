# SPEC 001: Semantic Content Recommendation & Embedding Cache Engine

## Problem Statement
Publishers struggle to retain readers with dumb keyword search and face prohibitive cloud costs for AI embedding generation.

## Solution
A semantic recommendation engine featuring thread-safe embedding caching, hybrid ranking, and clean FastAPI lifespan management.

## User Stories
1. As a reader, I want relevant articles suggested based on meaning rather than exact keywords, so that I discover interesting content.
2. As a platform owner, I want identical text embeddings cached, so that our AI API costs are kept minimal.

## Implementation Decisions
- Thread-safe cache in `embeddings/cache.py`.
- Lifespan context in `app/main.py`.

## Testing Decisions
- Seam: `tests/test_lifespan_and_embedding_cache.py`.
- Verify cache hit rates, thread safety under concurrency, and startup model warmup.
