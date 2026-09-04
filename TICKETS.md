# TICKETS — ContentRec Engine Pipeline

## [TICKET-001] Thread-Safe LRU Embedding Cache
- **Blocked by**: None
- **Delivers**: Memory cache storing vectors by content hash with thread-safe locking.
- **Verification**: `tests/test_lifespan_and_embedding_cache.py`

## [TICKET-002] Async FastAPI Lifespan Lifecycle Manager
- **Blocked by**: TICKET-001
- **Delivers**: Clean startup model pre-loading and graceful shutdown resource disposal.
- **Verification**: Lifecycle integration unit tests.
