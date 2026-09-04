# CONTEXT.md — Ubiquitous Domain Language (ContentRec)

## Core Entities
- **ContentArticle**: Indexed publication containing title, body text, tags, and semantic embedding vector.
- **EmbeddingLruCache**: Thread-safe memory cache avoiding redundant vector calculation for identical texts.
- **LifespanContext**: Application lifecycle manager controlling startup model warmup and graceful shutdown.
- **HybridRanker**: Scoring algorithm combining vector cosine proximity with recency decay factors.

## Domain Invariants
- Cache keys must be computed strictly via SHA-256 digest of normalized input text.
- External embedding API calls must never be invoked if the content hash exists in cache.

## Forbidden Terminology
- Do not call recommendations "suggestions"; use "RecommendationScore".
- Do not refer to content items as "posts"; use "ContentArticle".
