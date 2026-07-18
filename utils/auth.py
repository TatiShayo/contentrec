"""Optional API-key authentication middleware.

Authentication is *opt-in*: it activates only when ``config.API_KEY`` is set
(via the ``API_KEY`` environment variable). This keeps local development and the
test suite frictionless while giving real deployments a single-env-var switch to
lock every write/recommend endpoint behind a shared secret.

Public paths (health checks, docs, OpenAPI schema) are always exempt.
"""

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Paths reachable without an API key even when auth is enabled.
PUBLIC_PATHS = frozenset(
    {"/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require a valid ``X-API-Key`` header when an API key is configured."""

    async def dispatch(self, request, call_next):
        import config

        api_key = getattr(config, "API_KEY", "") or ""
        # Disabled unless a key is configured; also bypassed under tests.
        if not api_key or getattr(config, "TESTING", False):
            return await call_next(request)

        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        provided = request.headers.get("x-api-key", "")
        # Constant-time comparison to avoid timing side channels.
        if not provided or not hmac.compare_digest(provided, api_key):
            return JSONResponse(
                status_code=401, content={"detail": "Invalid or missing API key."}
            )

        return await call_next(request)
