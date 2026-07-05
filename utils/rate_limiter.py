import time
import threading
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimiter:
    """Thread-safe sliding-window rate limiter for client IP addresses."""

    def __init__(self, requests_limit: int = 60, window_sec: int = 60):
        self.requests_limit = requests_limit
        self.window_sec = window_sec
        self._requests = {}
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            # Fetch request history for this IP
            history = self._requests.get(ip, [])
            # Prune old requests outside the sliding window
            history = [t for t in history if now - t < self.window_sec]
            
            if len(history) >= self.requests_limit:
                self._requests[ip] = history
                return False
                
            history.append(now)
            self._requests[ip] = history
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that restricts request rates based on client IP."""

    def __init__(self, app, requests_limit: int = 60, window_sec: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(requests_limit, window_sec)

    async def dispatch(self, request: Request, call_next):
        # Bypass rate limits for health checks, metrics, and testing
        import config
        if request.url.path in ["/health", "/metrics"] or getattr(config, "TESTING", False):
            return await call_next(request)
            
        client_ip = request.client.host if request.client else "unknown"
        if not self.limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."}
            )
            
        return await call_next(request)
