import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


def setup_logging() -> None:
    """Initialize structured basic logging format for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Ensure uvicorn logs are structured cleanly
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class LoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware to measure and log HTTP request latencies."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logging.info(
            f"{request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Duration: {process_time:.4f}s"
        )
        return response
