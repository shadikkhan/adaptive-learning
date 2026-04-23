"""
Main FastAPI application for AgeXPlain
"""
import time
import uuid
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()  # loads agexplain/backend/.env into os.environ

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from configs import config
from api import routes
from services.json_logger import log_event


def _http_log_skip_prefixes() -> tuple[str, ...]:
    raw = os.getenv("HTTP_LOG_SKIP_PATH_PREFIXES", "/audio/")
    parts = [segment.strip() for segment in raw.split(",") if segment.strip()]
    return tuple(parts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise SQLite and reload RAG indices from persisted documents."""
    from db.database import init_db
    from services.rag_service import reload_indices_from_db
    from services.tts_service import cleanup_audio_files

    init_db()
    reload_indices_from_db()
    cleanup_audio_files(force=True)
    yield


# Initialize FastAPI app
app = FastAPI(title="AgeXplain API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware - must be before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_CREDENTIALS,
    allow_methods=config.CORS_METHODS,
    allow_headers=config.CORS_HEADERS,
    expose_headers=config.CORS_EXPOSE_HEADERS
)

# Include API routes
app.include_router(routes.router)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    skip_http_log = request.url.path.startswith(_http_log_skip_prefixes())
    if not skip_http_log:
        log_event("http.start", request_id=request_id, method=request.method, path=request.url.path)
    try:
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if not skip_http_log:
            log_event(
                "http.end",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                elapsed_ms=elapsed_ms,
            )
        response.headers["X-Request-Id"] = request_id
        return response
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            "http.error",
            level="ERROR",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            elapsed_ms=elapsed_ms,
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, access_log=False)

