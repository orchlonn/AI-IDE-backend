import time
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import config  # noqa: F401  — initialises logging on import

from api.chat import router as chat_router
from api.embeddings import router as embeddings_router
from api.terminal import router as terminal_router

logger = logging.getLogger(__name__)

app = FastAPI(title="AI IDE Backend", version="1.0.0")

# CORS - allow the Next.js frontend
origins = ["http://localhost:3000", "http://localhost:3001"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS origins: %s", origins)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info("--> %s %s", request.method, request.url.path)
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info("<-- %s %s %s (%.0fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


# Register routes
app.include_router(chat_router)
app.include_router(embeddings_router)
app.include_router(terminal_router)

logger.info("AI IDE Backend started — routes registered")


@app.get("/health")
async def health():
    return {"status": "ok"}
