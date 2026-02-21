from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from api.embeddings import router as embeddings_router
from api.terminal import router as terminal_router

app = FastAPI(title="AI IDE Backend", version="1.0.0")

# CORS - allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(chat_router)
app.include_router(embeddings_router)
app.include_router(terminal_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
