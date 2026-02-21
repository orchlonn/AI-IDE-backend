import time
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag.embeddings import index_project

logger = logging.getLogger(__name__)

router = APIRouter()


class EmbeddingsRequest(BaseModel):
    project_id: str


@router.post("/api/embeddings")
async def create_embeddings(req: EmbeddingsRequest):
    """Index a project's files into vector embeddings."""
    if not req.project_id:
        raise HTTPException(status_code=400, detail="Missing project_id")

    logger.info("Embeddings request  project=%s", req.project_id)
    start = time.time()

    try:
        chunks_indexed = await index_project(req.project_id)
        duration_ms = (time.time() - start) * 1000
        logger.info("Embeddings done  project=%s  chunks=%d  duration=%.0fms",
                     req.project_id, chunks_indexed, duration_ms)
        return {"chunksIndexed": chunks_indexed}
    except ValueError as e:
        logger.warning("Project not found: %s", req.project_id)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Embeddings failed  project=%s", req.project_id)
        raise HTTPException(status_code=500, detail="Embedding indexing error")
