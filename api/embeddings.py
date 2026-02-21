from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag.embeddings import index_project


router = APIRouter()


class EmbeddingsRequest(BaseModel):
    project_id: str


@router.post("/api/embeddings")
async def create_embeddings(req: EmbeddingsRequest):
    """Index a project's files into vector embeddings."""
    if not req.project_id:
        raise HTTPException(status_code=400, detail="Missing project_id")

    try:
        chunks_indexed = await index_project(req.project_id)
        return {"chunksIndexed": chunks_indexed}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
