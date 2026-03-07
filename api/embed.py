from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag.embed import embed_texts, embed_query

router = APIRouter()


class EmbedRequest(BaseModel):
    texts: list[str]
    prefix: str = "passage: "


class EmbedQueryRequest(BaseModel):
    text: str


@router.post("/api/embed")
async def embed(req: EmbedRequest):
    """Generate embeddings for a list of texts."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="Missing texts")
    embeddings = embed_texts(req.texts, prefix=req.prefix)
    return {"embeddings": embeddings}


@router.post("/api/embed-query")
async def embed_single_query(req: EmbedQueryRequest):
    """Generate an embedding for a single search query."""
    if not req.text:
        raise HTTPException(status_code=400, detail="Missing text")
    embedding = embed_query(req.text)
    return {"embedding": embedding}
