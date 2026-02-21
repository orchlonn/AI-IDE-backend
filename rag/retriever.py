import time
import logging

from openai import OpenAI
from supabase import create_client

from config import (
    OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY,
    EMBEDDING_MODEL, MATCH_THRESHOLD, MATCH_COUNT,
)

logger = logging.getLogger(__name__)


def retrieve_context(project_id: str, query: str) -> str:
    """Embed the query and retrieve relevant code chunks via pgvector similarity search."""
    logger.info("Retriever  project=%s  query=%.80s", project_id, query)

    openai = OpenAI(api_key=OPENAI_API_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Embed the query
    start = time.time()
    emb_res = openai.embeddings.create(model=EMBEDDING_MODEL, input=query)
    query_embedding = emb_res.data[0].embedding
    embed_ms = (time.time() - start) * 1000
    logger.debug("Retriever  query embedded  duration=%.0fms", embed_ms)

    # Vector similarity search
    start = time.time()
    result = supabase.rpc("match_code_chunks", {
        "query_embedding": query_embedding,
        "match_project_id": project_id,
        "match_threshold": MATCH_THRESHOLD,
        "match_count": MATCH_COUNT,
    }).execute()
    search_ms = (time.time() - start) * 1000

    matches = result.data or []
    logger.info("Retriever  matches=%d  search_duration=%.0fms", len(matches), search_ms)

    for m in matches:
        logger.debug("Retriever  match  file=%s  similarity=%.3f",
                      m.get("file_path", "?"), m.get("similarity", 0))

    context = "\n\n---\n\n".join(m["content"] for m in matches)
    return context
