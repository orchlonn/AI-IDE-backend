from supabase import create_client

from config import (
    SUPABASE_URL, SUPABASE_KEY,
    MATCH_THRESHOLD, MATCH_COUNT,
)
from rag.embed import embed_query


def retrieve_context(project_id: str, query: str) -> str:
    """Embed the query and retrieve relevant code chunks via pgvector similarity search."""
    if not project_id:
        return ""

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    query_embedding = embed_query(query)

    result = supabase.rpc("match_code_chunks", {
        "query_embedding": query_embedding,
        "match_project_id": project_id,
        "match_threshold": MATCH_THRESHOLD,
        "match_count": MATCH_COUNT,
    }).execute()

    matches = result.data or []
    context = "\n\n---\n\n".join(m["content"] for m in matches)
    return context
