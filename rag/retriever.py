from openai import OpenAI
from supabase import create_client

from config import (
    OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY,
    EMBEDDING_MODEL, MATCH_THRESHOLD, MATCH_COUNT,
)


def retrieve_context(project_id: str, query: str) -> str:
    """Embed the query and retrieve relevant code chunks via pgvector similarity search."""
    openai = OpenAI(api_key=OPENAI_API_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    emb_res = openai.embeddings.create(model=EMBEDDING_MODEL, input=query)
    query_embedding = emb_res.data[0].embedding

    result = supabase.rpc("match_code_chunks", {
        "query_embedding": query_embedding,
        "match_project_id": project_id,
        "match_threshold": MATCH_THRESHOLD,
        "match_count": MATCH_COUNT,
    }).execute()

    matches = result.data or []
    context = "\n\n---\n\n".join(m["content"] for m in matches)
    return context
