import time
import logging

from openai import OpenAI
from supabase import create_client

from config import (
    OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_BATCH_SIZE, INSERT_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


def chunk_file(file_path: str, content: str) -> list[dict]:
    """Split a file into overlapping chunks with headers."""
    lines = content.split("\n")

    if len(lines) <= CHUNK_SIZE:
        return [{
            "file_path": file_path,
            "chunk_index": 0,
            "content": f"// {file_path}\n{content}",
        }]

    chunks = []
    start = 0
    index = 0
    while start < len(lines):
        end = min(start + CHUNK_SIZE, len(lines))
        slice_content = "\n".join(lines[start:end])
        header = f"// {file_path} (lines {start + 1}-{end})"
        chunks.append({
            "file_path": file_path,
            "chunk_index": index,
            "content": f"{header}\n{slice_content}",
        })
        index += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_project(file_contents: dict[str, str]) -> list[dict]:
    """Chunk all files in a project."""
    all_chunks = []
    for path, content in file_contents.items():
        all_chunks.extend(chunk_file(path, content))
    return all_chunks


async def index_project(project_id: str) -> int:
    """Index a project's files into vector embeddings. Returns chunk count."""
    logger.info("Indexing project  id=%s", project_id)
    openai = OpenAI(api_key=OPENAI_API_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Load project file contents
    result = supabase.table("projects").select("file_contents").eq("id", project_id).single().execute()
    if not result.data:
        raise ValueError("Project not found")

    file_contents: dict[str, str] = result.data.get("file_contents", {})
    logger.info("Project loaded  files=%d", len(file_contents))

    chunks = chunk_project(file_contents)
    logger.info("Chunking done  total_chunks=%d", len(chunks))

    if not chunks:
        return 0

    # Generate embeddings in batches
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        start = time.time()
        res = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[c["content"] for c in batch],
        )
        duration_ms = (time.time() - start) * 1000
        for item in res.data:
            embeddings.append(item.embedding)
        logger.info("Embeddings batch  %d/%d  batch_size=%d  duration=%.0fms",
                     i + len(batch), len(chunks), len(batch), duration_ms)

    # Delete old chunks
    supabase.table("code_chunks").delete().eq("project_id", project_id).execute()
    logger.info("Old chunks deleted  project=%s", project_id)

    # Insert new chunks in batches
    rows = [
        {
            "project_id": project_id,
            "file_path": chunk["file_path"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "embedding": embeddings[i],
        }
        for i, chunk in enumerate(chunks)
    ]

    for i in range(0, len(rows), INSERT_BATCH_SIZE):
        batch = rows[i:i + INSERT_BATCH_SIZE]
        supabase.table("code_chunks").insert(batch).execute()
        logger.info("Insert batch  %d/%d", i + len(batch), len(rows))

    logger.info("Indexing complete  project=%s  chunks=%d", project_id, len(chunks))
    return len(chunks)
