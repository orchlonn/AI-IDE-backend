import time
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.graph import agent_graph

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    project_id: str
    question: str
    history: list[dict] = []
    current_file: dict | None = None


@router.post("/api/chat")
async def chat(req: ChatRequest):
    """Run the agent loop (generator → reviewer → iterate) and stream the final result."""
    if not req.project_id or not req.question:
        raise HTTPException(status_code=400, detail="Missing project_id or question")

    logger.info("Chat request  project=%s  question=%.100s  history_len=%d",
                req.project_id, req.question, len(req.history))

    current_path = ""
    current_content = ""
    if req.current_file:
        current_path = req.current_file.get("path", "")
        current_content = req.current_file.get("content", "")
        logger.info("Current file: %s (%d chars)", current_path, len(current_content))

    # Build initial state for the agent graph
    initial_state = {
        "user_prompt": req.question,
        "project_id": req.project_id,
        "rag_context": "",
        "current_file_path": current_path,
        "current_file_content": current_content,
        "conversation_history": req.history,
        "generated_code": "",
        "review_feedback": "",
        "review_decision": "",
        "iteration": 0,
        "final_response": "",
    }

    # Run the agent graph to completion
    start = time.time()
    try:
        result = agent_graph.invoke(initial_state)
    except Exception:
        logger.exception("Agent graph failed for project=%s", req.project_id)
        raise HTTPException(status_code=500, detail="Agent pipeline error")

    duration_ms = (time.time() - start) * 1000
    final_response = result.get("final_response", "")
    logger.info("Agent graph done  iterations=%d  response_len=%d  duration=%.0fms",
                result.get("iteration", 0), len(final_response), duration_ms)

    # Stream the final response back to the client (matching existing frontend contract)
    async def stream_response():
        # Send in chunks to match the streaming behavior the frontend expects
        chunk_size = 50
        for i in range(0, len(final_response), chunk_size):
            yield final_response[i:i + chunk_size]

    return StreamingResponse(
        stream_response(),
        media_type="text/plain; charset=utf-8",
        headers={"Transfer-Encoding": "chunked"},
    )
