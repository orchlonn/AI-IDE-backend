from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.graph import agent_graph


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

    current_path = ""
    current_content = ""
    if req.current_file:
        current_path = req.current_file.get("path", "")
        current_content = req.current_file.get("content", "")

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
    result = agent_graph.invoke(initial_state)
    final_response = result.get("final_response", "")

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
