from typing import TypedDict, Literal


class AgentState(TypedDict):
    """State that flows through the LangGraph agent loop."""
    # Input
    user_prompt: str
    project_id: str
    rag_context: str
    current_file_path: str
    current_file_content: str
    conversation_history: list[dict]

    # Agent working state
    generated_code: str
    review_feedback: str
    review_decision: Literal["APPROVE", "REVISE"]
    iteration: int

    # Output
    final_response: str
