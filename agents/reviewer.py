import time
import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from config import OPENAI_API_KEY, CHAT_MODEL
from agents.state import AgentState

logger = logging.getLogger(__name__)


class ReviewOutput(BaseModel):
    """Structured output for the reviewer agent."""
    decision: Literal["APPROVE", "REVISE"] = Field(description="Whether to approve the code or request revision")
    feedback: str = Field(description="Explanation of the decision with specific issues if revising")


def review_code(state: AgentState) -> dict:
    """Code Reviewer agent node. Reviews generated code and decides APPROVE or REVISE."""
    logger.info("[REVIEWER] Reviewing code (iteration %d, %d chars)...", state.get("iteration", 0), len(state.get("generated_code", "")))

    llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    structured_llm = llm.with_structured_output(ReviewOutput)

    system_content = """You are an expert code reviewer. Your job is to review generated code for quality and correctness.

Evaluate the code on:
1. **Correctness**: Does it fulfill the user's request?
2. **Completeness**: Is the full file content provided with the `// file: <filepath>` header?
3. **Code Quality**: Is it clean, readable, and well-structured?
4. **Security**: Are there any obvious security vulnerabilities?
5. **Best Practices**: Does it follow language conventions and best practices?

Important: Only request revision for real issues. Minor style preferences are not grounds for revision. If the code is functionally correct and reasonably clean, approve it."""

    user_content = f"""## User's Original Request
{state['user_prompt']}

## Generated Code
{state['generated_code']}"""

    if state["current_file_path"] and state["current_file_content"]:
        user_content += f"\n\n## Original File: {state['current_file_path']}\n{state['current_file_content']}"

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    start = time.time()
    response = structured_llm.invoke(messages)
    duration_ms = (time.time() - start) * 1000

    logger.info("[REVIEWER] Decision: %s (%.0fms)", response.decision, duration_ms)
    logger.info("[REVIEWER] Feedback: %s", response.feedback[:150])

    return {
        "review_decision": response.decision,
        "review_feedback": response.feedback,
    }
