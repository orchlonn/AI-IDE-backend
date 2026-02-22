import time
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import OPENAI_API_KEY, CHAT_MODEL
from agents.state import AgentState

logger = logging.getLogger(__name__)


def generate_code(state: AgentState) -> dict:
    """Code Generator agent node. Generates code based on prompt + RAG context."""
    iteration = state.get("iteration", 0) + 1
    logger.info("[GENERATOR] Generating code (iteration %d)...", iteration)

    llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    # Build system prompt
    system_content = """You are an expert code generator. Your job is to write high-quality, complete code based on the user's request.

Rules:
- Always provide the COMPLETE file content in a code block (not just a snippet)
- Add `// file: <filepath>` as the very first line of the code block so the system knows which file to update
- If modifying the currently open file, use its path
- Write clean, well-structured code
- Follow best practices for the language being used"""

    if state["current_file_path"] and state["current_file_content"]:
        system_content += f"\n\n## Currently Open File: {state['current_file_path']}\n{state['current_file_content']}"

    if state["rag_context"]:
        logger.debug("[GENERATOR] RAG context: %d chars", len(state["rag_context"]))
        system_content += f"\n\n## Related Code Context\n{state['rag_context']}"

    # If there's a plan from the planner agent, include it
    if state.get("plan"):
        logger.info("[GENERATOR] Following Planner's implementation plan (%d chars)", len(state["plan"]))
        system_content += f"\n\n## Implementation Plan\nFollow this plan created by the architect. Implement each step precisely:\n{state['plan']}"

    # If there's review feedback from a previous iteration, include it
    if state.get("review_feedback") and state["iteration"] > 0:
        logger.info("[GENERATOR] Applying Reviewer's feedback from iteration %d", state["iteration"])
        system_content += f"\n\n## Review Feedback (Iteration {state['iteration']})\nThe code reviewer found issues with your previous attempt. Address this feedback:\n{state['review_feedback']}"

    messages = [SystemMessage(content=system_content)]

    # Add conversation history
    for msg in state.get("conversation_history", [])[-20:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(SystemMessage(content=msg["content"]))

    messages.append(HumanMessage(content=state["user_prompt"]))

    start = time.time()
    response = llm.invoke(messages)
    duration_ms = (time.time() - start) * 1000

    logger.info("[GENERATOR] Code generated (%d chars, %.0fms)", len(response.content), duration_ms)

    return {
        "generated_code": response.content,
        "iteration": iteration,
    }
