import time
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import OPENAI_API_KEY, CHAT_MODEL
from agents.state import AgentState

logger = logging.getLogger(__name__)


def plan_code(state: AgentState) -> dict:
    """Planner agent node. Creates an implementation plan for complex tasks."""
    logger.info("Planner  prompt=%.80s", state["user_prompt"])

    llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    system_content = """You are an expert software architect and planner. Your job is to analyze a coding request and create a clear, actionable implementation plan that a code generator will follow.

Your plan should include:
1. **Overview**: Brief summary of what needs to be done
2. **Files to modify/create**: List each file with what changes are needed
3. **Step-by-step approach**: Ordered steps the generator should follow
4. **Key considerations**: Edge cases, dependencies, or important decisions

Rules:
- Be specific and actionable — the generator will follow your plan literally
- Reference actual file paths and function names from the codebase context when available
- Keep the plan concise but complete
- Focus on the *what* and *why*, not the exact code (the generator handles that)
- If the codebase context shows existing patterns, instruct the generator to follow them"""

    if state["current_file_path"] and state["current_file_content"]:
        system_content += f"\n\n## Currently Open File: {state['current_file_path']}\n{state['current_file_content']}"

    if state["rag_context"]:
        logger.debug("Planner  RAG context length: %d chars", len(state["rag_context"]))
        system_content += f"\n\n## Related Code Context\n{state['rag_context']}"

    messages = [SystemMessage(content=system_content)]

    # Add conversation history for context
    for msg in state.get("conversation_history", [])[-10:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(SystemMessage(content=msg["content"]))

    messages.append(HumanMessage(content=state["user_prompt"]))

    logger.debug("Planner  sending %d messages to LLM", len(messages))
    start = time.time()
    response = llm.invoke(messages)
    duration_ms = (time.time() - start) * 1000

    logger.info("Planner  done  plan_len=%d  duration=%.0fms", len(response.content), duration_ms)

    return {"plan": response.content}
