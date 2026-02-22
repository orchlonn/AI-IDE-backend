import time
import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from config import OPENAI_API_KEY, ROUTER_MODEL
from agents.state import AgentState

logger = logging.getLogger(__name__)


class RouterOutput(BaseModel):
    """Structured output for the router agent."""
    complexity: Literal["simple", "complex"] = Field(description="Task complexity classification")
    reason: str = Field(description="One sentence explanation for the classification")


def route_task(state: AgentState) -> dict:
    """Router agent node. Classifies the task as simple or complex."""
    logger.info("[2/ROUTER] Classifying task complexity...")

    llm = ChatOpenAI(model=ROUTER_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    structured_llm = llm.with_structured_output(RouterOutput)

    system_content = """You are a task complexity classifier for a code editor AI assistant. Your job is to decide whether a coding request is SIMPLE or COMPLEX.

SIMPLE tasks (skip planning, go straight to coding):
- Single-file edits or small changes
- Fixing typos, renaming variables, formatting
- Adding a log statement or console.log
- Writing a single function with clear requirements
- Small bug fixes with obvious solutions
- Adding comments or docstrings
- Simple Q&A or explanations about code

COMPLEX tasks (need an implementation plan first):
- Multi-file changes or new features
- Refactoring or restructuring code
- Building new modules, APIs, or components
- Architecture changes or design decisions
- Tasks requiring multiple steps or dependencies
- Performance optimization across files
- Adding authentication, database schemas, or integrations"""

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=state["user_prompt"]),
    ]

    start = time.time()
    response = structured_llm.invoke(messages)
    duration_ms = (time.time() - start) * 1000

    logger.info("[2/ROUTER] Decision: %s (%.0fms) - %s", response.complexity.upper(), duration_ms, response.reason)

    return {"task_complexity": response.complexity}
