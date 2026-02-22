import time
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import OPENAI_API_KEY, ROUTER_MODEL
from agents.state import AgentState

logger = logging.getLogger(__name__)


def route_task(state: AgentState) -> dict:
    """Router agent node. Classifies the task as simple or complex."""
    logger.info("Router  prompt=%.80s", state["user_prompt"])

    llm = ChatOpenAI(model=ROUTER_MODEL, api_key=OPENAI_API_KEY, temperature=0)

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
- Adding authentication, database schemas, or integrations

You MUST respond in EXACTLY this format:

COMPLEXITY: SIMPLE
REASON: [one sentence explanation]

OR

COMPLEXITY: COMPLEX
REASON: [one sentence explanation]"""

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=state["user_prompt"]),
    ]

    start = time.time()
    response = llm.invoke(messages)
    duration_ms = (time.time() - start) * 1000
    content = response.content

    # Parse complexity
    complexity = "simple"
    if "COMPLEXITY: COMPLEX" in content:
        complexity = "complex"
    elif "COMPLEXITY: SIMPLE" in content:
        complexity = "simple"

    logger.info("Router  decision=%s  duration=%.0fms  response=%s", complexity, duration_ms, content.strip())

    return {"task_complexity": complexity}
