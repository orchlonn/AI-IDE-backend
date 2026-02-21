import time
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import OPENAI_API_KEY, CHAT_MODEL
from agents.state import AgentState

logger = logging.getLogger(__name__)


def review_code(state: AgentState) -> dict:
    """Code Reviewer agent node. Reviews generated code and decides APPROVE or REVISE."""
    logger.info("Reviewer  iteration=%d  code_len=%d", state.get("iteration", 0), len(state.get("generated_code", "")))

    llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    system_content = """You are an expert code reviewer. Your job is to review generated code for quality and correctness.

Evaluate the code on:
1. **Correctness**: Does it fulfill the user's request?
2. **Completeness**: Is the full file content provided with the `// file: <filepath>` header?
3. **Code Quality**: Is it clean, readable, and well-structured?
4. **Security**: Are there any obvious security vulnerabilities?
5. **Best Practices**: Does it follow language conventions and best practices?

You MUST respond in EXACTLY this format:

DECISION: APPROVE
FEEDBACK: The code looks good. [brief explanation]

OR

DECISION: REVISE
FEEDBACK: [specific issues that need to be fixed, be actionable and clear]

Important: Only request revision for real issues. Minor style preferences are not grounds for revision. If the code is functionally correct and reasonably clean, APPROVE it."""

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
    response = llm.invoke(messages)
    duration_ms = (time.time() - start) * 1000
    content = response.content

    # Parse decision and feedback
    decision = "APPROVE"
    feedback = content

    if "DECISION: REVISE" in content:
        decision = "REVISE"
    elif "DECISION: APPROVE" in content:
        decision = "APPROVE"

    # Extract feedback after "FEEDBACK:" marker
    if "FEEDBACK:" in content:
        feedback = content.split("FEEDBACK:", 1)[1].strip()

    logger.info("Reviewer  decision=%s  duration=%.0fms  feedback=%.120s", decision, duration_ms, feedback)

    return {
        "review_decision": decision,
        "review_feedback": feedback,
    }
