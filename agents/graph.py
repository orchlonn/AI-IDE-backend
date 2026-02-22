import logging

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.generator import generate_code
from agents.reviewer import review_code
from agents.router import route_task
from agents.planner import plan_code
from rag.retriever import retrieve_context
from config import MAX_AGENT_ITERATIONS

logger = logging.getLogger(__name__)


def retrieve_context_node(state: AgentState) -> dict:
    """Retrieve RAG context for the user's prompt."""
    logger.info("[1/RAG] Retrieving context for project=%s", state["project_id"])
    context = retrieve_context(state["project_id"], state["user_prompt"])
    logger.info("[1/RAG] Retrieved %d chars of context", len(context))
    return {"rag_context": context}


def route_by_complexity(state: AgentState) -> str:
    """Route to planner or directly to generator based on task complexity."""
    complexity = state.get("task_complexity", "simple")
    if complexity == "complex":
        logger.info("[2/ROUTER] --> COMPLEX: routing to Planner first")
    else:
        logger.info("[2/ROUTER] --> SIMPLE: skipping Planner, going straight to Generator")
    return complexity


def should_continue(state: AgentState) -> str:
    """Decide whether to loop back to generator or finish."""
    if state["review_decision"] == "APPROVE":
        logger.info("[REVIEWER] --> APPROVED at iteration %d, finalizing", state["iteration"])
        return "approved"
    if state["iteration"] >= MAX_AGENT_ITERATIONS:
        logger.warning("[REVIEWER] --> MAX ITERATIONS reached (%d), finalizing with last output", state["iteration"])
        return "max_iterations"
    logger.info("[REVIEWER] --> REVISE: sending back to Generator (iteration %d)", state["iteration"])
    return "revise"


def finalize(state: AgentState) -> dict:
    """Produce final response from the approved/final generated code."""
    logger.info("[FINAL] Response ready (%d chars, %d iterations)", len(state["generated_code"]), state["iteration"])
    return {"final_response": state["generated_code"]}


def build_agent_graph() -> StateGraph:
    """Build the LangGraph state graph for the code generation pipeline.

    Flow:
        START → retrieve_context → route_task →
            (simple)  → generate_code → review_code → ...
            (complex) → plan_code → generate_code → review_code → ...
        review_code →
            (APPROVE or max iterations) → finalize → END
            (REVISE) → generate_code (loop)
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("route_task", route_task)
    graph.add_node("plan_code", plan_code)
    graph.add_node("generate_code", generate_code)
    graph.add_node("review_code", review_code)
    graph.add_node("finalize", finalize)

    # Set entry point
    graph.set_entry_point("retrieve_context")

    # RAG → Router
    graph.add_edge("retrieve_context", "route_task")

    # Router → conditional: skip or go through planner
    graph.add_conditional_edges(
        "route_task",
        route_by_complexity,
        {
            "simple": "generate_code",
            "complex": "plan_code",
        },
    )

    # Planner → Generator
    graph.add_edge("plan_code", "generate_code")

    # Generator → Reviewer
    graph.add_edge("generate_code", "review_code")

    # Conditional edge: review → finalize or loop back to generator
    graph.add_conditional_edges(
        "review_code",
        should_continue,
        {
            "approved": "finalize",
            "max_iterations": "finalize",
            "revise": "generate_code",
        },
    )

    graph.add_edge("finalize", END)

    return graph.compile()


# Pre-compiled graph instance
agent_graph = build_agent_graph()
