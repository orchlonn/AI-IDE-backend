from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.generator import generate_code
from agents.reviewer import review_code
from rag.retriever import retrieve_context
from config import MAX_AGENT_ITERATIONS


def retrieve_context_node(state: AgentState) -> dict:
    """Retrieve RAG context for the user's prompt."""
    context = retrieve_context(state["project_id"], state["user_prompt"])
    return {"rag_context": context}


def should_continue(state: AgentState) -> str:
    """Decide whether to loop back to generator or finish."""
    if state["review_decision"] == "APPROVE":
        return "approved"
    if state["iteration"] >= MAX_AGENT_ITERATIONS:
        return "max_iterations"
    return "revise"


def finalize(state: AgentState) -> dict:
    """Produce final response from the approved/final generated code."""
    return {"final_response": state["generated_code"]}


def build_agent_graph() -> StateGraph:
    """Build the LangGraph state graph for the code generation pipeline.

    Flow:
        START → retrieve_context → generate_code → review_code →
            (APPROVE or max iterations) → finalize → END
            (REVISE) → generate_code (loop)
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_code", generate_code)
    graph.add_node("review_code", review_code)
    graph.add_node("finalize", finalize)

    # Set entry point
    graph.set_entry_point("retrieve_context")

    # Add edges
    graph.add_edge("retrieve_context", "generate_code")
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
