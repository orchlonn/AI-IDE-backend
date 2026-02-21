from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import OPENAI_API_KEY, CHAT_MODEL
from agents.state import AgentState


def generate_code(state: AgentState) -> dict:
    """Code Generator agent node. Generates code based on prompt + RAG context."""
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
        system_content += f"\n\n## Related Code Context\n{state['rag_context']}"

    # If there's review feedback from a previous iteration, include it
    if state.get("review_feedback") and state["iteration"] > 0:
        system_content += f"\n\n## Review Feedback (Iteration {state['iteration']})\nThe code reviewer found issues with your previous attempt. Address this feedback:\n{state['review_feedback']}"

    messages = [SystemMessage(content=system_content)]

    # Add conversation history
    for msg in state.get("conversation_history", [])[-20:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(SystemMessage(content=msg["content"]))

    messages.append(HumanMessage(content=state["user_prompt"]))

    response = llm.invoke(messages)

    return {
        "generated_code": response.content,
        "iteration": state.get("iteration", 0) + 1,
    }
