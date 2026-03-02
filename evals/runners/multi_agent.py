import time

from evals.datasets.base import EvalProblem
from evals.runners.base import Runner, RunResult
from evals.extraction.code_extractor import extract_function_body, extract_complete_function
from agents.graph import agent_graph


class MultiAgentRunner(Runner):
    """Runs problems through the full LangGraph multi-agent pipeline."""

    def run(self, problem: EvalProblem) -> RunResult:
        start = time.time()

        initial_state = {
            "user_prompt": problem.prompt_for_model,
            "project_id": "",
            "rag_context": "",
            "current_file_path": "",
            "current_file_content": "",
            "conversation_history": [],
            "task_complexity": "",
            "plan": "",
            "generated_code": "",
            "review_feedback": "",
            "review_decision": "",
            "iteration": 0,
            "final_response": "",
        }

        try:
            result = agent_graph.invoke(initial_state)
            raw_code = result.get("final_response", "")
            iterations = result.get("iteration", 1)
        except Exception as e:
            return RunResult(
                task_id=problem.task_id,
                generated_code="",
                extracted_code="",
                iterations=0,
                duration_seconds=time.time() - start,
                error=str(e),
            )

        if problem.benchmark == "humaneval":
            extracted = extract_function_body(raw_code)
        else:
            extracted = extract_complete_function(raw_code)

        return RunResult(
            task_id=problem.task_id,
            generated_code=raw_code,
            extracted_code=extracted,
            iterations=iterations,
            duration_seconds=time.time() - start,
        )
