import gzip
import json
import os

from evals.datasets.base import BenchmarkDataset, EvalProblem

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "HumanEval.jsonl.gz")

# Cache loaded problems
_problems_cache: dict | None = None


def _get_problems() -> dict:
    global _problems_cache
    if _problems_cache is None:
        with gzip.open(_DATA_PATH, "rt") as f:
            _problems_cache = {p["task_id"]: p for p in (json.loads(line) for line in f)}
    return _problems_cache


class HumanEvalDataset(BenchmarkDataset):
    def load(self) -> list[EvalProblem]:
        problems = _get_problems()
        result = []
        for task_id, problem in sorted(problems.items()):
            prompt = (
                "Complete the following Python function. "
                "Return ONLY the function body (the indented lines after the signature), "
                "no markdown fences, no explanation, no function signature.\n\n"
                f"{problem['prompt']}"
            )
            result.append(EvalProblem(
                task_id=task_id,
                prompt_for_model=prompt,
                test_code=problem["test"],
                entry_point=problem["entry_point"],
                reference_solution=problem["canonical_solution"],
                benchmark="humaneval",
            ))
        return result

    def build_test_harness(self, problem: EvalProblem, generated_code: str) -> str:
        original_prompt = _get_problems()[problem.task_id]["prompt"]
        full_function = original_prompt + generated_code
        return (
            f"{full_function}\n\n"
            f"{problem.test_code}\n"
            f"check({problem.entry_point})\n"
        )
