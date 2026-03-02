import re

from datasets import load_dataset

from evals.datasets.base import BenchmarkDataset, EvalProblem


def _extract_function_name(test_list: list[str]) -> str:
    """Extract the expected function name from MBPP test assertions.

    e.g. 'assert remove_Occ("hello","l") == "heo"' → 'remove_Occ'
    """
    for test in test_list:
        match = re.search(r"assert\s+(\w+)\s*\(", test)
        if match:
            return match.group(1)
    return ""


class MBPPDataset(BenchmarkDataset):
    def load(self) -> list[EvalProblem]:
        ds = load_dataset("mbpp", "sanitized", split="test")
        result = []
        for row in ds:
            task_id = f"mbpp/{row['task_id']}"
            func_name = _extract_function_name(row["test_list"])

            prompt = (
                "Write a Python function to solve the following problem. "
                "Return ONLY the Python code (function definition), "
                "no markdown fences, no explanation.\n\n"
                f"{row['prompt']}"
            )
            if func_name:
                prompt += f"\n\nThe function must be named `{func_name}`."

            test_code = "\n".join(row["test_list"])
            result.append(EvalProblem(
                task_id=task_id,
                prompt_for_model=prompt,
                test_code=test_code,
                entry_point=func_name,
                reference_solution=row["code"],
                benchmark="mbpp",
            ))
        return result

    def build_test_harness(self, problem: EvalProblem, generated_code: str) -> str:
        return f"{generated_code}\n\n{problem.test_code}\n"
