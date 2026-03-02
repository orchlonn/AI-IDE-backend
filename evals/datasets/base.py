from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class EvalProblem:
    """Normalized representation of a benchmark problem."""
    task_id: str
    prompt_for_model: str  # Text sent to the LLM
    test_code: str         # Test code to run against output
    entry_point: str       # Function name to test (HumanEval) or "" (MBPP)
    reference_solution: str
    benchmark: str         # "humaneval" or "mbpp"


class BenchmarkDataset(ABC):
    @abstractmethod
    def load(self) -> list[EvalProblem]:
        ...

    @abstractmethod
    def build_test_harness(self, problem: EvalProblem, generated_code: str) -> str:
        """Build the full executable test string from generated code + test cases."""
        ...
