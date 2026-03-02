from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from evals.datasets.base import EvalProblem


@dataclass
class RunResult:
    """Raw output from running a single problem through the pipeline."""
    task_id: str
    generated_code: str     # Raw LLM output before extraction
    extracted_code: str     # Code after extraction
    iterations: int         # Number of generate/review loops (1 for baseline)
    duration_seconds: float
    error: Optional[str] = None


class Runner(ABC):
    @abstractmethod
    def run(self, problem: EvalProblem) -> RunResult:
        ...
