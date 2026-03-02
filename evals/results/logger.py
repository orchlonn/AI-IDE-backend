import json
import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TaskResult:
    """Complete result for a single eval task."""
    task_id: str
    benchmark: str
    mode: str
    model: str
    passed: bool
    generated_code: str
    extracted_code: str
    iterations: int
    duration_seconds: float
    pipeline_error: Optional[str]
    exec_error: Optional[str]
    exec_traceback: Optional[str]
    timestamp: str


class ResultLogger:
    """Accumulates TaskResults and writes them to a JSONL file."""

    def __init__(self, output_dir: str, run_id: str):
        self.output_dir = output_dir
        self.run_id = run_id
        self.results: list[TaskResult] = []
        os.makedirs(output_dir, exist_ok=True)
        self._filepath = os.path.join(output_dir, f"{run_id}_details.jsonl")

    def log(self, result: TaskResult):
        self.results.append(result)
        # Append immediately so results survive crashes
        with open(self._filepath, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    @property
    def filepath(self) -> str:
        return self._filepath
