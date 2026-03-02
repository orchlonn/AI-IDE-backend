from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EvalConfig:
    benchmark: Literal["humaneval", "mbpp"] = "humaneval"
    mode: Literal["baseline", "multi_agent"] = "baseline"
    baseline_model: str = "gpt-4.1-mini"
    limit: int = 0  # 0 = all problems
    offset: int = 0
    task_ids: list[str] = field(default_factory=list)
    exec_timeout: int = 10  # seconds per test execution
    output_dir: str = "evals/output"
    temperature: float = 0.0
