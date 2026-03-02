import time

from openai import OpenAI

from config import OPENAI_API_KEY
from evals.config import EvalConfig
from evals.datasets.base import EvalProblem
from evals.runners.base import Runner, RunResult
from evals.extraction.code_extractor import extract_function_body, extract_complete_function


class BaselineRunner(Runner):
    """Direct single-call to OpenAI. No router, no reviewer, no RAG."""

    def __init__(self, config: EvalConfig):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = config.baseline_model
        self.temperature = config.temperature

    def run(self, problem: EvalProblem) -> RunResult:
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert Python programmer. "
                        "Return ONLY code, no markdown fences, no explanation."
                    )},
                    {"role": "user", "content": problem.prompt_for_model},
                ],
            )
            raw_code = response.choices[0].message.content or ""
        except Exception as e:
            return RunResult(
                task_id=problem.task_id,
                generated_code="",
                extracted_code="",
                iterations=1,
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
            iterations=1,
            duration_seconds=time.time() - start,
        )
