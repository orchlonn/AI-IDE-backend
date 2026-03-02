"""
Evaluation harness CLI.

Usage:
    python -m evals.run --benchmark humaneval --mode baseline --limit 5
    python -m evals.run --benchmark humaneval --mode multi_agent
    python -m evals.run --benchmark mbpp --mode baseline --limit 20
    python -m evals.run --benchmark humaneval --mode baseline --tasks HumanEval/0,HumanEval/1
"""
import argparse
import sys
from datetime import datetime

# Load .env and configure logging before anything else
import config as app_config  # noqa: F401

from evals.config import EvalConfig
from evals.datasets.humaneval import HumanEvalDataset
from evals.datasets.mbpp import MBPPDataset
from evals.runners.baseline import BaselineRunner
from evals.runners.multi_agent import MultiAgentRunner
from evals.execution.sandbox import execute_with_timeout
from evals.results.logger import ResultLogger, TaskResult
from evals.results.reporter import Reporter


def main():
    parser = argparse.ArgumentParser(description="Run code generation benchmarks")
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--mode", choices=["baseline", "multi_agent"], default="baseline")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model for baseline mode")
    parser.add_argument("--limit", type=int, default=0, help="Max problems to run (0=all)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N problems")
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated task IDs")
    parser.add_argument("--timeout", type=int, default=10, help="Execution timeout in seconds")
    parser.add_argument("--output-dir", default="evals/output")
    args = parser.parse_args()

    eval_config = EvalConfig(
        benchmark=args.benchmark,
        mode=args.mode,
        baseline_model=args.model,
        limit=args.limit,
        offset=args.offset,
        task_ids=[t.strip() for t in args.tasks.split(",") if t.strip()],
        exec_timeout=args.timeout,
        output_dir=args.output_dir,
    )

    # Load dataset
    if eval_config.benchmark == "humaneval":
        dataset = HumanEvalDataset()
    else:
        dataset = MBPPDataset()

    problems = dataset.load()

    # Apply filtering
    if eval_config.task_ids:
        problems = [p for p in problems if p.task_id in eval_config.task_ids]
    else:
        problems = problems[eval_config.offset:]
        if eval_config.limit > 0:
            problems = problems[:eval_config.limit]

    if not problems:
        print("No problems to run. Check your filters.")
        sys.exit(1)

    # Select runner
    if eval_config.mode == "baseline":
        runner = BaselineRunner(eval_config)
    else:
        runner = MultiAgentRunner()

    # Set up logging
    run_id = f"{eval_config.benchmark}_{eval_config.mode}_{datetime.now():%Y%m%d_%H%M%S}"
    result_logger = ResultLogger(eval_config.output_dir, run_id)

    model_name = eval_config.baseline_model if eval_config.mode == "baseline" else app_config.CHAT_MODEL

    print(f"Starting eval: {eval_config.benchmark} / {eval_config.mode} / {model_name}")
    print(f"Problems: {len(problems)}")
    print(f"Run ID: {run_id}")
    print("-" * 60)

    # Main eval loop
    for i, problem in enumerate(problems):
        print(f"[{i + 1}/{len(problems)}] {problem.task_id} ... ", end="", flush=True)

        # Step 1: Run through pipeline
        run_result = runner.run(problem)

        # Step 2: Build test harness and execute
        passed = False
        exec_error = None
        exec_tb = None

        if run_result.error:
            exec_error = f"Pipeline error: {run_result.error}"
        else:
            test_code = dataset.build_test_harness(problem, run_result.extracted_code)
            exec_result = execute_with_timeout(test_code, timeout=eval_config.exec_timeout)
            passed = exec_result.passed
            exec_error = exec_result.error
            exec_tb = exec_result.traceback

        # Step 3: Log result
        task_result = TaskResult(
            task_id=problem.task_id,
            benchmark=eval_config.benchmark,
            mode=eval_config.mode,
            model=model_name,
            passed=passed,
            generated_code=run_result.generated_code,
            extracted_code=run_result.extracted_code,
            iterations=run_result.iterations,
            duration_seconds=round(run_result.duration_seconds, 2),
            pipeline_error=run_result.error,
            exec_error=exec_error,
            exec_traceback=exec_tb,
            timestamp=datetime.now().isoformat(),
        )
        result_logger.log(task_result)

        status = "PASS" if passed else "FAIL"
        print(f"{status} ({run_result.duration_seconds:.1f}s, {run_result.iterations} iter)")

    # Final report
    reporter = Reporter(result_logger)
    reporter.print_summary()
    reporter.save_summary()


if __name__ == "__main__":
    main()
