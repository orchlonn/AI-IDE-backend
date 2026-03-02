import json
import os

from evals.results.logger import ResultLogger


class Reporter:
    """Generate summary statistics from a completed eval run."""

    def __init__(self, logger: ResultLogger):
        self.logger = logger

    def compute_summary(self) -> dict:
        results = self.logger.results
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        pipeline_errors = sum(1 for r in results if r.pipeline_error)
        exec_errors = sum(1 for r in results if not r.passed and r.exec_error and not r.pipeline_error)

        avg_duration = sum(r.duration_seconds for r in results) / total if total else 0
        avg_iterations = sum(r.iterations for r in results) / total if total else 0

        return {
            "run_id": self.logger.run_id,
            "benchmark": results[0].benchmark if results else "",
            "mode": results[0].mode if results else "",
            "model": results[0].model if results else "",
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_at_1": round(passed / total, 4) if total else 0,
            "pipeline_errors": pipeline_errors,
            "execution_errors": exec_errors,
            "avg_duration_seconds": round(avg_duration, 2),
            "avg_iterations": round(avg_iterations, 2),
        }

    def print_summary(self):
        s = self.compute_summary()
        print(f"\n{'=' * 60}")
        print(f"EVAL RESULTS: {s['benchmark']} / {s['mode']}")
        print(f"{'=' * 60}")
        print(f"  Model:            {s['model']}")
        print(f"  Total problems:   {s['total']}")
        print(f"  Passed:           {s['passed']}")
        print(f"  Failed:           {s['failed']}")
        print(f"  pass@1:           {s['pass_at_1']:.1%}")
        print(f"  Pipeline errors:  {s['pipeline_errors']}")
        print(f"  Execution errors: {s['execution_errors']}")
        print(f"  Avg duration:     {s['avg_duration_seconds']:.1f}s")
        print(f"  Avg iterations:   {s['avg_iterations']:.1f}")
        print(f"{'=' * 60}")

    def save_summary(self):
        summary = self.compute_summary()
        path = os.path.join(self.logger.output_dir, f"{self.logger.run_id}_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {path}")

        failed_ids = [r.task_id for r in self.logger.results if not r.passed]
        if failed_ids:
            failed_path = os.path.join(self.logger.output_dir, f"{self.logger.run_id}_failed.json")
            with open(failed_path, "w") as f:
                json.dump(failed_ids, f, indent=2)
            print(f"Failed task IDs saved to {failed_path}")
