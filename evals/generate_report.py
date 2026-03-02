"""
Generate report.md from evaluation results.

Usage:
    python -m evals.generate_report
    python -m evals.generate_report --output-dir evals/output --report report.md
"""
import argparse
import json
import os
import glob
from datetime import datetime
from collections import defaultdict


def load_latest_results(output_dir: str) -> dict:
    """Load the latest summary + details for each (benchmark, mode) pair."""
    summary_files = sorted(glob.glob(os.path.join(output_dir, "*_summary.json")))

    # Group by (benchmark, mode), keep the latest
    latest = {}
    for path in summary_files:
        with open(path) as f:
            summary = json.load(f)
        key = (summary["benchmark"], summary["mode"])
        latest[key] = {"summary": summary, "summary_path": path}

    # Load corresponding detail files
    for key, entry in latest.items():
        run_id = entry["summary"]["run_id"]
        details_path = os.path.join(output_dir, f"{run_id}_details.jsonl")
        details = []
        if os.path.exists(details_path):
            with open(details_path) as f:
                for line in f:
                    if line.strip():
                        details.append(json.loads(line))
        entry["details"] = details

    return latest


def classify_error(task: dict) -> str:
    """Classify the error type for a failed task."""
    if task.get("pipeline_error"):
        return "Pipeline Error"
    err = task.get("exec_error") or ""
    if "TimeoutError" in err:
        return "Timeout"
    if "SyntaxError" in err:
        return "Syntax Error"
    if "NameError" in err:
        return "Name Error"
    if "TypeError" in err:
        return "Type Error"
    if "AssertionError" in err:
        return "Wrong Answer"
    if "IndentationError" in err:
        return "Indentation Error"
    return "Runtime Error"


def generate_report(results: dict, report_path: str):
    """Generate the markdown report."""
    lines = []

    lines.append("# Evaluation Report: Multi-Agent AI Code Assistant")
    lines.append("")
    lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    # --- Overview ---
    lines.append("## Overview")
    lines.append("")
    lines.append("This report compares a **single-LLM baseline** against a **multi-agent pipeline** ")
    lines.append("(Router → Generator → Reviewer feedback loop) on standard code generation benchmarks.")
    lines.append("")

    # --- Summary Table ---
    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Benchmark | Mode | Model | Total | Passed | Failed | pass@1 | Avg Duration | Avg Iterations |")
    lines.append("|-----------|------|-------|------:|-------:|-------:|-------:|-------------:|---------------:|")

    for key in sorted(results.keys()):
        s = results[key]["summary"]
        lines.append(
            f"| {s['benchmark']} | {s['mode']} | {s['model']} "
            f"| {s['total']} | {s['passed']} | {s['failed']} "
            f"| **{s['pass_at_1']:.1%}** | {s['avg_duration_seconds']:.1f}s "
            f"| {s['avg_iterations']:.1f} |"
        )

    lines.append("")

    # --- Per-Benchmark Comparison ---
    benchmarks = sorted(set(k[0] for k in results.keys()))

    for benchmark in benchmarks:
        lines.append(f"## {benchmark.upper()} Benchmark")
        lines.append("")

        baseline_key = (benchmark, "baseline")
        multi_key = (benchmark, "multi_agent")

        baseline = results.get(baseline_key)
        multi = results.get(multi_key)

        if baseline and multi:
            bs = baseline["summary"]
            ms = multi["summary"]
            delta = ms["pass_at_1"] - bs["pass_at_1"]
            sign = "+" if delta >= 0 else ""
            lines.append(f"**Baseline pass@1:** {bs['pass_at_1']:.1%}  ")
            lines.append(f"**Multi-Agent pass@1:** {ms['pass_at_1']:.1%}  ")
            lines.append(f"**Delta:** {sign}{delta:.1%}")
            lines.append("")

        # Error breakdown for each mode
        for mode_key in [baseline_key, multi_key]:
            entry = results.get(mode_key)
            if not entry:
                continue
            details = entry["details"]
            failed = [t for t in details if not t["passed"]]
            if not failed:
                lines.append(f"### {entry['summary']['mode']} — All problems passed!")
                lines.append("")
                continue

            error_counts = defaultdict(int)
            for t in failed:
                error_counts[classify_error(t)] += 1

            lines.append(f"### {entry['summary']['mode']} — Error Breakdown ({len(failed)} failures)")
            lines.append("")
            lines.append("| Error Type | Count | % of Failures |")
            lines.append("|------------|------:|--------------:|")
            for err_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                pct = count / len(failed) * 100
                lines.append(f"| {err_type} | {count} | {pct:.1f}% |")
            lines.append("")

        # Differential analysis
        if baseline and multi:
            baseline_details = {t["task_id"]: t for t in baseline["details"]}
            multi_details = {t["task_id"]: t for t in multi["details"]}

            common_ids = set(baseline_details.keys()) & set(multi_details.keys())

            multi_wins = [tid for tid in sorted(common_ids)
                          if not baseline_details[tid]["passed"] and multi_details[tid]["passed"]]
            baseline_wins = [tid for tid in sorted(common_ids)
                             if baseline_details[tid]["passed"] and not multi_details[tid]["passed"]]

            if multi_wins or baseline_wins:
                lines.append("### Differential Analysis")
                lines.append("")

            if multi_wins:
                lines.append(f"**Multi-agent solved but baseline failed ({len(multi_wins)}):**")
                lines.append("")
                for tid in multi_wins[:20]:  # Cap at 20
                    iters = multi_details[tid]["iterations"]
                    lines.append(f"- `{tid}` ({iters} iteration{'s' if iters != 1 else ''})")
                if len(multi_wins) > 20:
                    lines.append(f"- ... and {len(multi_wins) - 20} more")
                lines.append("")

            if baseline_wins:
                lines.append(f"**Baseline solved but multi-agent failed ({len(baseline_wins)}):**")
                lines.append("")
                for tid in baseline_wins[:20]:
                    lines.append(f"- `{tid}`")
                if len(baseline_wins) > 20:
                    lines.append(f"- ... and {len(baseline_wins) - 20} more")
                lines.append("")

    # --- Iteration Analysis (multi-agent only) ---
    multi_entries = {k: v for k, v in results.items() if k[1] == "multi_agent"}
    if multi_entries:
        lines.append("## Multi-Agent Iteration Analysis")
        lines.append("")
        lines.append("How many revision cycles did the reviewer request?")
        lines.append("")
        lines.append("| Benchmark | 1 Iteration (approved first try) | 2 Iterations | 3 Iterations (max) |")
        lines.append("|-----------|--:|--:|--:|")

        for key in sorted(multi_entries.keys()):
            details = multi_entries[key]["details"]
            iter_counts = defaultdict(int)
            for t in details:
                i = min(t["iterations"], 3)
                iter_counts[i] += 1
            total = len(details)
            row = f"| {key[0]}"
            for i in [1, 2, 3]:
                c = iter_counts.get(i, 0)
                pct = c / total * 100 if total else 0
                row += f" | {c} ({pct:.0f}%)"
            row += " |"
            lines.append(row)
        lines.append("")

    # --- Methodology ---
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Baseline:** Single OpenAI API call with temperature=0. No retrieval, no reviewer.")
    lines.append("- **Multi-Agent:** Full LangGraph pipeline — Router classifies complexity, Generator ")
    lines.append("  produces code, Reviewer evaluates and may request up to 3 revision iterations.")
    lines.append("- **Execution:** Each generated solution is run in a sandboxed subprocess with a 10-second timeout.")
    lines.append("- **Metric:** pass@1 — fraction of problems where the generated code passes all unit tests on the first attempt.")
    lines.append("- **HumanEval:** 164 hand-crafted Python problems (OpenAI).")
    lines.append("- **MBPP:** Mostly Basic Python Problems, sanitized test split (~430 problems, Google Research).")
    lines.append("")

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--output-dir", default="evals/output")
    parser.add_argument("--report", default="report.md")
    args = parser.parse_args()

    results = load_latest_results(args.output_dir)
    if not results:
        print(f"No results found in {args.output_dir}. Run evaluations first.")
        return

    print(f"Found results for: {', '.join(f'{b}/{m}' for b, m in sorted(results.keys()))}")
    generate_report(results, args.report)


if __name__ == "__main__":
    main()
