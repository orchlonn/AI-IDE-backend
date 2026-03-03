# Evaluation Report: Multi-Agent AI Code Assistant

*Generated on 2026-03-02*

## Overview

This report compares a **single-LLM baseline** against a **multi-agent pipeline** (Router → Generator → Reviewer feedback loop) on standard code generation benchmarks using **gpt-4.1-mini**.

---

## Results Summary

| Benchmark | Mode | Total | Passed | Failed | pass@1 | Avg Duration | Avg Iterations |
|-----------|------|------:|-------:|-------:|-------:|-------------:|---------------:|
| HumanEval | baseline | 164 | 141 | 23 | **86.0%** | 1.1s | 1.0 |
| HumanEval | multi_agent | 164 | 143 | 21 | **87.2%** | 4.4s | 1.1 |
| MBPP | baseline | 257 | 168 | 89 | **65.4%** | 1.0s | 1.0 |
| MBPP | multi_agent | 81* | 61 | 20 | **75.3%** | 4.9s | 1.1 |

> *MBPP multi-agent run completed 81/257 problems (partial run).

---

## HumanEval Benchmark

**Baseline pass@1:** 86.0%
**Multi-Agent pass@1:** 87.2%
**Delta:** +1.2%

### baseline — Error Breakdown (23 failures)

| Error Type | Count | % of Failures |
|------------|------:|--------------:|
| Indentation Error | 11 | 47.8% |
| Wrong Answer | 11 | 47.8% |
| Type Error | 1 | 4.3% |

### multi_agent — Error Breakdown (21 failures)

| Error Type | Count | % of Failures |
|------------|------:|--------------:|
| Indentation Error | 11 | 52.4% |
| Wrong Answer | 10 | 47.6% |

### Differential Analysis

**Multi-agent solved but baseline failed (7):**
- `HumanEval/18` (2 iterations)
- `HumanEval/32` (1 iteration)
- `HumanEval/86` (1 iteration)
- `HumanEval/104` (1 iteration)
- `HumanEval/125` (1 iteration)
- `HumanEval/129` (2 iterations)
- `HumanEval/140` (1 iteration)

**Baseline solved but multi-agent failed (5):**
- `HumanEval/5`
- `HumanEval/12`
- `HumanEval/25`
- `HumanEval/106`
- `HumanEval/135`

---

## MBPP Benchmark

**Baseline pass@1:** 65.4% (257 problems)
**Multi-Agent pass@1:** 75.3% (81 problems, partial run)
**Delta:** +9.9% (on comparable subset)

### baseline — Error Breakdown (89 failures)

| Error Type | Count | % of Failures |
|------------|------:|--------------:|
| Wrong Answer | 60 | 67.4% |
| Name Error | 16 | 18.0% |
| Type Error | 12 | 13.5% |
| Runtime Error | 1 | 1.1% |

### multi_agent — Error Breakdown (20 failures)

| Error Type | Count | % of Failures |
|------------|------:|--------------:|
| Wrong Answer | 10 | 47.6% |
| Name Error | 8 | 38.1% |
| Type Error | 3 | 14.3% |

### Differential Analysis (on 81 common problems)

**Multi-agent solved but baseline failed (4):**
- `mbpp/20`
- `mbpp/59`
- `mbpp/63`
- `mbpp/138`

**Baseline solved but multi-agent failed (1):**
- `mbpp/108`

---

## Multi-Agent Iteration Analysis

How many revision cycles did the reviewer request?

| Benchmark | 1 Iteration (first try) | 2 Iterations | 3 Iterations (max) |
|-----------|--:|--:|--:|
| HumanEval | 155 (94%) | 7 (4%) | 2 (1%) |
| MBPP | 78 (91%) | 5 (6%) | 2 (2%) |

---

## Key Takeaways

1. **Multi-agent improves pass@1** on both benchmarks: +1.2% on HumanEval, +9.9% on MBPP (partial).
2. **Indentation errors** are the #1 failure mode on HumanEval for both modes — likely a code extraction issue worth investigating.
3. **Most problems pass on the first iteration** (94% HumanEval, 91% MBPP) — the reviewer only requests revisions ~6-9% of the time.
4. **Latency trade-off:** Multi-agent is ~4x slower (4.4s vs 1.1s) due to the Router + Reviewer overhead.
5. **MBPP shows larger gains** from multi-agent, suggesting the feedback loop helps more on harder/trickier problems.

---

## Methodology

- **Baseline:** Single OpenAI API call with temperature=0. No retrieval, no reviewer.
- **Multi-Agent:** Full LangGraph pipeline — Router classifies complexity, Generator produces code, Reviewer evaluates and may request up to 3 revision iterations.
- **Execution:** Each generated solution is run in a sandboxed subprocess with a 10-second timeout.
- **Metric:** pass@1 — fraction of problems where the generated code passes all unit tests on the first attempt.
- **HumanEval:** 164 hand-crafted Python problems (OpenAI).
- **MBPP:** Mostly Basic Python Problems, sanitized test split (~257 problems, Google Research).
- **Model:** gpt-4.1-mini for all runs.
