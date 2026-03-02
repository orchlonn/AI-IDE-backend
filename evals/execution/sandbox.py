import multiprocessing
import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecResult:
    passed: bool
    error: Optional[str] = None
    traceback: Optional[str] = None


def _run_code(code: str, result_queue: multiprocessing.Queue):
    """Execute code in a child process."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        result_queue.put(ExecResult(passed=True))
    except AssertionError as e:
        result_queue.put(ExecResult(
            passed=False,
            error=f"AssertionError: {e}",
            traceback=traceback.format_exc(),
        ))
    except Exception as e:
        result_queue.put(ExecResult(
            passed=False,
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),
        ))


def execute_with_timeout(code: str, timeout: int = 10) -> ExecResult:
    """Execute Python code in a separate process with a timeout.

    Uses multiprocessing so infinite loops or crashes can be killed.
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_run_code, args=(code, result_queue))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.kill()
        process.join()
        return ExecResult(
            passed=False,
            error="TimeoutError: execution exceeded time limit",
        )

    if result_queue.empty():
        return ExecResult(
            passed=False,
            error="Process exited without producing a result (possible crash)",
        )

    return result_queue.get()
