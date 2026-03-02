import re


def _extract_from_response(raw_response: str, preserve_indent: bool = False) -> str:
    """Extract Python code from an LLM response.

    Handles:
    - Raw code (no markdown)
    - ```python ... ``` blocks
    - ``` ... ``` blocks
    - // file: or # file: headers on the first line

    If preserve_indent is True, leading whitespace is kept (needed for
    HumanEval function bodies that must stay indented).
    """
    code = raw_response

    # Extract from markdown code blocks if present
    for pattern in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        match = re.search(pattern, code, re.DOTALL)
        if match:
            code = match.group(1)
            break

    # Strip // file: or # file: header
    lines = code.split("\n")
    if lines and (lines[0].strip().startswith("// file:") or lines[0].strip().startswith("# file:")):
        lines = lines[1:]

    # Remove leading/trailing blank lines but preserve indentation within lines
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    code = "\n".join(lines)

    if preserve_indent:
        return code
    return code.strip()


def _ensure_indented(code: str, indent: str = "    ") -> str:
    """Ensure code lines are indented (for HumanEval function bodies)."""
    lines = code.split("\n")
    # Check if the first non-empty line is already indented
    for line in lines:
        if line.strip():
            if line.startswith((" ", "\t")):
                return code  # Already indented
            break
    # Re-indent all lines
    return "\n".join(indent + line if line.strip() else line for line in lines)


def extract_function_body(raw_response: str) -> str:
    """For HumanEval: extract just the function body from the response.

    Returns indented code suitable for concatenation after a function signature.
    If the model returned the full function (with def line), strip the
    signature and docstring to get just the body.
    """
    code = _extract_from_response(raw_response, preserve_indent=True)
    lines = code.split("\n")

    # If the response starts with 'def ', the model included the signature
    if lines and lines[0].strip().startswith("def "):
        in_docstring = False
        body_start = 1
        for i, line in enumerate(lines[1:], start=1):
            stripped = line.strip()
            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    body_start = i + 1
                    continue
                in_docstring = True
                continue
            if in_docstring and ('"""' in stripped or "'''" in stripped):
                in_docstring = False
                body_start = i + 1
                continue
            if not in_docstring and stripped:
                body_start = i
                break
        code = "\n".join(lines[body_start:])

    return _ensure_indented(code)


def extract_complete_function(raw_response: str) -> str:
    """For MBPP: extract the complete function definition from the response."""
    return _extract_from_response(raw_response, preserve_indent=False)
