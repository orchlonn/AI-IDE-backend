import asyncio
import os
import time
import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class TerminalRequest(BaseModel):
    command: str
    cwd: str | None = None
    file: dict | None = None


@router.post("/api/terminal")
async def run_terminal(req: TerminalRequest):
    """Execute a command and stream stdout/stderr."""
    if not req.command:
        raise HTTPException(status_code=400, detail="command is required")

    logger.info("Terminal request  command=%.120s  cwd=%s", req.command, req.cwd)

    actual_command = req.command
    temp_dir = None

    # Write file to temp if provided
    if req.file and req.file.get("name") and req.file.get("content") is not None:
        temp_dir = tempfile.mkdtemp(prefix="ai-ide-run-")
        file_path = os.path.join(temp_dir, req.file["name"])
        Path(file_path).write_text(req.file["content"], encoding="utf-8")
        actual_command = req.command.replace(f'"{req.file["name"]}"', f'"{file_path}"')
        logger.debug("Temp file created: %s", file_path)

    working_dir = temp_dir or req.cwd or os.environ.get("HOME", "/")

    async def stream_output():
        start = time.time()
        process = await asyncio.create_subprocess_shell(
            actual_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
            env={**os.environ, "TERM": "dumb", "FORCE_COLOR": "0"},
        )

        async def read_stream(stream):
            while True:
                line = await stream.read(1024)
                if not line:
                    break
                yield line.decode()

        # Read stdout and stderr concurrently
        tasks = []
        if process.stdout:
            async for chunk in read_stream(process.stdout):
                yield chunk
        if process.stderr:
            async for chunk in read_stream(process.stderr):
                yield chunk

        exit_code = await process.wait()
        duration_ms = (time.time() - start) * 1000

        logger.info("Terminal done  exit_code=%d  duration=%.0fms  command=%.80s",
                     exit_code, duration_ms, req.command)

        # Clean up temp file
        if temp_dir and req.file and req.file.get("name"):
            try:
                os.unlink(os.path.join(temp_dir, req.file["name"]))
                logger.debug("Temp file cleaned up: %s", temp_dir)
            except OSError:
                logger.warning("Failed to clean up temp file: %s", temp_dir)

        yield f"\n__EXIT_CODE__:{exit_code}\n"

    return StreamingResponse(
        stream_output(),
        media_type="text/plain; charset=utf-8",
        headers={"Transfer-Encoding": "chunked"},
    )
