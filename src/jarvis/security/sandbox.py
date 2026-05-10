"""
Sandboxed command execution.

v6 final architecture:
- Main agent daemon runs as systemd service (NOT in Docker)
- ALL shell commands go through Docker sandbox container
- Docker container has restricted volumes and network
- Full command timeout support (per-subcommand, not hardcoded)

v6 fixes:
- env=os.environ.copy() + override (not env={'TERM':'dumb'} alone = clears PATH)
- start_new_session=True (os.killpg only kills mpv, not daemon)
- communicate() not pipe (avoids 64KB deadlock)
- No limit= on create_subprocess_exec (invalid keyword)
- ANSI stripping in executor if output > 10KB
- Full subcommand timeout ('git clone' not just 'git')
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from jarvis.config import get_config
from jarvis.observability.logger import get_logger, get_audit
from jarvis.security.zones import validate_command
from jarvis.security.rollback import create_rollback_point

log = get_logger("security.sandbox")

ANSI_ESCAPE = re.compile(
    r"\x1b(\[([0-9;]*[mABCDEFGHJKSTflnprsu])|[()][AB012]|[=>]|\x1b)"
)

_ansi_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ansi")

# v6 fix: full subcommand matching
COMMAND_TIMEOUTS: dict[str, int] = {
    "git clone": 1800,
    "git pull": 120,
    "git push": 120,
    "git fetch": 120,
    "npm install": 600,
    "npm ci": 600,
    "pip install": 300,
    "pip3 install": 300,
    "apt install": 300,
    "apt-get install": 300,
    "docker build": 600,
    "docker pull": 300,
    "python3": 120,
    "python": 120,
    "nmap": 300,
    "tcpdump": 60,
    "default": 60,
}


def get_timeout(cmd: list[str]) -> int:
    """v6 fix: match on 'git clone' not just 'git'."""
    if len(cmd) >= 2:
        full = f"{cmd[0]} {cmd[1]}"
        if full in COMMAND_TIMEOUTS:
            return COMMAND_TIMEOUTS[full]
    return COMMAND_TIMEOUTS.get(cmd[0], COMMAND_TIMEOUTS["default"])


async def strip_ansi_async(text: str) -> str:
    """v6 fix: offload to executor if > 10KB to avoid blocking event loop."""
    if len(text) > 10_000:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _ansi_executor, lambda: ANSI_ESCAPE.sub("", text)
        )
    return ANSI_ESCAPE.sub("", text)


async def execute_sandboxed(
    cmd: list[str],
    working_dir: str | None = None,
    timeout: int | None = None,
    create_snapshot: bool = False,
    stdin_data: bytes | None = None,
) -> tuple[str, str, int]:
    """
    Execute command in Docker sandbox container.
    Returns (stdout, stderr, returncode).

    v6 architecture: all terminal commands go through docker exec.
    """
    cfg = get_config()

    # Validate command against security policy
    allowed, reason = validate_command(cmd)
    if not allowed:
        raise PermissionError(f"Command blocked: {reason}")

    # Create rollback point before potentially destructive commands
    if create_snapshot:
        await create_rollback_point(f"pre_{cmd[0]}")

    # Build docker exec command
    docker_cmd = [
        "docker", "exec",
        "-w", working_dir or "/workspace",
        cfg.security.sandbox_container,
    ] + cmd

    # v6 fix: copy env then override (not replace entirely)
    safe_env = os.environ.copy()
    safe_env["TERM"] = "dumb"
    safe_env["NO_COLOR"] = "1"
    safe_env["DEBIAN_FRONTEND"] = "noninteractive"

    effective_timeout = timeout or get_timeout(cmd)
    start_ts = time.time()

    proc = await asyncio.create_subprocess_exec(
        *docker_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=safe_env,
        start_new_session=True,  # v6 fix: separate process group
        cwd=working_dir,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(stdin_data),
            timeout=effective_timeout,
        )
    except asyncio.TimeoutError:
        log.warning(f"Command timeout after {effective_timeout}s: {' '.join(cmd)}")
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Command timed out after {effective_timeout}s: {' '.join(cmd[:3])}")

    duration = time.time() - start_ts

    # v6 fix: decode before ANSI strip
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    # Strip ANSI (in executor if large)
    stdout = await strip_ansi_async(stdout)
    stderr = await strip_ansi_async(stderr)

    # Enforce output size limit (50KB = ~12K tokens)
    max_bytes = cfg.security.max_file_size_bytes
    if len(stdout) > max_bytes:
        stdout = stdout[:max_bytes] + f"\n... [OUTPUT TRUNCATED at {max_bytes} bytes]"

    get_audit().log_action(
        tool=f"bash:{cmd[0]}",
        input_data=" ".join(cmd[:5]),
        output=stdout[:500],
        success=(proc.returncode == 0),
        duration_ms=duration * 1000,
        zone=cfg.security.zone,
    )

    return stdout, stderr, proc.returncode or 0


async def execute_direct(
    cmd: list[str],
    working_dir: str | None = None,
    timeout: int | None = None,
) -> tuple[str, str, int]:
    """
    Execute command directly (no Docker sandbox).
    Use only for trusted agent operations — Bash tool in Green zone.
    """
    allowed, reason = validate_command(cmd)
    if not allowed:
        raise PermissionError(f"Command blocked: {reason}")

    safe_env = os.environ.copy()
    safe_env["TERM"] = "dumb"
    safe_env["NO_COLOR"] = "1"
    effective_timeout = timeout or get_timeout(cmd)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=safe_env,
        cwd=working_dir,
        start_new_session=True,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=effective_timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Command timed out: {' '.join(cmd[:3])}")

    stdout = await strip_ansi_async(stdout_bytes.decode("utf-8", errors="replace"))
    stderr = await strip_ansi_async(stderr_bytes.decode("utf-8", errors="replace"))

    cfg = get_config()
    if len(stdout) > cfg.security.max_file_size_bytes:
        stdout = stdout[:cfg.security.max_file_size_bytes] + "\n...[TRUNCATED]"

    return stdout, stderr, proc.returncode or 0
