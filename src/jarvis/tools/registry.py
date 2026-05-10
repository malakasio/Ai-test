"""
Tool registry — registers all available tools for agent use.

v5 fix: Tool sets loaded per task type (pre-defined, not dynamic).
v4 fix: max 5 tools active at once per context.
v4 fix: echo, cp, mv removed — file writes via Python API only.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from jarvis.config import get_config, WORKSPACE_DIR
from jarvis.observability.logger import get_logger, get_audit
from jarvis.security.zones import can_access, classify_path, get_directory_size
from jarvis.security.sandbox import execute_direct

log = get_logger("tools")


# ─── File System Tools ────────────────────────────────────────────────────────

async def tool_read_file(path: str) -> str:
    """Read a file. Enforces zone access and 50KB limit."""
    cfg = get_config()
    allowed, reason = can_access(path, write=False)
    if not allowed:
        raise PermissionError(reason)

    p = Path(path)
    if not p.exists():
        return f"[File not found: {path}]"
    if not p.is_file():
        return f"[Not a file: {path}]"

    size = p.stat().st_size
    if size > cfg.security.max_cat_bytes:
        return f"[File too large: {size} bytes > {cfg.security.max_cat_bytes} limit. Use head/tail.]"

    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[Read error: {e}]"


async def tool_write_file(path: str, content: str) -> str:
    """Write to a file. Enforces zone access. Creates parent dirs if needed."""
    allowed, reason = can_access(path, write=True)
    if not allowed:
        raise PermissionError(reason)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    get_audit().log_action("file_write", path, f"{len(content)} bytes", True, 0, affected=[path])
    return f"Written {len(content)} bytes to {path}"


async def tool_list_dir(path: str = ".") -> str:
    """List directory contents."""
    allowed, reason = can_access(path, write=False)
    if not allowed:
        raise PermissionError(reason)

    p = Path(path)
    if not p.exists():
        return f"[Directory not found: {path}]"

    entries = []
    for item in sorted(p.iterdir()):
        size = item.stat().st_size if item.is_file() else "-"
        entries.append(f"{'d' if item.is_dir() else 'f'} {item.name} ({size})")

    return "\n".join(entries) or "(empty)"


async def tool_bash(command: str, timeout: int = 60) -> str:
    """
    Execute a shell command (direct, not sandboxed).
    Only safe commands allowed (whitelist enforced).
    """
    cmd_parts = command.split()
    if not cmd_parts:
        return "[Empty command]"

    try:
        stdout, stderr, returncode = await execute_direct(cmd_parts, timeout=timeout)
        result = stdout
        if stderr.strip():
            result += f"\n[stderr]: {stderr.strip()[:500]}"
        if returncode != 0:
            result += f"\n[exit code: {returncode}]"
        return result or "(no output)"
    except PermissionError as e:
        return f"[BLOCKED: {e}]"
    except TimeoutError as e:
        return f"[TIMEOUT: {e}]"
    except Exception as e:
        return f"[Error: {e}]"


async def tool_web_search(query: str, max_results: int = 5) -> str:
    """Web search using DuckDuckGo (free, no API key)."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"**{r.get('title', 'No title')}**\n{r.get('href', '')}\n{r.get('body', '')[:300]}")
        return "\n\n---\n\n".join(lines)
    except Exception as e:
        return f"[Search failed: {e}]"


async def tool_memory_search(query: str, days_back: int | None = None) -> str:
    """Search agent's long-term memory."""
    from jarvis.memory.store import search_memories
    memories = await search_memories(query, top_k=5, days_back=days_back)
    if not memories:
        return "No relevant memories found."
    lines = [f"[{m['time_human']}] (score={m['fusion_score']:.2f}) {m['content'][:300]}" for m in memories]
    return "\n\n".join(lines)


async def tool_memory_save(content: str, importance: float = 0.5, tags: list[str] | None = None) -> str:
    """Save something to long-term memory."""
    from jarvis.memory.store import save_memory
    row_id = await save_memory(content, memory_type="episodic", importance=importance, tags=tags)
    return f"Saved to memory (id={row_id})"


async def tool_get_status() -> str:
    """Get system status."""
    from jarvis.observability.metrics import get_metrics
    metrics = get_metrics()
    data = metrics.to_dashboard_dict()
    return json.dumps(data, indent=2, ensure_ascii=False)


# ─── Lab Tools (JARVIS_LAB_MODE=true only) ───────────────────────────────────

async def tool_network_scan(target: str, scan_type: str = "ping") -> str:
    """
    Network scanning (lab mode only).
    scan_type: 'ping' | 'port' | 'service'
    """
    cfg = get_config()
    if not cfg.security.lab_mode:
        raise PermissionError("Network scanning requires JARVIS_LAB_MODE=true")

    if scan_type == "ping":
        cmd = ["nmap", "-sn", target, "-oG", "-"]
    elif scan_type == "port":
        cmd = ["nmap", "-p", "1-1000", "--open", target]
    elif scan_type == "service":
        cmd = ["nmap", "-sV", "--version-intensity", "3", target]
    else:
        return f"[Unknown scan type: {scan_type}]"

    stdout, stderr, rc = await execute_direct(cmd, timeout=120)
    return stdout or stderr or "[No output]"


async def tool_http_request(url: str, method: str = "GET", headers: dict | None = None, body: str | None = None) -> str:
    """HTTP request (lab mode allows non-whitelisted domains)."""
    import aiohttp
    cfg = get_config()

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            req_headers = headers or {}
            async with session.request(method.upper(), url, headers=req_headers, data=body) as resp:
                text = await resp.text()
                return f"Status: {resp.status}\n\n{text[:5000]}"
    except Exception as e:
        return f"[HTTP error: {e}]"


# ─── Tool definitions (for LLM) ───────────────────────────────────────────────

ALL_TOOLS: dict[str, dict] = {
    "read_file": {
        "description": "Read a file (max 50KB). Zone-aware.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path to read"}},
            "required": ["path"],
        },
        "handler": tool_read_file,
    },
    "write_file": {
        "description": "Write content to a file. Creates parent dirs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        "handler": tool_write_file,
    },
    "list_dir": {
        "description": "List directory contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path (default: current)"}},
        },
        "handler": tool_list_dir,
    },
    "bash": {
        "description": "Run a shell command (whitelist-enforced). Use for ls, cat, grep, git, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 60},
            },
            "required": ["command"],
        },
        "handler": tool_bash,
    },
    "web_search": {
        "description": "Search the web with DuckDuckGo (free, no API key).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        "handler": tool_web_search,
    },
    "memory_search": {
        "description": "Search agent's long-term memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "days_back": {"type": "integer", "description": "Limit to last N days (optional)"},
            },
            "required": ["query"],
        },
        "handler": tool_memory_search,
    },
    "memory_save": {
        "description": "Save important information to long-term memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "importance": {"type": "number", "description": "0.0-1.0"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["content"],
        },
        "handler": tool_memory_save,
    },
    "get_status": {
        "description": "Get current system status and metrics.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_get_status,
    },
    "network_scan": {
        "description": "Scan network targets (lab mode only). scan_type: ping|port|service",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "IP, hostname, or CIDR range"},
                "scan_type": {"type": "string", "enum": ["ping", "port", "service"]},
            },
            "required": ["target"],
        },
        "handler": tool_network_scan,
    },
    "http_request": {
        "description": "Make an HTTP request to any URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "default": "GET"},
                "headers": {"type": "object"},
                "body": {"type": "string"},
            },
            "required": ["url"],
        },
        "handler": tool_http_request,
    },
}


def get_tools_for_set(tool_set: list[str]) -> tuple[list[dict], dict]:
    """Get LLM tool definitions and handlers for a named set."""
    if not tool_set:
        # Return all non-lab tools by default
        cfg = get_config()
        tools_to_include = [
            k for k in ALL_TOOLS
            if k != "network_scan" or cfg.security.lab_mode
        ]
    else:
        tools_to_include = tool_set

    tool_defs = []
    handlers = {}
    for name in tools_to_include:
        if name in ALL_TOOLS:
            spec = ALL_TOOLS[name]
            tool_defs.append({
                "name": name,
                "description": spec["description"],
                "input_schema": spec["input_schema"],
            })
            handlers[name] = spec["handler"]

    # v4 fix: max 5 tools per context
    return tool_defs[:5], handlers
