"""
Structured logging with JSON output and rotating file handler.
Uses Python's built-in RotatingFileHandler — no external logrotate for Jarvis logs.
(v6 fix: copytruncate race condition avoided by using Python's own rotation)
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from jarvis.config import LOG_DIR, AUDIT_LOG_PATH

USER_TZ = ZoneInfo(os.environ.get("USER_TIMEZONE", "Europe/Athens"))

_action_counter = 0


def _next_action_id() -> str:
    global _action_counter
    _action_counter += 1
    return f"act_{int(time.time())}_{_action_counter:04d}"


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log consumption."""

    def format(self, record: logging.LogRecord) -> str:
        now = datetime.now(USER_TZ)
        data: dict[str, Any] = {
            "ts": now.isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = traceback.format_exception(*record.exc_info)
        if hasattr(record, "extra"):
            data.update(record.extra)
        return json.dumps(data, ensure_ascii=False)


def setup_logging(name: str = "jarvis", debug: bool = False) -> logging.Logger:
    """
    Initialize structured logging.
    - Console: human-readable
    - File: JSON rotated (50MB, 7 backups)
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    # Console handler (human-readable)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    # Rotating JSON file handler
    log_file = LOG_DIR / f"{name}.log"
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50_000_000,  # 50MB
        backupCount=7,
        delay=False,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"jarvis.{name}")


class AuditLogger:
    """
    Append-only JSONL audit log for all agent actions.
    Every action logged with: id, timestamp, tool, input, output, success, duration_ms, score.
    """

    def __init__(self, path: Path = AUDIT_LOG_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)  # line-buffered

    def log_action(
        self,
        tool: str,
        input_data: Any,
        output: Any,
        success: bool,
        duration_ms: float,
        score: float | None = None,
        model_used: str | None = None,
        tokens_used: int | None = None,
        affected: list[str] | None = None,
        zone: str = "green",
    ) -> str:
        action_id = _next_action_id()
        record = {
            "action_id": action_id,
            "timestamp": datetime.now(USER_TZ).isoformat(),
            "tool": tool,
            "input": input_data if isinstance(input_data, (str, int, float, bool, type(None))) else str(input_data),
            "output": output if isinstance(output, (str, int, float, bool, type(None))) else str(output)[:500],
            "success": success,
            "duration_ms": round(duration_ms, 2),
            "score": score,
            "model_used": model_used,
            "tokens_used": tokens_used,
            "affected": affected or [],
            "zone": zone,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return action_id

    def close(self):
        self._fh.close()


_audit: AuditLogger | None = None


def get_audit() -> AuditLogger:
    global _audit
    if _audit is None:
        _audit = AuditLogger()
    return _audit
