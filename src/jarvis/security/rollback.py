"""
Rollback system — three-tier approach (v4 fix):

Tier 1: Files → git stash/restore
Tier 2: Database → SQLite .backup API + Litestream (NOT cp on live DB)
Tier 3: External actions → dry-run mode + confirmation

v5 fix: rollback CLI requires daemon stop first (not while running).
Emergency recovery: PyInstaller-compiled CLI + SSH key always available.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from jarvis.config import get_config, JARVIS_HOME, DB_PATH
from jarvis.observability.logger import get_logger

log = get_logger("security.rollback")

ROLLBACK_LOG = JARVIS_HOME / "data" / "rollback_log.jsonl"


@dataclass
class RollbackPoint:
    id: str
    timestamp: float
    description: str
    git_stash: str | None = None
    db_backup: str | None = None
    affected_files: list[str] = None

    def __post_init__(self):
        if self.affected_files is None:
            self.affected_files = []


async def create_rollback_point(description: str) -> RollbackPoint:
    """
    Create a rollback point before a potentially destructive action.
    Stores git stash hash and optionally DB backup path.
    """
    rp_id = f"rp_{int(time.time())}"
    workspace = str(JARVIS_HOME)

    # Git stash
    git_stash_hash = None
    try:
        result = subprocess.run(
            ["git", "-C", workspace, "stash", "push", "-m", f"pre-action-{rp_id}"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and "HEAD" in result.stdout:
            # Get the stash hash
            stash_result = subprocess.run(
                ["git", "-C", workspace, "stash", "list", "--max-count=1", "--format=%H"],
                capture_output=True, text=True, timeout=10,
            )
            git_stash_hash = stash_result.stdout.strip()
    except Exception as e:
        log.warning(f"Git stash failed: {e}")

    # DB backup (using SQLite .backup API — safe)
    db_backup_path = None
    if DB_PATH.exists():
        backup_path = str(JARVIS_HOME / "backups" / f"jarvis_{rp_id}.db")
        try:
            result = subprocess.run(
                ["sqlite3", str(DB_PATH), f".backup {backup_path}"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                db_backup_path = backup_path
        except Exception as e:
            log.warning(f"DB backup failed: {e}")

    rp = RollbackPoint(
        id=rp_id,
        timestamp=time.time(),
        description=description,
        git_stash=git_stash_hash,
        db_backup=db_backup_path,
    )

    # Log rollback point
    ROLLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ROLLBACK_LOG, "a") as f:
        f.write(json.dumps(asdict(rp)) + "\n")

    log.info(f"Rollback point created: {rp_id} ({description})")
    return rp


async def rollback_to_point(rp_id: str):
    """
    Restore state to a specific rollback point.
    NOTE: Must stop daemon first before DB restore (v5 fix).
    """
    workspace = str(JARVIS_HOME)

    # Find rollback point
    rp = None
    if ROLLBACK_LOG.exists():
        with open(ROLLBACK_LOG) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("id") == rp_id:
                        rp = RollbackPoint(**data)
                        break
                except Exception:
                    pass

    if not rp:
        raise ValueError(f"Rollback point not found: {rp_id}")

    log.warning(f"Rolling back to {rp_id}: {rp.description}")

    # Restore git stash
    if rp.git_stash:
        try:
            subprocess.run(
                ["git", "-C", workspace, "stash", "pop"],
                capture_output=True, text=True, timeout=30,
            )
            log.info("Git stash restored")
        except Exception as e:
            log.error(f"Git restore failed: {e}")

    # Restore DB (ONLY safe outside daemon)
    if rp.db_backup and Path(rp.db_backup).exists():
        log.warning("DB restore requires stopping the daemon first!")
        log.warning(f"Run: systemctl stop jarvis && cp {rp.db_backup} {DB_PATH}")


async def rollback_last():
    """Roll back the most recent rollback point."""
    if not ROLLBACK_LOG.exists():
        raise FileNotFoundError("No rollback points found")

    lines = ROLLBACK_LOG.read_text().strip().split("\n")
    if not lines:
        raise ValueError("No rollback points found")

    last = json.loads(lines[-1])
    await rollback_to_point(last["id"])


def list_rollback_points(n: int = 10) -> list[dict]:
    """List recent rollback points."""
    if not ROLLBACK_LOG.exists():
        return []
    lines = ROLLBACK_LOG.read_text().strip().split("\n")
    points = []
    for line in lines[-n:]:
        try:
            points.append(json.loads(line))
        except Exception:
            pass
    return list(reversed(points))
