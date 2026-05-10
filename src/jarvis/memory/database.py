"""
SQLite database layer with all v6 fixes applied:
- Single writer via asyncio.Queue + Future (bidirectional)
- WAL mode + required PRAGMAs
- Supervised writer task with drain-on-crash
- Hourly WAL checkpoint (PASSIVE only, never TRUNCATE with Litestream)
- Absolute path, never relative
- Alembic migrations
- inotify-safe (Python RotatingFileHandler used, not system logrotate)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from jarvis.config import DB_PATH, get_config
from jarvis.observability.logger import get_logger

log = get_logger("db")


@dataclass
class DBRequest:
    sql: str
    params: tuple
    future: asyncio.Future
    fetch_mode: str = "none"  # "none" | "one" | "all"


# Global single-writer queue
db_queue: asyncio.Queue[DBRequest | None] = asyncio.Queue()
_db_ready = asyncio.Event()


INIT_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=10000;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-65536;
PRAGMA foreign_keys=ON;
PRAGMA temp_store=MEMORY;

CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    content     TEXT    NOT NULL,
    embedding   BLOB,
    importance  REAL    DEFAULT 0.5,
    memory_type TEXT    NOT NULL CHECK(memory_type IN ('episodic','semantic')),
    tags        TEXT    DEFAULT '[]',
    session_id  TEXT,
    source      TEXT,
    confidence  REAL    DEFAULT 0.8,
    consolidated INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    role        TEXT    NOT NULL CHECK(role IN ('user','assistant','system','tool')),
    content     TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    tool_name   TEXT,
    compressed  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT    PRIMARY KEY,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    priority    INTEGER DEFAULT 3,
    task_type   TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    status      TEXT    DEFAULT 'pending' CHECK(status IN ('pending','running','completed','failed','cancelled')),
    result      TEXT,
    error       TEXT,
    score       REAL,
    started_at  REAL,
    finished_at REAL,
    agent_id    TEXT
);

CREATE TABLE IF NOT EXISTS api_costs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    model       TEXT    NOT NULL,
    input_tok   INTEGER DEFAULT 0,
    output_tok  INTEGER DEFAULT 0,
    cost_usd    REAL    DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS checkpoints (
    task_id     TEXT    PRIMARY KEY,
    ts          REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    state_json  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_proposals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    skill_name  TEXT    NOT NULL,
    proposal    TEXT    NOT NULL,
    status      TEXT    DEFAULT 'pending' CHECK(status IN ('pending','accepted','rejected'))
);

CREATE TABLE IF NOT EXISTS action_log (
    action_id   TEXT    PRIMARY KEY,
    ts          REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    tool        TEXT    NOT NULL,
    input       TEXT,
    output      TEXT,
    success     INTEGER DEFAULT 1,
    duration_ms REAL,
    score       REAL,
    model_used  TEXT,
    tokens_used INTEGER,
    affected    TEXT    DEFAULT '[]',
    zone        TEXT    DEFAULT 'green'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(content, tokenize='unicode61');

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_consolidated ON memories(consolidated);
CREATE INDEX IF NOT EXISTS idx_sessions_session ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_compressed ON sessions(compressed);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, priority);
CREATE INDEX IF NOT EXISTS idx_api_costs_ts ON api_costs(ts);
"""


async def db_writer_task():
    """
    Single writer coroutine — owns the DB connection exclusively.
    All writes from any coroutine go through the shared db_queue.
    On crash: drain queue with errors before restarting.
    """
    db_path = Path(get_config().memory.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        await db.executescript(INIT_SQL)
        await db.commit()
        _db_ready.set()
        log.info(f"Database ready at {db_path}")

        last_checkpoint = time.time()

        while True:
            try:
                req = await asyncio.wait_for(db_queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                # Hourly WAL checkpoint — PASSIVE (safe with Litestream)
                if time.time() - last_checkpoint > 3600:
                    try:
                        await db.execute("PRAGMA wal_checkpoint(PASSIVE)")
                        last_checkpoint = time.time()
                        log.debug("WAL checkpoint completed")
                    except Exception as e:
                        log.warning(f"WAL checkpoint failed: {e}")
                continue

            if req is None:
                # Poison pill — graceful shutdown
                log.info("DB writer shutting down")
                break

            try:
                if req.fetch_mode == "all":
                    cursor = await db.execute(req.sql, req.params)
                    rows = await cursor.fetchall()
                    req.future.set_result([dict(r) for r in rows])
                elif req.fetch_mode == "one":
                    cursor = await db.execute(req.sql, req.params)
                    row = await cursor.fetchone()
                    req.future.set_result(dict(row) if row else None)
                else:
                    cursor = await db.execute(req.sql, req.params)
                    await db.commit()
                    req.future.set_result(cursor.lastrowid)
            except Exception as e:
                req.future.set_exception(e)
            finally:
                db_queue.task_done()


async def supervised_db_writer():
    """Restarts db_writer_task on crash, draining queue to unblock waiters."""
    from jarvis.observability.logger import get_logger
    log = get_logger("db.supervisor")

    while True:
        try:
            await db_writer_task()
            return  # clean shutdown via poison pill
        except Exception as e:
            log.critical(f"DB writer crashed: {e}", exc_info=True)
            _db_ready.clear()
            # Drain pending requests with error to unblock callers
            while not db_queue.empty():
                try:
                    req = db_queue.get_nowait()
                    if req and not req.future.done():
                        req.future.set_exception(RuntimeError("DB writer restarted"))
                except asyncio.QueueEmpty:
                    break
            await asyncio.sleep(2)
            _db_ready.set()  # allow reconnection


async def db_write(sql: str, params: tuple = ()) -> int:
    """Execute a write. Returns lastrowid."""
    await _db_ready.wait()
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    await db_queue.put(DBRequest(sql, params, future, "none"))
    return await future


async def db_fetch_one(sql: str, params: tuple = ()) -> dict[str, Any] | None:
    """Fetch a single row."""
    await _db_ready.wait()
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    await db_queue.put(DBRequest(sql, params, future, "one"))
    return await future


async def db_fetch_all(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Fetch all rows."""
    await _db_ready.wait()
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    await db_queue.put(DBRequest(sql, params, future, "all"))
    return await future


async def shutdown_db():
    """Send poison pill to writer task."""
    await db_queue.put(None)
