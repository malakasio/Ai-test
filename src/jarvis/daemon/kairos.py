"""
KAIROS — Always-on background daemon.

From leaked Claude Code source (confirmed real feature):
- Runs every 5 minutes
- Checks task queue for pending work
- Monitors GitHub repos for changes
- Sends push notifications for important events  
- Triggers autoDream during idle periods (>15min inactivity)
- Performs system health checks

v6 architecture:
- Runs as systemd service (NOT Docker)
- Async event loop with APScheduler
- sd_notify READY=1 via socket (no systemd Python lib needed)
- TimeoutStartSec=300 (heavy init after READY=1)
- Graceful shutdown: checkpoint → cancel tasks → close DB
- inotify watcher for config hot-reload (debounced 1s)
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import signal
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from jarvis.config import get_config, JARVIS_HOME
from jarvis.observability.logger import get_logger, setup_logging
from jarvis.observability.metrics import get_metrics

log = get_logger("kairos")

_accepting_tasks = True
_current_task_state = None
_last_activity_time = time.time()


def sd_notify_ready():
    """
    v6 fix: systemd sd_notify without external library.
    Sends READY=1 via Unix socket.
    """
    notify_socket = os.environ.get("NOTIFY_SOCKET")
    if not notify_socket:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(notify_socket)
            s.sendall(b"READY=1")
        log.debug("sd_notify: READY=1 sent")
    except Exception as e:
        log.warning(f"sd_notify failed: {e}")


def sd_notify_watchdog():
    """Send watchdog keepalive."""
    notify_socket = os.environ.get("NOTIFY_SOCKET")
    if not notify_socket:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(notify_socket)
            s.sendall(b"WATCHDOG=1")
    except Exception:
        pass


class KAIROSDaemon:
    """
    KAIROS background daemon — runs independently of terminal.
    """

    def __init__(self):
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._last_dream_time = time.time()
        self._last_activity = time.time()
        self._config_reload_task: Optional[asyncio.Task] = None

    async def start(self):
        """
        Start the KAIROS daemon.
        Notifies systemd READY=1 BEFORE heavy initialization.
        """
        log.info("KAIROS daemon starting...")

        # Start DB writer FIRST
        from jarvis.memory.database import supervised_db_writer
        db_task = asyncio.create_task(supervised_db_writer(), name="db_writer")
        self._tasks.append(db_task)

        # Notify systemd we're ready (before heavy init)
        # v6 fix: sd_notify BEFORE spaCy/Alembic init
        sd_notify_ready()

        # Now do heavy initialization (after READY=1)
        await self._heavy_init()

        self._running = True
        log.info("KAIROS daemon ready")

        # Start scheduler loops
        cfg = get_config()
        self._tasks.extend([
            asyncio.create_task(self._poll_loop(), name="kairos_poll"),
            asyncio.create_task(self._health_loop(), name="health_check"),
            asyncio.create_task(self._watchdog_loop(), name="watchdog"),
        ])

        if cfg.kairos.github_repos:
            self._tasks.append(
                asyncio.create_task(self._github_monitor_loop(), name="github_monitor")
            )

    async def _heavy_init(self):
        """Heavy initialization — runs AFTER sd_notify READY=1."""
        # Load STT/TTS models
        try:
            from jarvis.voice.stt import ensure_stt_ready
            from jarvis.voice.tts import ensure_tts_ready
            await asyncio.gather(ensure_stt_ready(), ensure_tts_ready())
        except Exception as e:
            log.warning(f"Voice models failed to load: {e}")

        # Load embedding model
        try:
            from jarvis.memory.embeddings import ensure_model_loaded
            await ensure_model_loaded()
        except Exception as e:
            log.warning(f"Embedding model failed to load: {e}")

        # Start config file watcher
        self._start_config_watcher()

    def _start_config_watcher(self):
        """Watch CLAUDE.md and SKILL.md for changes (hot-reload)."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            main_loop = asyncio.get_event_loop()

            class ConfigHandler(FileSystemEventHandler):
                def on_modified(self, event):
                    if any(event.src_path.endswith(f) for f in ["CLAUDE.md", "SKILL.md", ".env"]):
                        # v5 fix: asyncio.run_coroutine_threadsafe (not create_task from OS thread)
                        asyncio.run_coroutine_threadsafe(
                            self._debounced_reload(), main_loop
                        )

            observer = Observer()
            observer.schedule(ConfigHandler(), path=".", recursive=True)
            observer.start()
            log.info("Config file watcher started")
        except ImportError:
            log.warning("watchdog not installed — no hot-reload")
        except Exception as e:
            log.warning(f"Config watcher failed: {e}")

    _reload_task_ref: Optional[asyncio.Task] = None

    async def _debounced_reload(self):
        """v5 fix: debounce config reloads (1s delay)."""
        if self._reload_task_ref:
            self._reload_task_ref.cancel()
        self._reload_task_ref = asyncio.create_task(self._delayed_reload())

    async def _delayed_reload(self):
        await asyncio.sleep(1.0)
        log.info("Config changed — reloading")
        # Invalidate config singleton
        from jarvis import config as cfg_module
        cfg_module._config = None

    async def _poll_loop(self):
        """Main KAIROS poll loop — every 5 minutes."""
        cfg = get_config()
        while self._running:
            try:
                await self._poll_once()
            except Exception as e:
                log.error(f"KAIROS poll error: {e}", exc_info=True)

            get_metrics().kairos_runs.inc()
            await asyncio.sleep(cfg.kairos.poll_interval_s)

    async def _poll_once(self):
        """Single KAIROS poll cycle."""
        cfg = get_config()

        # 1. Check task queue
        await self._process_pending_tasks()

        # 2. Check if autoDream should run
        idle_time = time.time() - _last_activity_time
        if idle_time > cfg.kairos.dream_idle_threshold_s:
            if time.time() - self._last_dream_time > 3600:  # max once/hour
                await self._run_dream()

        # 3. Send scheduled notifications
        await self._check_scheduled_notifications()

    async def _process_pending_tasks(self):
        """Check for pending background tasks."""
        from jarvis.memory.database import db_fetch_all
        from jarvis.agents.base import BaseAgent
        from jarvis.tools.registry import get_tools_for_set

        pending = await db_fetch_all(
            "SELECT * FROM tasks WHERE status='pending' AND priority >= 4 ORDER BY created_at LIMIT 5"
        )

        for task_row in pending:
            try:
                payload = json.loads(task_row["payload"])
                task_text = payload.get("text", str(payload))

                agent = BaseAgent(agent_id="kairos_worker")
                tool_defs, handlers = get_tools_for_set([])
                for tool_def in tool_defs:
                    agent.register_tool(
                        name=tool_def["name"],
                        description=tool_def["description"],
                        input_schema=tool_def["input_schema"],
                        handler=handlers[tool_def["name"]],
                    )
                await agent.initialize()
                await agent.run_task(task_text, task_id=task_row["id"])
            except Exception as e:
                log.error(f"Background task {task_row['id'][:8]} failed: {e}")

    async def _run_dream(self):
        """Trigger autoDream memory consolidation."""
        log.info("KAIROS: triggering autoDream")
        self._last_dream_time = time.time()
        try:
            from jarvis.memory.store import run_auto_dream
            from jarvis.llm.client import simple_completion
            await run_auto_dream(simple_completion)
        except Exception as e:
            log.error(f"autoDream failed: {e}")

    async def _check_scheduled_notifications(self):
        """Send any due notifications via Telegram."""
        cfg = get_config()
        if not cfg.telegram.enabled:
            return

        from jarvis.memory.database import db_fetch_all
        due = await db_fetch_all(
            """SELECT * FROM tasks
               WHERE status='pending' AND task_type='notification'
               AND created_at <= ? ORDER BY created_at LIMIT 10""",
            (time.time(),),
        )
        for notif in due:
            try:
                payload = json.loads(notif["payload"])
                await self._send_telegram(payload.get("text", str(payload)))
                from jarvis.memory.database import db_write
                await db_write(
                    "UPDATE tasks SET status='completed', finished_at=? WHERE id=?",
                    (time.time(), notif["id"]),
                )
            except Exception as e:
                log.error(f"Notification send failed: {e}")

    async def _send_telegram(self, text: str):
        """Send Telegram notification."""
        cfg = get_config()
        if not cfg.telegram.enabled:
            return
        import aiohttp
        url = f"https://api.telegram.org/bot{cfg.telegram.bot_token}/sendMessage"
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={
                "chat_id": cfg.telegram.allowed_user_id,
                "text": text[:4096],
                "parse_mode": "Markdown",
            }, timeout=aiohttp.ClientTimeout(total=10))

    async def _github_monitor_loop(self):
        """Monitor GitHub repos for changes."""
        cfg = get_config()
        known_shas: dict[str, str] = {}

        while self._running:
            for repo in cfg.kairos.github_repos:
                try:
                    sha = await self._get_latest_commit(repo)
                    if repo in known_shas and known_shas[repo] != sha:
                        log.info(f"GitHub change detected: {repo}")
                        await self._send_telegram(f"📦 New commit in {repo}")
                    known_shas[repo] = sha
                except Exception as e:
                    log.debug(f"GitHub check failed for {repo}: {e}")

            await asyncio.sleep(300)

    async def _get_latest_commit(self, repo: str) -> str:
        """Get latest commit SHA for a GitHub repo."""
        import aiohttp
        parts = repo.strip("/").split("/")
        owner, name = parts[-2], parts[-1]
        url = f"https://api.github.com/repos/{owner}/{name}/commits?per_page=1"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                return data[0]["sha"] if data else ""

    async def _health_loop(self):
        """System health check every 60 seconds."""
        while self._running:
            try:
                await self._check_health()
            except Exception as e:
                log.error(f"Health check error: {e}")
            await asyncio.sleep(60)

    async def _check_health(self):
        """Check system health and alert on issues."""
        import shutil

        # Disk usage check
        usage = shutil.disk_usage(str(JARVIS_HOME))
        pct = usage.used / usage.total * 100
        if pct > 85:
            log.warning(f"Disk usage at {pct:.0f}%")
            await self._send_telegram(f"⚠️ Disk usage: {pct:.0f}%")

        # Memory check
        try:
            with open("/proc/meminfo") as f:
                lines = {l.split(":")[0]: l.split(":")[1].strip() for l in f if ":" in l}
            total = int(lines["MemTotal"].split()[0])
            available = int(lines["MemAvailable"].split()[0])
            pct_used = (total - available) / total * 100
            if pct_used > 90:
                log.warning(f"Memory usage at {pct_used:.0f}%")
        except Exception:
            pass

    async def _watchdog_loop(self):
        """Send systemd watchdog keepalive."""
        while self._running:
            sd_notify_watchdog()
            await asyncio.sleep(30)

    async def graceful_shutdown(self):
        """
        v6 fix: proper graceful shutdown.
        1. Stop accepting new tasks
        2. Checkpoint current state to DB
        3. Cancel tasks AND WAIT for them
        4. Close DB
        """
        global _accepting_tasks
        log.info("KAIROS graceful shutdown initiated")
        _accepting_tasks = False
        self._running = False

        # Checkpoint current state
        if _current_task_state:
            try:
                await _current_task_state.save()
            except Exception:
                pass

        # Cancel all tasks and wait
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close DB (send poison pill)
        from jarvis.memory.database import shutdown_db
        await shutdown_db()

        log.info("KAIROS shutdown complete")


async def run_daemon():
    """Entry point for the KAIROS daemon."""
    setup_logging("jarvis", debug=os.environ.get("JARVIS_DEBUG", "").lower() == "true")

    daemon = KAIROSDaemon()

    # Register signal handlers
    loop = asyncio.get_event_loop()

    async def _shutdown(sig):
        log.info(f"Received {sig.name}, shutting down...")
        await daemon.graceful_shutdown()
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig, lambda s=sig: asyncio.create_task(_shutdown(s))
        )

    await daemon.start()

    # Keep running
    try:
        while daemon._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(run_daemon())
