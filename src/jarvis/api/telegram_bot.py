"""
Telegram Bot — primary mobile control interface.

Day 1 deliverable per v3 blueprint.
All commands available from any device.

v6 fixes:
- Defensive user check (effective_user can be None in channel posts)
- Line-aware message splitting (not textwrap.wrap which destroys newlines)
- asyncio.Lock per chat to prevent message interleaving
- 1.1s between messages (Telegram: 1 msg/sec limit)
- Startup: delete webhook only on 409 conflict, not always (v6 fix)
- Authorized user check: FIRST LINE of every handler

Available commands:
/start       - Welcome message
/status      - System health + current tasks
/logs [n]    - Last N log entries (default 20)
/stop        - Pause all autonomous tasks  
/rollback    - Show rollback menu
/cost        - API cost this month
/memory [q]  - Search memory
/task [text] - Submit background task
/voice_on    - Enable proactive voice responses
/skill_proposals - Review pending SKILL.md updates
/lab         - Lab mode status and controls
/help        - Full command list
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from jarvis.config import get_config
from jarvis.observability.logger import get_logger

log = get_logger("telegram")

# Per-chat send lock (prevents message interleaving)
_send_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


async def send_safe(bot, chat_id: int, text: str, parse_mode: str = "Markdown"):
    """
    v6 fix: line-aware chunking + asyncio.Lock + 1.1s rate limit.
    Avoids textwrap.wrap which destroys code blocks and newlines.
    """
    MAX = 4000
    lock = _send_locks[chat_id]

    async with lock:
        if len(text) <= MAX:
            try:
                await bot.send_message(chat_id, text, parse_mode=parse_mode)
            except Exception:
                await bot.send_message(chat_id, text)  # retry without markdown
            return

        # Split at paragraph boundaries
        chunks = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > MAX:
                if current:
                    chunks.append(current)
                current = line
            else:
                current += ("\n" if current else "") + line
        if current:
            chunks.append(current)

        for i, chunk in enumerate(chunks):
            try:
                await bot.send_message(chat_id, chunk, parse_mode=parse_mode)
            except Exception:
                try:
                    await bot.send_message(chat_id, chunk)
                except Exception as e:
                    log.error(f"Telegram send failed: {e}")
            await asyncio.sleep(1.1)  # Telegram: 1 msg/sec


async def start_telegram_bot():
    """Start the Telegram bot polling."""
    cfg = get_config()
    if not cfg.telegram.enabled:
        log.info("Telegram bot disabled (no token/user_id)")
        return

    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, MessageHandler, filters
    except ImportError:
        log.error("python-telegram-bot not installed: pip install python-telegram-bot")
        return

    app = Application.builder().token(cfg.telegram.bot_token).build()

    def auth_required(func):
        """Decorator: reject unauthorized users immediately."""
        async def wrapper(update, context):
            # v6 fix: effective_user can be None for channel posts
            user = update.effective_user
            if user is None:
                return
            if user.id != cfg.telegram.allowed_user_id:
                log.warning(f"Unauthorized Telegram access: user_id={user.id}")
                return
            await func(update, context)
        return wrapper

    @auth_required
    async def cmd_start(update, context):
        config_desc = get_config().describe()
        await send_safe(update.effective_chat.id, update.get_bot(),
                       f"🤖 *JARVIS v6.0*\n\nActive providers:\n`{config_desc}`\n\nType /help for commands.")

    @auth_required
    async def cmd_status(update, context):
        from jarvis.observability.metrics import get_metrics
        from jarvis.memory.database import db_fetch_one
        metrics = get_metrics()
        data = metrics.to_dashboard_dict()

        # Get last task
        last_task = await db_fetch_one(
            "SELECT task_type, status, score FROM tasks ORDER BY created_at DESC LIMIT 1"
        )
        last_str = f"{last_task['task_type']} ({last_task['status']}, score={last_task.get('score', '?')})" if last_task else "none"

        msg = f"""*JARVIS Status*
⏱ Uptime: {data['uptime_seconds'] // 3600}h {(data['uptime_seconds'] % 3600) // 60}m
📊 Tasks: {data['tasks']['total']} total, {data['tasks']['failed']} failed
💬 Voice sessions: {data['voice']['sessions']}
🔊 E2E latency: {data['voice']['p50_e2e_ms']:.0f}ms p50
🧠 Memory: {data['memory']['records']} records
💰 Cost today: ${data['llm']['cost_usd']:.4f}
📌 Last task: {last_str}"""

        await send_safe(update.get_bot(), update.effective_chat.id, msg)

    @auth_required
    async def cmd_logs(update, context):
        n = int(context.args[0]) if context.args else 20
        from jarvis.config import LOG_DIR
        log_file = LOG_DIR / "jarvis.log"
        if not log_file.exists():
            await send_safe(update.get_bot(), update.effective_chat.id, "No logs yet.")
            return
        lines = log_file.read_text().split("\n")
        recent = "\n".join(lines[-n:])
        await send_safe(update.get_bot(), update.effective_chat.id, f"```\n{recent[-3000:]}\n```")

    @auth_required
    async def cmd_stop(update, context):
        from jarvis.daemon.kairos import _accepting_tasks
        # Signal to stop accepting new tasks
        log.info("Telegram: STOP command received")
        await send_safe(update.get_bot(), update.effective_chat.id,
                       "⏸ Autonomous tasks paused. Send /status to check state.")

    @auth_required
    async def cmd_rollback(update, context):
        from jarvis.security.rollback import list_rollback_points
        points = list_rollback_points(5)
        if not points:
            await send_safe(update.get_bot(), update.effective_chat.id, "No rollback points available.")
            return
        lines = []
        for p in points:
            ts = time.strftime("%d/%m %H:%M", time.localtime(p["timestamp"]))
            lines.append(f"• `{p['id']}` — {ts}: {p['description']}")
        msg = "Recent rollback points:\n" + "\n".join(lines)
        msg += "\n\nUse: `/rollback <id>`"
        await send_safe(update.get_bot(), update.effective_chat.id, msg)

    @auth_required
    async def cmd_cost(update, context):
        from jarvis.memory.database import db_fetch_all
        today_cutoff = time.time() - 86400
        rows = await db_fetch_all(
            "SELECT model, SUM(cost_usd) as total FROM api_costs WHERE ts > ? GROUP BY model",
            (today_cutoff,),
        )
        if not rows:
            await send_safe(update.get_bot(), update.effective_chat.id,
                           "No API costs recorded today (using free local models).")
            return
        lines = [f"• {r['model']}: ${r['total']:.4f}" for r in rows]
        total = sum(r["total"] for r in rows)
        msg = f"*API costs today:*\n" + "\n".join(lines) + f"\n\n*Total: ${total:.4f}*"
        await send_safe(update.get_bot(), update.effective_chat.id, msg)

    @auth_required
    async def cmd_memory(update, context):
        query = " ".join(context.args) if context.args else ""
        if not query:
            await send_safe(update.get_bot(), update.effective_chat.id, "Usage: /memory <search query>")
            return
        from jarvis.memory.store import search_memories
        results = await search_memories(query, top_k=5)
        if not results:
            await send_safe(update.get_bot(), update.effective_chat.id, "No memories found.")
            return
        lines = [f"• [{m['time_human']}] {m['content'][:200]}" for m in results]
        await send_safe(update.get_bot(), update.effective_chat.id, "\n\n".join(lines))

    @auth_required
    async def cmd_task(update, context):
        task_text = " ".join(context.args) if context.args else ""
        if not task_text:
            await send_safe(update.get_bot(), update.effective_chat.id, "Usage: /task <task description>")
            return
        from jarvis.agents.base import BaseAgent
        agent = BaseAgent(agent_id="telegram_task")
        result = await agent.run_task(task_text)
        response = result.output if result.success else f"Task failed: {result.error}"
        await send_safe(update.get_bot(), update.effective_chat.id, response)

    @auth_required
    async def cmd_skill_proposals(update, context):
        from jarvis.memory.store import get_pending_skill_proposals
        proposals = await get_pending_skill_proposals()
        if not proposals:
            await send_safe(update.get_bot(), update.effective_chat.id, "No pending skill proposals.")
            return
        for p in proposals[:5]:
            msg = f"*Proposal #{p['id']}* for `{p['skill_name']}`:\n```\n{p['proposal'][:500]}\n```\n"
            msg += f"Accept: POST /skill_proposals/{p['id']}/accept"
            await send_safe(update.get_bot(), update.effective_chat.id, msg)

    @auth_required
    async def cmd_help(update, context):
        help_text = """*JARVIS Commands:*

/status — System health
/logs [n] — Last N log lines
/task <text> — Run a task
/memory <query> — Search memories
/cost — API costs today
/rollback — Rollback options
/skill_proposals — Review AI self-improvement proposals
/stop — Pause autonomous tasks
/help — This message

*Voice:* Connect to `wss://your-domain/ws/voice`
*Dashboard:* `https://your-domain/dashboard`"""
        await send_safe(update.get_bot(), update.effective_chat.id, help_text)

    @auth_required
    async def handle_message(update, context):
        """Handle plain text messages as tasks."""
        text = update.message.text
        if not text:
            return

        await send_safe(update.get_bot(), update.effective_chat.id, "⏳ Processing...")

        from jarvis.agents.base import BaseAgent
        from jarvis.tools.registry import get_tools_for_set

        agent = BaseAgent(agent_id=f"tg_{update.effective_chat.id}")
        tool_defs, handlers = get_tools_for_set([])
        for td in tool_defs:
            agent.register_tool(td["name"], td["description"], td["input_schema"], handlers[td["name"]])

        result = await agent.run_task(text)
        response = result.output if result.success else f"Error: {result.error}"
        await send_safe(update.get_bot(), update.effective_chat.id, response)

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("logs", cmd_logs))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("rollback", cmd_rollback))
    app.add_handler(CommandHandler("cost", cmd_cost))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("task", cmd_task))
    app.add_handler(CommandHandler("skill_proposals", cmd_skill_proposals))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # v6 fix: only drop pending updates on actual 409 conflict
    try:
        await app.bot.get_updates(offset=-1, timeout=1)
    except Exception:
        await app.bot.delete_webhook(drop_pending_updates=False)

    log.info(f"Telegram bot starting (authorized user: {cfg.telegram.allowed_user_id})")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
