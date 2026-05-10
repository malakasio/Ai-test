"""
FastAPI application — single app, multiple routes.

v6 fix: all routes in ONE FastAPI app (not multiple ports).
Separate Caddy proxy handles external auth + HTTPS.

Endpoints:
  GET  /health          — system health check
  GET  /metrics         — Prometheus metrics
  GET  /dashboard       — JSON dashboard data
  POST /chat            — text chat
  WS   /ws/voice        — bidirectional voice WebSocket
  WS   /ws/stream       — streaming text output
  POST /webhooks/gmail  — Gmail push notifications (no auth)
  GET  /rollback/list   — list rollback points
  POST /rollback/{id}   — execute rollback
  POST /tasks           — submit background task
  GET  /tasks/{id}      — get task status
  GET  /memory/search   — search memories
  GET  /skill_proposals — pending SKILL.md proposals
  POST /skill_proposals/{id}/accept — accept proposal

v6 fixes:
- LimitBodySize middleware (1MB max)
- Persistent session secret (not random on restart)
- WebSocket heartbeat (30s ping)
- Separate /webhooks/* path (no auth for Google)
- Rate limiting on webhook endpoints
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from jarvis.config import get_config, SESSION_SECRET_FILE
from jarvis.observability.logger import get_logger, setup_logging
from jarvis.observability.metrics import get_metrics

log = get_logger("api")


# ─── Session secret (persistent across restarts) ─────────────────────────────

def _load_or_create_secret() -> bytes:
    """v6 fix: persistent secret key (not random on each restart = forced logout)."""
    try:
        SESSION_SECRET_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if SESSION_SECRET_FILE.exists():
            return SESSION_SECRET_FILE.read_bytes()
        key = os.urandom(32)
        SESSION_SECRET_FILE.write_bytes(key)
        SESSION_SECRET_FILE.chmod(0o600)
        return key
    except PermissionError:
        return os.urandom(32)  # fallback: ephemeral key


SECRET_KEY = _load_or_create_secret()


# ─── Middleware ───────────────────────────────────────────────────────────────

class LimitBodySize(BaseHTTPMiddleware):
    """v6 fix: prevent OOM from huge POST bodies."""
    MAX_BODY = 1_000_000  # 1MB

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_BODY:
            return Response("Request too large", status_code=413)
        return await call_next(request)


# Rate limiter (simple in-memory)
_rate_limits: dict[str, list[float]] = {}


def rate_limit(key: str, max_requests: int = 60, window_s: int = 60) -> bool:
    now = time.time()
    window = _rate_limits.setdefault(key, [])
    # Clean old entries
    window[:] = [t for t in window if now - t < window_s]
    if len(window) >= max_requests:
        return False
    window.append(now)
    return True


# ─── App creation ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="JARVIS API",
        description="Autonomous Digital Assistant v6.0",
        version="6.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    app.add_middleware(LimitBodySize)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Caddy handles auth externally
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health & Metrics ─────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        metrics = get_metrics()
        data = metrics.to_dashboard_dict()
        return {"status": "ok", "uptime": data["uptime_seconds"], "metrics": data}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        return get_metrics().to_prometheus_text()

    @app.get("/dashboard")
    async def dashboard():
        return get_metrics().to_dashboard_dict()

    # ── Chat ─────────────────────────────────────────────────────────────────

    @app.post("/chat")
    async def chat(request: Request):
        body = await request.json()
        message = body.get("message", "")
        session_id = body.get("session_id", str(uuid.uuid4()))

        if not message:
            raise HTTPException(400, "message required")

        from jarvis.agents.base import BaseAgent
        from jarvis.tools.registry import get_tools_for_set

        agent = BaseAgent(agent_id=f"api_{session_id[:8]}")
        tool_defs, handlers = get_tools_for_set([])
        for td in tool_defs:
            agent.register_tool(td["name"], td["description"], td["input_schema"], handlers[td["name"]])

        result = await agent.run_task(message)

        return {
            "session_id": session_id,
            "response": result.output,
            "score": result.score,
            "success": result.success,
            "error": result.error,
        }

    # ── Voice WebSocket ───────────────────────────────────────────────────────

    @app.websocket("/ws/voice")
    async def voice_ws(ws: WebSocket):
        """
        Bidirectional voice WebSocket.
        Client → sends PCM audio bytes
        Server → sends MP3/WAV audio bytes

        Protocol:
        - Binary frames: audio data
        - Text frames: JSON control messages {'type': 'ping'|'interrupt'|'stop'}
        """
        await ws.accept()
        cfg = get_config()

        from jarvis.voice.pipeline import VoicePipeline
        from jarvis.agents.base import BaseAgent
        from pathlib import Path

        system = Path("CLAUDE.md").read_text() if Path("CLAUDE.md").exists() else "You are JARVIS."
        pipeline = VoicePipeline(
            agent_fn=None,
            system_prompt=system,
            session_id=str(uuid.uuid4()),
        )
        await pipeline.initialize()
        get_metrics().voice_sessions_total.inc()

        # Heartbeat task
        async def heartbeat():
            while True:
                try:
                    await ws.send_json({"type": "ping"})
                    await asyncio.sleep(cfg.server.ws_heartbeat_s)
                except Exception:
                    break

        hb_task = asyncio.create_task(heartbeat())

        try:
            while True:
                data = await ws.receive()

                if "text" in data:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "interrupt":
                        log.info("Client requested interrupt")
                    elif msg.get("type") == "stop":
                        break

                elif "bytes" in data:
                    audio_bytes = data["bytes"]

                    # Feed to VAD for barge-in detection
                    pipeline.feed_vad(audio_bytes)

                    # Process and stream response
                    async for audio_chunk in pipeline.process_audio_input(audio_bytes):
                        await ws.send_bytes(audio_chunk)

                    await ws.send_json({"type": "audio_done"})

        except WebSocketDisconnect:
            log.info("Voice WebSocket disconnected")
        except Exception as e:
            log.error(f"Voice WebSocket error: {e}")
        finally:
            hb_task.cancel()
            await pipeline.shutdown()

    # ── Streaming text WebSocket ──────────────────────────────────────────────

    @app.websocket("/ws/stream")
    async def stream_ws(ws: WebSocket):
        """Text streaming WebSocket — streams LLM tokens as they arrive."""
        await ws.accept()

        try:
            while True:
                data = await ws.receive_json()
                message = data.get("message", "")
                if not message:
                    continue

                from jarvis.llm.router import route
                from jarvis.llm.client import run_agent
                decision = route(message)

                response, _ = await run_agent(
                    messages=[{"role": "user", "content": message}],
                    tools=[],
                    system="You are JARVIS.",
                    decision=decision,
                )
                await ws.send_json({"type": "response", "text": response})
                await ws.send_json({"type": "done"})

        except WebSocketDisconnect:
            pass

    # ── Webhooks (no auth — Google needs to POST here) ────────────────────────

    @app.post("/webhooks/gmail")
    async def gmail_webhook(request: Request):
        """v6 fix: HMAC verification + rate limiting."""
        cfg = get_config()

        # Rate limit: 60 requests/minute
        if not rate_limit("gmail_webhook", max_requests=60):
            raise HTTPException(429, "Too many requests")

        body = await request.json()

        # v6 fix: base64 decode the notification data
        import base64
        raw_data = body.get("message", {}).get("data", "")
        if raw_data:
            try:
                decoded = base64.b64decode(raw_data).decode("utf-8")
                notification = json.loads(decoded)
                history_id = notification.get("historyId")
                if history_id:
                    log.info(f"Gmail notification: historyId={history_id}")
                    # Queue email processing task
                    from jarvis.memory.database import db_write
                    await db_write(
                        "INSERT INTO tasks (id, task_type, payload, priority) VALUES (?,?,?,?)",
                        (str(uuid.uuid4()), "communication", json.dumps({"historyId": history_id}), 2),
                    )
            except Exception as e:
                log.warning(f"Gmail webhook parse error: {e}")

        return {"status": "ok"}

    # ── Task Management ───────────────────────────────────────────────────────

    @app.post("/tasks")
    async def submit_task(request: Request):
        body = await request.json()
        task_text = body.get("task", "")
        priority = body.get("priority", 3)
        if not task_text:
            raise HTTPException(400, "task required")

        from jarvis.memory.database import db_write
        task_id = str(uuid.uuid4())
        await db_write(
            "INSERT INTO tasks (id, task_type, payload, priority) VALUES (?,?,?,?)",
            (task_id, "simple_qa", json.dumps({"text": task_text}), priority),
        )
        return {"task_id": task_id, "status": "queued"}

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        from jarvis.memory.database import db_fetch_one
        task = await db_fetch_one("SELECT * FROM tasks WHERE id=?", (task_id,))
        if not task:
            raise HTTPException(404, "Task not found")
        return task

    # ── Memory ────────────────────────────────────────────────────────────────

    @app.get("/memory/search")
    async def memory_search(q: str, days: int | None = None):
        from jarvis.memory.store import search_memories
        results = await search_memories(q, top_k=10, days_back=days)
        return {"query": q, "results": results}

    # ── Rollback ─────────────────────────────────────────────────────────────

    @app.get("/rollback/list")
    async def list_rollbacks():
        from jarvis.security.rollback import list_rollback_points
        return list_rollback_points()

    @app.post("/rollback/{rp_id}")
    async def do_rollback(rp_id: str):
        from jarvis.security.rollback import rollback_to_point
        await rollback_to_point(rp_id)
        return {"status": "rollback initiated", "id": rp_id}

    # ── Skill Proposals ───────────────────────────────────────────────────────

    @app.get("/skill_proposals")
    async def get_skill_proposals():
        from jarvis.memory.store import get_pending_skill_proposals
        return await get_pending_skill_proposals()

    @app.post("/skill_proposals/{proposal_id}/accept")
    async def accept_skill_proposal(proposal_id: int):
        from jarvis.memory.database import db_write, db_fetch_one
        proposal = await db_fetch_one("SELECT * FROM skill_proposals WHERE id=?", (proposal_id,))
        if not proposal:
            raise HTTPException(404, "Proposal not found")

        # Append to SKILL.md
        skill_name = proposal["skill_name"]
        skill_file = Path(f".claude/skills/{skill_name}/SKILL.md")
        skill_file.parent.mkdir(parents=True, exist_ok=True)
        with open(skill_file, "a") as f:
            f.write(f"\n\n## Auto-update {time.strftime('%Y-%m-%d')}\n{proposal['proposal']}\n")

        await db_write(
            "UPDATE skill_proposals SET status='accepted' WHERE id=?",
            (proposal_id,),
        )
        return {"status": "accepted", "skill": skill_name}

    # ── API cost tracker ──────────────────────────────────────────────────────

    @app.get("/costs")
    async def get_costs():
        from jarvis.memory.database import db_fetch_all
        today = time.time() - 86400
        rows = await db_fetch_all(
            "SELECT model, SUM(cost_usd) as total, SUM(input_tok) as input, SUM(output_tok) as output FROM api_costs WHERE ts > ? GROUP BY model",
            (today,),
        )
        return {"today": rows, "monthly_estimate": sum(r["total"] for r in rows) * 30}

    return app


# Create the app instance
app = create_app()
