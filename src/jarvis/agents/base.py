"""
Base agent class with:
- Full agentic tool-use loop (v6 fix)
- Action evaluation/scoring
- Task state checkpointing
- Failure handling matrix (from v3 blueprint)
- Self-improvement proposal generation
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

from jarvis.config import get_config
from jarvis.llm.client import run_agent, simple_completion
from jarvis.llm.router import route, select_model, TASK_TOOL_SETS
from jarvis.memory.database import db_write, db_fetch_one
from jarvis.memory.store import save_memory, load_procedural_memory, propose_skill_update
from jarvis.observability.logger import get_logger, get_audit
from jarvis.observability.metrics import get_metrics
from jarvis.security.rollback import create_rollback_point

log = get_logger("agent")


@dataclass
class TaskState:
    """v6 fix: custom TaskState (asyncio.Task has no .checkpoint())"""
    task_id: str
    step: int = 0
    data: dict = field(default_factory=dict)
    status: str = "running"

    async def save(self):
        from jarvis.memory.database import db_write
        await db_write(
            "INSERT OR REPLACE INTO checkpoints (task_id, state_json) VALUES (?,?)",
            (self.task_id, json.dumps(asdict(self))),
        )

    @classmethod
    async def load(cls, task_id: str) -> Optional["TaskState"]:
        from jarvis.memory.database import db_fetch_one
        row = await db_fetch_one(
            "SELECT state_json FROM checkpoints WHERE task_id=?",
            (task_id,),
        )
        if row:
            return cls(**json.loads(row["state_json"]))
        return None


@dataclass
class ActionResult:
    success: bool
    output: str
    score: float = 0.0
    error: str | None = None
    tokens_used: int = 0
    duration_ms: float = 0.0


class BaseAgent:
    """
    Base agent with tool execution, evaluation, and self-improvement.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "jarvis",
        system_prompt: str | None = None,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.name = name
        self._system_prompt = system_prompt
        self._tools: list[dict] = []
        self._tool_handlers: dict[str, Callable] = {}
        self._skill_cache: dict[str, str] = {}
        self._initialized = False

    async def initialize(self):
        """Load system prompt and skills."""
        if self._initialized:
            return

        if self._system_prompt is None:
            from pathlib import Path
            claude_md = Path("CLAUDE.md")
            self._system_prompt = claude_md.read_text() if claude_md.exists() else (
                "You are JARVIS, an autonomous assistant. Be concise and direct."
            )

        # Load procedural memory (SKILL.md files)
        self._skill_cache = load_procedural_memory()
        if self._skill_cache:
            skill_text = "\n\n".join(f"## {name}\n{content}" for name, content in self._skill_cache.items())
            self._system_prompt += f"\n\n# Loaded Skills\n{skill_text[:5000]}"

        self._initialized = True
        log.info(f"Agent {self.agent_id} initialized ({len(self._skill_cache)} skills loaded)")

    def register_tool(self, name: str, description: str, input_schema: dict, handler: Callable):
        """Register a tool for this agent."""
        self._tools.append({
            "name": name,
            "description": description,
            "input_schema": input_schema,
        })
        self._tool_handlers[name] = handler

    async def _execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute a registered tool."""
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            return f"[Unknown tool: {tool_name}]"
        try:
            result = await handler(**args) if asyncio.iscoroutinefunction(handler) else handler(**args)
            return result
        except Exception as e:
            log.error(f"Tool {tool_name} failed: {e}")
            return f"[Tool error: {e}]"

    async def run_task(
        self,
        task: str,
        task_id: str | None = None,
        context: dict | None = None,
        max_retries: int = 3,
    ) -> ActionResult:
        """
        Run a task with full agentic loop, evaluation, and retry.
        """
        if not self._initialized:
            await self.initialize()

        task_id = task_id or str(uuid.uuid4())
        metrics = get_metrics()
        start_ts = time.time()

        # Determine routing
        decision = route(task)
        tool_set = TASK_TOOL_SETS.get(decision.task_type, [])
        active_tools = [t for t in self._tools if t["name"] in tool_set or not tool_set]

        # Get memory context
        from jarvis.memory.store import search_memories, inject_time_context
        memories = await search_memories(task, top_k=5)
        memory_context = ""
        if memories:
            memory_items = "\n".join(f"- [{m['time_human']}] {m['content'][:200]}" for m in memories[:3])
            memory_context = f"\n\nΣχετικές αναμνήσεις:\n{memory_items}"

        enriched_task = inject_time_context(task) + memory_context

        # Create task record
        await db_write(
            "INSERT INTO tasks (id, task_type, payload, status, agent_id) VALUES (?,?,?,?,?)",
            (task_id, decision.task_type, task[:1000], "running", self.agent_id),
        )

        task_state = TaskState(task_id=task_id, data={"task": task})
        await task_state.save()

        last_error = None
        for attempt in range(max_retries):
            try:
                # Create rollback point before mutating actions
                if decision.task_type in ("code_generation", "system_mgmt", "architecture"):
                    await create_rollback_point(f"pre-task-{task_id[:8]}")

                messages = [{"role": "user", "content": enriched_task}]

                text, usage = await run_agent(
                    messages=messages,
                    tools=active_tools,
                    system=self._system_prompt,
                    decision=decision,
                    tool_executor=self._execute_tool,
                )

                duration_ms = (time.time() - start_ts) * 1000

                # Evaluate output quality
                score = await self._evaluate_output(task, text, decision.task_type)

                # Update task record
                await db_write(
                    "UPDATE tasks SET status='completed', result=?, score=?, finished_at=? WHERE id=?",
                    (text[:5000], score, time.time(), task_id),
                )

                # Save to episodic memory
                await save_memory(
                    content=f"Task: {task[:200]}\nResult: {text[:300]}\nScore: {score}",
                    memory_type="episodic",
                    importance=min(1.0, score / 100),
                    tags=[decision.task_type, self.agent_id],
                )

                # Self-improvement: propose skill update if score < 70
                if score < 70:
                    await self._propose_improvement(task, text, score, decision.task_type)

                metrics.tasks_total.inc()
                metrics.tasks_success.inc()
                metrics.task_score.observe(score)

                result = ActionResult(
                    success=True,
                    output=text,
                    score=score,
                    tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    duration_ms=duration_ms,
                )

                get_audit().log_action(
                    tool=f"agent:{decision.task_type}",
                    input_data=task[:200],
                    output=text[:500],
                    success=True,
                    duration_ms=duration_ms,
                    score=score,
                    model_used=decision.model,
                    tokens_used=result.tokens_used,
                )

                return result

            except Exception as e:
                last_error = str(e)
                log.warning(f"Task attempt {attempt + 1} failed: {e}")
                task_state.step = attempt + 1
                task_state.data["last_error"] = last_error
                await task_state.save()

                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # exponential backoff

        # All retries failed
        await db_write(
            "UPDATE tasks SET status='failed', error=?, finished_at=? WHERE id=?",
            (last_error, time.time(), task_id),
        )
        metrics.tasks_failed.inc()
        metrics.record_error()

        return ActionResult(
            success=False,
            output="",
            error=last_error,
            duration_ms=(time.time() - start_ts) * 1000,
        )

    async def _evaluate_output(self, task: str, output: str, task_type: str) -> float:
        """
        Evaluate output quality (0-100 score).
        Uses a lightweight evaluation LLM call.
        """
        if not output.strip():
            return 0.0
        if len(output) < 10:
            return 20.0

        # Quick heuristic evaluation (no LLM call to save tokens)
        score = 50.0

        # Length appropriateness
        if 50 < len(output) < 10000:
            score += 15

        # No error indicators
        error_keywords = ["[TOOL ERROR", "[ERROR:", "failed to", "exception", "traceback"]
        if not any(kw.lower() in output.lower() for kw in error_keywords):
            score += 15

        # Completeness: doesn't end mid-sentence
        if output.strip()[-1] in ".!?»\n":
            score += 10

        # Task-specific checks
        if task_type in ("code_generation", "code_review"):
            if "```" in output or "def " in output or "class " in output:
                score += 10  # Has code

        return min(100.0, score)

    async def _propose_improvement(self, task: str, output: str, score: float, task_type: str):
        """
        v4 Level 1 self-improvement: propose SKILL.md update (not auto-apply).
        Human reviews via /skill_proposals Telegram command.
        """
        proposal = f"""Low score task ({score:.0f}/100):
Task: {task[:200]}
Output quality issues: needs improvement
Suggested rule: Review this task type and add specific guidance.
"""
        await propose_skill_update(task_type, proposal)
        log.info(f"Skill improvement proposed for {task_type} (score={score:.0f})")
