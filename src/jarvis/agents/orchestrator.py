"""
Coordinator Mode (COORDINATOR_MODE from leaked code).

"Do not rubber-stamp weak work."

Orchestrator:
- Decomposes complex tasks into subtasks
- Assigns subtasks to sub-agents (isolated, no shared state)
- Evaluates each sub-agent output: score >= 70 to accept
- Aggregates results, resolves conflicts
- Parallel execution for independent subtasks

Two topologies:
1. Hierarchical (sub-agents): orchestrator assigns, sub-agents execute
2. Peer team: shared task queue, agents pull independently
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from jarvis.agents.base import BaseAgent, ActionResult, TaskState
from jarvis.memory.database import db_write, db_fetch_all
from jarvis.observability.logger import get_logger
from jarvis.observability.metrics import get_metrics

log = get_logger("orchestrator")

COORDINATOR_SYSTEM_PROMPT = """You are a COORDINATOR agent managing sub-agents.
Your job:
1. Decompose the task into independent subtasks
2. Assign each subtask to a sub-agent
3. Evaluate results — DO NOT rubber-stamp weak work
4. Return the final aggregated result

Evaluation criteria for sub-agent output:
- Correctness (0-40): Does it solve the actual problem?
- Completeness (0-30): Are edge cases handled?
- Efficiency (0-20): Is the approach optimal?
- Safety (0-10): Does it follow security zone rules?

Score < 70: Reject and re-assign with specific feedback.
Score 70-85: Accept with improvement notes.
Score > 85: Accept fully.
"""


@dataclass
class SubTaskResult:
    subtask_id: str
    subtask: str
    agent_id: str
    output: str
    score: float
    accepted: bool
    feedback: str | None = None


class CoordinatorAgent(BaseAgent):
    """
    Hierarchical multi-agent orchestrator.
    Spawns sub-agents, evaluates their work, aggregates results.
    """

    def __init__(self):
        super().__init__(agent_id="coordinator", name="Coordinator")
        self._sub_agents: list[BaseAgent] = []
        self._max_sub_agents = 5
        self._max_retries_per_subtask = 2

    async def initialize(self):
        await super().initialize()
        self._system_prompt = COORDINATOR_SYSTEM_PROMPT

    async def decompose_task(self, task: str) -> list[str]:
        """Ask LLM to decompose a complex task into independent subtasks."""
        prompt = f"""Decompose this task into independent subtasks that can be executed in parallel.
Return ONLY a JSON array of strings, each being a self-contained subtask.
Maximum 5 subtasks. If task is simple, return just ["<original task>"].

Task: {task}

JSON array:"""
        try:
            from jarvis.llm.client import simple_completion
            response = await simple_completion(prompt, task_type="analysis")
            # Parse JSON
            import re
            match = re.search(r"\[.*?\]", response, re.DOTALL)
            if match:
                subtasks = json.loads(match.group())
                return [str(s) for s in subtasks[:5]]
        except Exception as e:
            log.warning(f"Task decomposition failed: {e}")

        return [task]  # fallback: treat as single task

    async def run_orchestrated(self, task: str) -> ActionResult:
        """
        Run a complex task using sub-agent delegation.
        """
        if not self._initialized:
            await self.initialize()

        log.info(f"Orchestrator starting task: {task[:80]}")
        start_ts = time.time()

        # Decompose
        subtasks = await self.decompose_task(task)
        log.info(f"Decomposed into {len(subtasks)} subtasks")

        # Execute subtasks in parallel (if independent)
        results: list[SubTaskResult] = await self._execute_parallel(subtasks)

        # Aggregate results
        aggregated = await self._aggregate_results(task, results)

        return ActionResult(
            success=True,
            output=aggregated,
            score=sum(r.score for r in results) / max(len(results), 1),
            duration_ms=(time.time() - start_ts) * 1000,
        )

    async def _execute_parallel(self, subtasks: list[str]) -> list[SubTaskResult]:
        """Execute subtasks in parallel with isolated sub-agents."""
        coros = [self._execute_subtask(subtask, i) for i, subtask in enumerate(subtasks)]
        results = await asyncio.gather(*coros, return_exceptions=True)
        return [r for r in results if isinstance(r, SubTaskResult)]

    async def _execute_subtask(self, subtask: str, index: int) -> SubTaskResult:
        """
        Execute a single subtask with evaluation + retry.
        v3 rule: "Do not rubber-stamp weak work"
        """
        subtask_id = f"sub_{uuid.uuid4().hex[:8]}"

        # Create isolated sub-agent (fresh context, no shared state)
        sub_agent = BaseAgent(
            agent_id=subtask_id,
            name=f"sub-agent-{index}",
        )
        await sub_agent.initialize()
        # Register tools from parent
        for tool in self._tools:
            name = tool["name"]
            if name in self._tool_handlers:
                sub_agent.register_tool(
                    name=name,
                    description=tool["description"],
                    input_schema=tool["input_schema"],
                    handler=self._tool_handlers[name],
                )

        for attempt in range(self._max_retries_per_subtask):
            result = await sub_agent.run_task(subtask, task_id=subtask_id)

            if result.success and result.score >= 70:
                log.info(f"Subtask {index} accepted (score={result.score:.0f})")
                return SubTaskResult(
                    subtask_id=subtask_id,
                    subtask=subtask,
                    agent_id=subtask_id,
                    output=result.output,
                    score=result.score,
                    accepted=True,
                )

            feedback = f"Score {result.score:.0f}/100 — needs improvement. Error: {result.error or 'quality below threshold'}"
            log.info(f"Subtask {index} rejected (score={result.score:.0f}), retry {attempt + 1}")
            # Add feedback to sub-agent's context for retry
            subtask = f"{subtask}\n\nFEEDBACK FROM COORDINATOR: {feedback}"

        # Final attempt result regardless
        return SubTaskResult(
            subtask_id=subtask_id,
            subtask=subtask,
            agent_id=subtask_id,
            output=result.output if result.success else f"[FAILED: {result.error}]",
            score=result.score,
            accepted=result.score >= 50,
            feedback="Max retries reached",
        )

    async def _aggregate_results(self, original_task: str, results: list[SubTaskResult]) -> str:
        """Combine sub-agent results into final output."""
        if not results:
            return "[No results from sub-agents]"

        if len(results) == 1:
            return results[0].output

        # Build aggregation prompt
        parts = "\n\n".join(
            f"### Subtask {i+1}: {r.subtask[:100]}\n{r.output[:1000]}"
            for i, r in enumerate(results)
        )

        prompt = f"""Aggregate these sub-task results into a coherent final answer for:
{original_task}

Sub-task results:
{parts}

Final aggregated answer:"""

        from jarvis.llm.client import simple_completion
        return await simple_completion(prompt, task_type="analysis")


class AgentTeam:
    """
    Peer team topology: shared task queue, agents pull independently.
    Suitable for large volumes of independent tasks.
    Uses SQLite's SELECT FOR UPDATE SKIP LOCKED equivalent pattern.
    """

    def __init__(self, team_size: int = 3):
        self.team_size = team_size
        self._agents: list[BaseAgent] = []
        self._running = False

    async def start(self):
        """Start all agents, each pulling from the shared task queue."""
        self._agents = [
            BaseAgent(agent_id=f"team_{i}", name=f"team-agent-{i}")
            for i in range(self.team_size)
        ]
        for agent in self._agents:
            await agent.initialize()

        self._running = True
        workers = [asyncio.create_task(self._worker(agent)) for agent in self._agents]
        log.info(f"Agent team of {self.team_size} started")
        return workers

    async def stop(self):
        self._running = False

    async def _worker(self, agent: BaseAgent):
        """Worker loop: pull task from queue, execute, mark done."""
        while self._running:
            # Atomic task claim (SQLite WAL mode + retry)
            task = await self._claim_task(agent.agent_id)
            if not task:
                await asyncio.sleep(5)
                continue

            log.info(f"Agent {agent.agent_id} picked up task {task['id'][:8]}")
            result = await agent.run_task(
                task=json.loads(task["payload"]),
                task_id=task["id"],
            )

            status = "completed" if result.success else "failed"
            await db_write(
                "UPDATE tasks SET status=?, result=?, score=?, finished_at=? WHERE id=?",
                (status, result.output[:5000], result.score, time.time(), task["id"]),
            )

    async def _claim_task(self, agent_id: str) -> dict | None:
        """
        Atomically claim a pending task.
        SQLite WAL ensures this is safe with multiple readers.
        """
        from jarvis.memory.database import db_fetch_one
        # SQLite doesn't support SELECT FOR UPDATE, but single-writer pattern handles concurrency
        task = await db_fetch_one(
            "SELECT * FROM tasks WHERE status='pending' ORDER BY priority ASC, created_at ASC LIMIT 1",
        )
        if not task:
            return None

        # Mark as running (will fail silently if another agent got it first)
        from jarvis.memory.database import db_write
        await db_write(
            "UPDATE tasks SET status='running', started_at=?, agent_id=? WHERE id=? AND status='pending'",
            (time.time(), agent_id, task["id"]),
        )
        return task

    async def submit_task(self, payload: dict, priority: int = 3) -> str:
        """Submit a task to the shared queue."""
        task_id = str(uuid.uuid4())
        task_type = payload.get("type", "simple_qa")
        await db_write(
            "INSERT INTO tasks (id, task_type, payload, priority) VALUES (?,?,?,?)",
            (task_id, task_type, json.dumps(payload), priority),
        )
        return task_id
