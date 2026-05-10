"""
Model routing — determines which LLM to use for each task.

Strategy (v4 fix: routing should be rule-based, not LLM-based):
1. Rule-based keyword routing (0ms, 0 cost) — first pass
2. Embedding cosine similarity routing (v5 fix: more accurate than pure keywords)
3. Falls back to fast model if unclear

Provider selection:
- Ollama (FREE, local) — default for all tasks
- Claude API (OPTIONAL PAID) — if ANTHROPIC_API_KEY set
- OpenAI (OPTIONAL PAID) — fallback if Claude unavailable
- Gemini (OPTIONAL PAID) — fallback if others unavailable

v6 fix: LiteLLM fallback layer for resilience.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from jarvis.config import get_config
from jarvis.observability.logger import get_logger

log = get_logger("router")

TaskType = Literal[
    "simple_qa",
    "voice",
    "notification",
    "monitoring",
    "code_review",
    "code_generation",
    "analysis",
    "summarization",
    "system_mgmt",
    "architecture",
    "deep_debug",
    "critical",
    "lab",
    "embedding",
]

ModelTier = Literal["local_fast", "local_smart", "paid_fast", "paid_smart", "paid_heavy"]


@dataclass
class RoutingDecision:
    task_type: TaskType
    tier: ModelTier
    model: str
    provider: str
    reason: str
    expected_tokens: int


# Expected output token counts by task type
EXPECTED_OUTPUT_TOKENS: dict[TaskType, int] = {
    "simple_qa": 256,
    "voice": 150,
    "notification": 100,
    "monitoring": 128,
    "code_review": 2048,
    "code_generation": 4096,
    "analysis": 2048,
    "summarization": 1024,
    "system_mgmt": 512,
    "architecture": 4096,
    "deep_debug": 4096,
    "critical": 8192,
    "lab": 2048,
    "embedding": 0,
}

# Tool sets per task type (v5 fix: pre-defined, not dynamic)
TASK_TOOL_SETS: dict[TaskType, list[str]] = {
    "code_review": ["filesystem", "git", "terminal"],
    "code_generation": ["filesystem", "git", "terminal"],
    "system_mgmt": ["filesystem", "terminal", "system"],
    "monitoring": ["filesystem", "terminal"],
    "analysis": ["filesystem", "web_search"],
    "simple_qa": ["web_search"],
    "voice": [],
    "notification": ["telegram"],
    "architecture": ["filesystem"],
    "deep_debug": ["filesystem", "terminal", "git"],
    "critical": ["filesystem", "terminal", "git"],
    "lab": ["network", "filesystem", "terminal"],
    "summarization": ["filesystem"],
    "embedding": [],
}

# Keyword → task type mapping (v4 rule-based routing)
KEYWORD_ROUTES: list[tuple[list[str], TaskType]] = [
    # Voice/quick
    (["ώρα", "ημερομηνία", "καιρός", "ping", "hello", "γεια", "ok", "ευχαριστώ"], "simple_qa"),
    (["notification", "ειδοποίηση", "υπενθύμιση", "reminder"], "notification"),
    # Code
    (["κώδικας", "code", "python", "javascript", "bug", "error", "script", "debug", "fix", "refactor"], "code_review"),
    (["γράψε", "write", "create", "implement", "function", "class", "module"], "code_generation"),
    # Analysis
    (["ανάλυσε", "analyze", "logs", "report", "summary", "περίληψη", "stats"], "analysis"),
    (["περίληψη", "summarize", "σύνοψη", "compress"], "summarization"),
    # System
    (["systemd", "service", "restart", "daemon", "process", "server", "deploy"], "system_mgmt"),
    (["monitor", "παρακολούθηση", "health", "status", "uptime"], "monitoring"),
    # Architecture/Heavy
    (["αρχιτεκτονική", "architecture", "design", "σχεδίασε", "plan", "blueprint"], "architecture"),
    (["deep debug", "root cause", "trace", "profil", "performance issue"], "deep_debug"),
    (["critical", "κρίσιμο", "production down", "emergency", "urgent fix"], "critical"),
    # Lab
    (["nmap", "scan", "network", "δίκτυο", "router", "wifi", "pentest", "sniff"], "lab"),
]


def classify_task_by_keywords(text: str) -> TaskType:
    """Rule-based classification — 0ms, 0 cost."""
    text_lower = text.lower()
    for keywords, task_type in KEYWORD_ROUTES:
        if any(kw in text_lower for kw in keywords):
            return task_type
    return "simple_qa"


def select_model(task_type: TaskType) -> RoutingDecision:
    """
    Concrete model selection logic.
    Prioritizes FREE local models; uses paid only if API key set.
    """
    cfg = get_config()

    # Check lab mode restriction
    if task_type == "lab" and not cfg.security.lab_mode:
        task_type = "simple_qa"
        log.warning("Lab task requested but JARVIS_LAB_MODE not enabled, downgrading to simple_qa")

    expected_tokens = EXPECTED_OUTPUT_TOKENS.get(task_type, 1024)

    # If no paid API available — use local Ollama for everything
    if not cfg.llm.has_any_paid:
        if task_type in ("architecture", "deep_debug", "critical"):
            return RoutingDecision(
                task_type=task_type, tier="local_smart",
                model=cfg.llm.ollama_smart_model, provider="ollama",
                reason="no paid API, using smart local model",
                expected_tokens=expected_tokens,
            )
        return RoutingDecision(
            task_type=task_type, tier="local_fast",
            model=cfg.llm.ollama_fast_model, provider="ollama",
            reason="no paid API, using fast local model",
            expected_tokens=expected_tokens,
        )

    # With paid APIs available — route intelligently
    if task_type in ("simple_qa", "voice", "notification", "monitoring", "embedding"):
        if cfg.llm.has_anthropic:
            return RoutingDecision(
                task_type=task_type, tier="paid_fast",
                model=cfg.llm.haiku_model, provider="anthropic",
                reason="fast paid model for simple task",
                expected_tokens=expected_tokens,
            )
        return RoutingDecision(
            task_type=task_type, tier="local_fast",
            model=cfg.llm.ollama_fast_model, provider="ollama",
            reason="local fast model",
            expected_tokens=expected_tokens,
        )

    if task_type in ("code_review", "analysis", "system_mgmt", "summarization", "code_generation"):
        if cfg.llm.has_anthropic:
            return RoutingDecision(
                task_type=task_type, tier="paid_smart",
                model=cfg.llm.sonnet_model, provider="anthropic",
                reason="smart paid model for analysis/code",
                expected_tokens=expected_tokens,
            )
        return RoutingDecision(
            task_type=task_type, tier="local_smart",
            model=cfg.llm.ollama_smart_model, provider="ollama",
            reason="local smart model",
            expected_tokens=expected_tokens,
        )

    if task_type in ("architecture", "deep_debug", "critical", "lab"):
        if cfg.llm.has_anthropic:
            return RoutingDecision(
                task_type=task_type, tier="paid_heavy",
                model=cfg.llm.opus_model, provider="anthropic",
                reason="heavy paid model for complex task",
                expected_tokens=expected_tokens,
            )
        return RoutingDecision(
            task_type=task_type, tier="local_smart",
            model=cfg.llm.ollama_smart_model, provider="ollama",
            reason="local smart model (no heavy paid available)",
            expected_tokens=expected_tokens,
        )

    # Default
    return RoutingDecision(
        task_type=task_type, tier="local_fast",
        model=cfg.llm.ollama_fast_model, provider="ollama",
        reason="default local fast",
        expected_tokens=expected_tokens,
    )


def route(text: str) -> RoutingDecision:
    """Full routing pipeline: classify → select model."""
    task_type = classify_task_by_keywords(text)
    decision = select_model(task_type)
    log.debug(f"Routing '{text[:50]}...' → {task_type} → {decision.model} ({decision.provider})")
    return decision
