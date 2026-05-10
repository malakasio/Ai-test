"""
In-process metrics collection with Prometheus-compatible exposition.
No external Prometheus required — serves /metrics endpoint directly.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Counter:
    name: str
    help: str
    _value: float = 0.0

    def inc(self, amount: float = 1.0):
        self._value += amount

    @property
    def value(self) -> float:
        return self._value


@dataclass
class Gauge:
    name: str
    help: str
    _value: float = 0.0

    def set(self, v: float):
        self._value = v

    def inc(self, amount: float = 1.0):
        self._value += amount

    def dec(self, amount: float = 1.0):
        self._value -= amount

    @property
    def value(self) -> float:
        return self._value


@dataclass
class Histogram:
    name: str
    help: str
    buckets: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    _observations: list[float] = field(default_factory=list)

    def observe(self, value: float):
        self._observations.append(value)
        if len(self._observations) > 10_000:
            self._observations = self._observations[-5_000:]

    @property
    def count(self) -> int:
        return len(self._observations)

    @property
    def sum(self) -> float:
        return sum(self._observations)

    @property
    def p50(self) -> float:
        if not self._observations:
            return 0.0
        s = sorted(self._observations)
        return s[len(s) // 2]

    @property
    def p95(self) -> float:
        if not self._observations:
            return 0.0
        s = sorted(self._observations)
        return s[int(len(s) * 0.95)]

    @property
    def p99(self) -> float:
        if not self._observations:
            return 0.0
        s = sorted(self._observations)
        return s[int(len(s) * 0.99)]


class MetricsRegistry:
    """Central registry of all metrics."""

    def __init__(self):
        # LLM metrics
        self.llm_requests_total = Counter("jarvis_llm_requests_total", "Total LLM requests")
        self.llm_errors_total = Counter("jarvis_llm_errors_total", "Total LLM errors")
        self.llm_tokens_total = Counter("jarvis_llm_tokens_total", "Total tokens consumed")
        self.llm_cost_usd = Counter("jarvis_llm_cost_usd", "Estimated LLM cost in USD")
        self.llm_latency = Histogram("jarvis_llm_latency_seconds", "LLM response latency")
        self.llm_tokens_per_day = Gauge("jarvis_llm_tokens_per_day", "Tokens consumed today")

        # Voice metrics
        self.voice_sessions_total = Counter("jarvis_voice_sessions_total", "Total voice sessions")
        self.voice_latency = Histogram("jarvis_voice_e2e_latency_seconds", "End-to-end voice latency")
        self.stt_latency = Histogram("jarvis_stt_latency_seconds", "STT latency")
        self.tts_latency = Histogram("jarvis_tts_latency_seconds", "TTS latency")
        self.barge_in_total = Counter("jarvis_barge_in_total", "Total barge-in events")

        # Agent metrics
        self.tasks_total = Counter("jarvis_tasks_total", "Total tasks processed")
        self.tasks_success = Counter("jarvis_tasks_success", "Successful tasks")
        self.tasks_failed = Counter("jarvis_tasks_failed", "Failed tasks")
        self.task_score = Histogram("jarvis_task_score", "Task quality score (0-100)")
        self.circuit_breaker_trips = Counter("jarvis_circuit_breaker_trips", "Circuit breaker activations")

        # Memory metrics
        self.memory_writes = Counter("jarvis_memory_writes_total", "Memory write operations")
        self.memory_reads = Counter("jarvis_memory_reads_total", "Memory read operations")
        self.memory_size = Gauge("jarvis_memory_records", "Total memory records")

        # System metrics
        self.uptime_seconds = Gauge("jarvis_uptime_seconds", "Agent uptime")
        self.kairos_runs = Counter("jarvis_kairos_runs_total", "KAIROS daemon runs")

        self._start_time = time.time()

        # Rolling error window for health checks (last 100 events)
        self._recent_errors: deque[float] = deque(maxlen=100)
        self._daily_tokens: dict[str, int] = defaultdict(int)

    def record_error(self):
        self._recent_errors.append(time.time())

    @property
    def error_rate_1h(self) -> float:
        cutoff = time.time() - 3600
        recent = sum(1 for t in self._recent_errors if t > cutoff)
        return recent / max(1.0, self.tasks_total.value) * 100

    def record_llm_cost(self, model: str, input_tokens: int, output_tokens: int):
        costs = {
            "claude-haiku": (0.00025, 0.00125),
            "claude-sonnet": (0.003, 0.015),
            "claude-opus": (0.015, 0.075),
            "gpt-4o-mini": (0.00015, 0.0006),
            "ollama": (0.0, 0.0),
        }
        prefix = next((k for k in costs if k in model.lower()), "ollama")
        input_cost, output_cost = costs[prefix]
        total_cost = (input_tokens / 1000) * input_cost + (output_tokens / 1000) * output_cost
        self.llm_cost_usd.inc(total_cost)
        self.llm_tokens_total.inc(input_tokens + output_tokens)

    def to_prometheus_text(self) -> str:
        """Exports all metrics in Prometheus text format."""
        lines = []
        metrics = [
            self.llm_requests_total, self.llm_errors_total, self.llm_tokens_total,
            self.llm_cost_usd, self.voice_sessions_total, self.barge_in_total,
            self.tasks_total, self.tasks_success, self.tasks_failed,
            self.circuit_breaker_trips, self.memory_writes, self.memory_reads,
            self.kairos_runs,
        ]
        for m in metrics:
            lines.append(f"# HELP {m.name} {m.help}")
            lines.append(f"# TYPE {m.name} counter")
            lines.append(f"{m.name} {m.value}")

        gauges = [self.memory_size, self.uptime_seconds, self.llm_tokens_per_day]
        for g in gauges:
            lines.append(f"# HELP {g.name} {g.help}")
            lines.append(f"# TYPE {g.name} gauge")
            lines.append(f"{g.name} {g.value}")

        histograms = [self.llm_latency, self.voice_latency, self.stt_latency, self.tts_latency, self.task_score]
        for h in histograms:
            lines.append(f"# HELP {h.name} {h.help}")
            lines.append(f"# TYPE {h.name} histogram")
            lines.append(f"{h.name}_count {h.count}")
            lines.append(f"{h.name}_sum {h.sum:.4f}")
            lines.append(f"{h.name}_p50 {h.p50:.4f}")
            lines.append(f"{h.name}_p95 {h.p95:.4f}")
            lines.append(f"{h.name}_p99 {h.p99:.4f}")

        return "\n".join(lines)

    def to_dashboard_dict(self) -> dict[str, Any]:
        self.uptime_seconds.set(time.time() - self._start_time)
        return {
            "uptime_seconds": int(time.time() - self._start_time),
            "llm": {
                "requests": int(self.llm_requests_total.value),
                "errors": int(self.llm_errors_total.value),
                "tokens_total": int(self.llm_tokens_total.value),
                "cost_usd": round(self.llm_cost_usd.value, 4),
                "p50_latency_ms": round(self.llm_latency.p50 * 1000, 1),
                "p95_latency_ms": round(self.llm_latency.p95 * 1000, 1),
            },
            "voice": {
                "sessions": int(self.voice_sessions_total.value),
                "barge_ins": int(self.barge_in_total.value),
                "p50_e2e_ms": round(self.voice_latency.p50 * 1000, 1),
                "p95_stt_ms": round(self.stt_latency.p95 * 1000, 1),
                "p95_tts_ms": round(self.tts_latency.p95 * 1000, 1),
            },
            "tasks": {
                "total": int(self.tasks_total.value),
                "success": int(self.tasks_success.value),
                "failed": int(self.tasks_failed.value),
                "avg_score": round(self.task_score.p50, 1),
                "error_rate_1h_pct": round(self.error_rate_1h, 2),
            },
            "memory": {
                "records": int(self.memory_size.value),
                "writes": int(self.memory_writes.value),
                "reads": int(self.memory_reads.value),
            },
        }


_registry: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry
