"""
4-Level Memory System (v6 architecture):

L1: Working Memory — in-process Python dict (current session)
L2: Session Memory — SQLite sessions table (24h, compressed after idle)
L3: Long-term Memory — SQLite memories table + embeddings (semantic search)
L4: Procedural Memory — SKILL.md filesystem files (read at startup)

v6 fixes:
- FIFO trim removes PAIRS (user+assistant), never single messages
- System prompt ALWAYS pinned (never trimmed)
- Pre-flight token count with tiktoken
- Debounced session compression (15min idle)
- Temporal context injected into retrieval results
- Hybrid search: vector + FTS5 keyword + RRF fusion
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from jarvis.config import get_config, JARVIS_HOME
from jarvis.memory.database import db_write, db_fetch_one, db_fetch_all
from jarvis.memory.embeddings import embed_text, embed_texts, cosine_similarity, pack_embedding, unpack_embedding
from jarvis.observability.logger import get_logger
from jarvis.observability.metrics import get_metrics

log = get_logger("memory")


# ─── L1: Working Memory ───────────────────────────────────────────────────────

@dataclass
class WorkingMemory:
    """In-context state for the current session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(default_factory=list)
    task_state: dict = field(default_factory=dict)
    context_vars: dict = field(default_factory=dict)
    last_message_time: float = field(default_factory=time.time)
    _compression_task: Any = None

    def add_message(self, role: str, content: Any, tool_name: str | None = None):
        self.messages.append({"role": role, "content": content, "tool_name": tool_name})
        self.last_message_time = time.time()

    def is_idle(self, threshold_s: int = 900) -> bool:
        return (time.time() - self.last_message_time) > threshold_s

    def token_count_estimate(self) -> int:
        """Fast estimate: ~4 chars per token."""
        total = 0
        for m in self.messages:
            c = m.get("content", "")
            if isinstance(c, str):
                total += len(c) // 4
            elif isinstance(c, list):
                total += sum(len(str(item)) // 4 for item in c)
            elif isinstance(c, dict):
                total += len(json.dumps(c)) // 4
        return total


def trim_context(messages: list[dict], system: str, max_tokens: int) -> list[dict]:
    """
    v6 fix: removes message PAIRS (user+assistant) not single messages.
    System message always pinned first.

    Args:
        messages: list of message dicts (role + content)
        system: system prompt text (already separate)
        max_tokens: maximum total token budget
    Returns:
        trimmed messages list
    """
    system_tokens = len(system) // 4
    rest = [m for m in messages if m.get("role") != "system"]

    avg_pair_tokens = 500  # rough average for fast batch removal
    total = system_tokens + sum(len(str(m.get("content", ""))) // 4 for m in rest)

    if total <= max_tokens:
        return rest

    # Batch removal: estimate pairs to remove
    pairs_to_remove = max(0, (total - max_tokens) // avg_pair_tokens)
    rest = rest[pairs_to_remove * 2:]  # remove N pairs at once

    # Ensure starts with 'user' (API requirement)
    if rest and rest[0].get("role") == "assistant":
        rest = rest[1:]

    # Fine-tune single pair removal if still over
    while len(rest) > 2:
        total = system_tokens + sum(len(str(m.get("content", ""))) // 4 for m in rest)
        if total <= max_tokens:
            break
        # Remove oldest pair
        rest = rest[2:]
        if rest and rest[0].get("role") == "assistant":
            rest = rest[1:]

    return rest


def inject_time_context(message: str) -> str:
    """v5 fix: inject current time in user's timezone on every message."""
    cfg = get_config()
    tz = cfg.server.tz
    now = datetime.now(tz)
    time_str = now.strftime("%A %d/%m/%Y %H:%M %Z")
    return f"[Τρέχων χρόνος: {time_str}]\n{message}"


# ─── L2/L3: Persistent Memory Operations ──────────────────────────────────────

async def save_session_message(
    session_id: str, role: str, content: Any, tool_name: str | None = None
):
    """Save a conversation message to sessions table (L2)."""
    content_str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    await db_write(
        "INSERT INTO sessions (session_id, role, content, tool_name) VALUES (?,?,?,?)",
        (session_id, role, content_str, tool_name),
    )
    get_metrics().memory_writes.inc()


async def load_session_messages(session_id: str) -> list[dict]:
    """Load all messages for a session (L2)."""
    rows = await db_fetch_all(
        "SELECT role, content, tool_name FROM sessions WHERE session_id=? AND compressed=0 ORDER BY created_at",
        (session_id,),
    )
    get_metrics().memory_reads.inc()
    return [
        {"role": r["role"], "content": r["content"], "tool_name": r["tool_name"]}
        for r in rows
    ]


async def save_memory(
    content: str,
    memory_type: str = "episodic",
    importance: float = 0.5,
    tags: list[str] | None = None,
    session_id: str | None = None,
    source: str | None = None,
) -> int:
    """
    Save to long-term memory (L3).
    Automatically generates and stores embedding.
    """
    embedding = await embed_text(content)
    embedding_bytes = pack_embedding(embedding)
    tags_json = json.dumps(tags or [], ensure_ascii=False)

    rowid = await db_write(
        """INSERT INTO memories (content, embedding, importance, memory_type, tags, session_id, source)
           VALUES (?,?,?,?,?,?,?)""",
        (content, embedding_bytes, importance, memory_type, tags_json, session_id, source),
    )

    # Also add to FTS index
    await db_write(
        "INSERT INTO memories_fts(rowid, content) VALUES (?,?)",
        (rowid, content),
    )

    get_metrics().memory_writes.inc()
    log.debug(f"Saved {memory_type} memory: {content[:80]}...")
    return rowid


async def search_memories(
    query: str,
    top_k: int = 5,
    memory_type: str | None = None,
    min_similarity: float = 0.6,
    days_back: int | None = None,
) -> list[dict]:
    """
    Hybrid search: vector similarity + FTS5 keyword search + RRF fusion.
    v5 fix: vector-only was missing keyword hits.
    """
    cfg = get_config()
    query_vec = await embed_text(query)
    cutoff_ts = (time.time() - days_back * 86400) if days_back else 0.0

    # Build type filter
    type_clause = "AND memory_type=?" if memory_type else ""
    type_params = (memory_type,) if memory_type else ()

    # 1. Vector search — fetch candidates (brute force, fine for <100K records)
    rows = await db_fetch_all(
        f"""SELECT id, content, embedding, importance, memory_type, tags, created_at
            FROM memories
            WHERE embedding IS NOT NULL {type_clause}
            AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT 1000""",
        type_params + (cutoff_ts,),
    )

    # Compute cosine similarities
    vector_results: list[tuple[int, float, dict]] = []
    for row in rows:
        if row["embedding"]:
            mem_vec = unpack_embedding(row["embedding"])
            sim = cosine_similarity(query_vec, mem_vec)
            if sim >= min_similarity:
                vector_results.append((row["id"], sim, row))

    # Sort by similarity
    vector_results.sort(key=lambda x: x[1], reverse=True)
    vector_top = {r[0]: (rank + 1, r[1], r[2]) for rank, r in enumerate(vector_results[:top_k * 2])}

    # 2. FTS5 keyword search
    try:
        fts_rows = await db_fetch_all(
            f"""SELECT m.id, m.content, m.embedding, m.importance, m.memory_type, m.tags, m.created_at,
                       bm25(memories_fts) as score
                FROM memories_fts
                JOIN memories m ON memories_fts.rowid = m.id
                WHERE memories_fts MATCH ?
                {type_clause}
                ORDER BY bm25(memories_fts)
                LIMIT {top_k * 2}""",
            (query,) + type_params,
        )
        fts_top = {r["id"]: (rank + 1, r) for rank, r in enumerate(fts_rows)}
    except Exception:
        fts_top = {}

    # 3. RRF fusion: score = 1/(k+rank_vector) + 1/(k+rank_keyword)
    k = 60
    all_ids = set(vector_top.keys()) | set(fts_top.keys())
    fused: list[tuple[float, dict]] = []

    for doc_id in all_ids:
        score = 0.0
        row_data = None
        sim_score = 0.0

        if doc_id in vector_top:
            rank, sim, row = vector_top[doc_id]
            score += 1.0 / (k + rank)
            row_data = row
            sim_score = sim

        if doc_id in fts_top:
            rank, row = fts_top[doc_id]
            score += 1.0 / (k + rank)
            if row_data is None:
                row_data = row

        if row_data:
            fused.append((score, {**row_data, "similarity": sim_score, "fusion_score": score}))

    fused.sort(key=lambda x: x[0], reverse=True)

    # Attach human-readable timestamps
    tz = get_config().server.tz
    results = []
    for _, item in fused[:top_k]:
        ts = item.get("created_at", 0)
        dt = datetime.fromtimestamp(ts, tz=tz)
        now = datetime.now(tz)
        delta = now - dt
        if delta.days == 0:
            relative = "σήμερα"
        elif delta.days == 1:
            relative = "χθες"
        elif delta.days < 7:
            relative = f"{delta.days} μέρες πριν"
        elif delta.days < 30:
            relative = f"{delta.days // 7} εβδομάδες πριν"
        else:
            relative = f"{delta.days // 30} μήνες πριν"

        item["time_human"] = f"{dt.strftime('%d/%m/%Y %H:%M')} ({relative})"
        item["tags"] = json.loads(item.get("tags", "[]"))
        results.append(item)

    get_metrics().memory_reads.inc()
    return results


# ─── L4: Procedural Memory (SKILL.md files) ───────────────────────────────────

def load_procedural_memory() -> dict[str, str]:
    """Load all SKILL.md files from ~/.claude/skills/ and jarvis/skills/."""
    skills: dict[str, str] = {}

    # Project skills
    project_skills = Path(".claude/skills")
    if project_skills.exists():
        for skill_dir in project_skills.iterdir():
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                skills[skill_dir.name] = skill_file.read_text()

    # Jarvis home skills
    jarvis_skills = JARVIS_HOME / "skills"
    if jarvis_skills.exists():
        for skill_dir in jarvis_skills.iterdir():
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                skills[f"custom/{skill_dir.name}"] = skill_file.read_text()

    return skills


async def propose_skill_update(skill_name: str, proposal: str):
    """
    v5 fix: agent PROPOSES skill updates, never writes SKILL.md directly.
    Human reviews via /skill_proposals command.
    """
    await db_write(
        "INSERT INTO skill_proposals (skill_name, proposal) VALUES (?,?)",
        (skill_name, proposal),
    )
    log.info(f"Skill proposal created for '{skill_name}'")


async def get_pending_skill_proposals() -> list[dict]:
    return await db_fetch_all(
        "SELECT * FROM skill_proposals WHERE status='pending' ORDER BY created_at DESC",
    )


# ─── autoDream: Memory Consolidation ──────────────────────────────────────────

async def compress_session(session_id: str, llm_call_fn) -> str | None:
    """
    Debounced session compression (runs after 15min idle).
    - Reads uncompressed messages
    - Summarizes with LLM
    - Saves summary to long-term semantic memory
    - Marks session messages as compressed
    """
    msgs = await db_fetch_all(
        "SELECT role, content FROM sessions WHERE session_id=? AND compressed=0 ORDER BY created_at",
        (session_id,),
    )
    if len(msgs) < 4:
        return None

    # Build conversation text
    convo = "\n".join(f"{m['role'].upper()}: {m['content'][:500]}" for m in msgs)

    # Ask LLM to extract facts
    summary_prompt = f"""Εξάγαγε τα σημαντικά facts, αποφάσεις και αποτελέσματα από αυτή τη συνομιλία.
Γράψε σε bullet points, μόνο τα ουσιαστικά. Χωρίς εισαγωγές.

Συνομιλία:
{convo[:8000]}

Facts:"""

    try:
        summary = await llm_call_fn(summary_prompt, task_type="summarization")
        if summary:
            await save_memory(
                content=summary,
                memory_type="semantic",
                importance=0.7,
                tags=["session_summary"],
                session_id=session_id,
                source="autoDream",
            )
            # Mark session messages as compressed
            await db_write(
                "UPDATE sessions SET compressed=1 WHERE session_id=?",
                (session_id,),
            )
            log.info(f"Session {session_id[:8]} compressed → {len(summary)} chars")
            return summary
    except Exception as e:
        log.error(f"Session compression failed: {e}")
    return None


async def run_auto_dream(llm_call_fn):
    """
    autoDream: consolidate episodic memories into semantic facts.
    Runs when idle > 15 minutes (called by KAIROS daemon).
    """
    log.info("autoDream starting memory consolidation...")

    # Fetch recent unconsolidated episodic memories
    cutoff = time.time() - 86400 * 7  # last 7 days
    episodes = await db_fetch_all(
        """SELECT id, content, created_at FROM memories
           WHERE memory_type='episodic' AND consolidated=0 AND created_at > ?
           ORDER BY created_at DESC LIMIT 50""",
        (cutoff,),
    )

    if not episodes:
        log.info("autoDream: nothing to consolidate")
        return

    # Group into batches
    batch_text = "\n---\n".join(e["content"][:500] for e in episodes)
    consolidation_prompt = f"""Ανάλυσε τα παρακάτω επεισόδια και εξάγαγε:
1. Επαναλαμβανόμενα patterns
2. Σημαντικά facts για τον χρήστη
3. Αντιφάσεις (και ποιο είναι πιο πρόσφατο/αξιόπιστο)

Επεισόδια:
{batch_text[:6000]}

Consolidated facts (bullet points):"""

    try:
        consolidated = await llm_call_fn(consolidation_prompt, task_type="analysis")
        if consolidated:
            await save_memory(
                content=consolidated,
                memory_type="semantic",
                importance=0.8,
                tags=["auto_dream", "consolidated"],
                source="autoDream",
            )
            # Mark episodes as consolidated
            ids = tuple(e["id"] for e in episodes)
            placeholders = ",".join("?" * len(ids))
            await db_write(
                f"UPDATE memories SET consolidated=1 WHERE id IN ({placeholders})",
                ids,
            )
            log.info(f"autoDream consolidated {len(episodes)} episodes into semantic memory")
    except Exception as e:
        log.error(f"autoDream failed: {e}")
