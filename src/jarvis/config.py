"""
Central configuration system for JARVIS.
All settings come from environment variables with sane free-tier defaults.
Paid API keys are fully OPTIONAL — the system runs 100% free without them.
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Literal


# ─── Paths ────────────────────────────────────────────────────────────────────

JARVIS_HOME = Path(os.environ.get("JARVIS_HOME", Path.home() / "jarvis"))
DATA_DIR = JARVIS_HOME / "data"
LOG_DIR = JARVIS_HOME / "logs"
SKILLS_DIR = JARVIS_HOME / "skills"
BACKUP_DIR = JARVIS_HOME / "backups"
LAB_DIR = JARVIS_HOME / "lab"
VAULT_DIR = JARVIS_HOME / "vault"
WORKSPACE_DIR = JARVIS_HOME / "workspace"

DB_PATH = DATA_DIR / "jarvis.db"
AUDIT_LOG_PATH = LOG_DIR / "audit.jsonl"
SESSION_SECRET_FILE = Path("/etc/jarvis/session_secret")


def _ensure_dirs():
    for d in [DATA_DIR, LOG_DIR, SKILLS_DIR, BACKUP_DIR, LAB_DIR, VAULT_DIR, WORKSPACE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ─── Credentials (read from systemd LoadCredential or env) ────────────────────

def _read_credential(name: str, env_fallback: str = "") -> str:
    """
    Reads from systemd CREDENTIALS_DIRECTORY first (most secure),
    then falls back to env var. Never errors — returns empty string if missing.
    """
    creds_dir = os.environ.get("CREDENTIALS_DIRECTORY", "")
    if creds_dir:
        path = Path(creds_dir) / name
        if path.exists():
            return path.read_text().strip()
    return os.environ.get(env_fallback, "")


@dataclass
class LLMConfig:
    # ── FREE (default) ──
    ollama_base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_fast_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_FAST_MODEL", "llama3.2:3b"))
    ollama_smart_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_SMART_MODEL", "mistral:7b"))
    ollama_embed_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    # ── OPTIONAL PAID (set to use) ──
    anthropic_api_key: str = field(default_factory=lambda: _read_credential("anthropic_key", "ANTHROPIC_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: _read_credential("openai_key", "OPENAI_API_KEY"))
    gemini_api_key: str = field(default_factory=lambda: _read_credential("gemini_key", "GEMINI_API_KEY"))

    # Model names when using paid APIs
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-5-20251001"
    opus_model: str = "claude-opus-4-5-20251001"
    gpt_fallback: str = "gpt-4o-mini"
    gemini_fallback: str = "gemini/gemini-1.5-flash"

    # Limits
    max_tokens_fast: int = 8192
    max_tokens_analysis: int = 8192
    daily_token_budget: int = int(os.environ.get("DAILY_TOKEN_BUDGET", "500000"))
    max_iterations: int = 20

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_any_paid(self) -> bool:
        return self.has_anthropic or self.has_openai or bool(self.gemini_api_key)


@dataclass
class VoiceConfig:
    # ── FREE STT (default: faster-whisper local) ──
    stt_provider: str = field(default_factory=lambda: os.environ.get("STT_PROVIDER", "whisper"))
    whisper_model: str = field(default_factory=lambda: os.environ.get("WHISPER_MODEL", "base"))
    whisper_device: str = field(default_factory=lambda: os.environ.get("WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = field(default_factory=lambda: os.environ.get("WHISPER_COMPUTE_TYPE", "int8"))
    whisper_language: str = field(default_factory=lambda: os.environ.get("WHISPER_LANGUAGE", "el"))

    # ── OPTIONAL PAID STT (Deepgram) ──
    deepgram_api_key: str = field(default_factory=lambda: _read_credential("deepgram_key", "DEEPGRAM_API_KEY"))
    deepgram_model: str = "nova-3"

    # ── FREE TTS (default: edge-tts — Microsoft free online, no key needed) ──
    tts_provider: str = field(default_factory=lambda: os.environ.get("TTS_PROVIDER", "edge"))
    edge_tts_voice: str = field(default_factory=lambda: os.environ.get("EDGE_TTS_VOICE", "el-GR-NestorasNeural"))
    kokoro_model: str = field(default_factory=lambda: os.environ.get("KOKORO_MODEL", "kokoro-v0_19.onnx"))
    kokoro_voice: str = field(default_factory=lambda: os.environ.get("KOKORO_VOICE", "af_sarah"))

    # ── OPTIONAL PAID TTS (ElevenLabs) ──
    elevenlabs_api_key: str = field(default_factory=lambda: _read_credential("elevenlabs_key", "ELEVENLABS_API_KEY"))
    elevenlabs_voice_id: str = field(default_factory=lambda: os.environ.get("ELEVENLABS_VOICE_ID", ""))
    elevenlabs_model: str = "eleven_flash_v2_5"

    # VAD settings
    vad_aggressiveness: int = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))
    barge_in_threshold_ms: int = int(os.environ.get("BARGE_IN_THRESHOLD_MS", "300"))
    sample_rate: int = 16000
    chunk_duration_ms: int = 30

    # Latency targets
    target_latency_ms: int = int(os.environ.get("TARGET_LATENCY_MS", "900"))

    @property
    def has_deepgram(self) -> bool:
        return bool(self.deepgram_api_key)

    @property
    def has_elevenlabs(self) -> bool:
        return bool(self.elevenlabs_api_key)


@dataclass
class MemoryConfig:
    # SQLite (free, v6 recommendation for single-user)
    db_path: Path = field(default_factory=lambda: DB_PATH)
    wal_checkpoint_interval_s: int = 3600
    session_compression_idle_s: int = 900
    max_working_memory_tokens: int = 180_000
    context_trim_target_pct: float = 0.80
    memory_retrieval_top_k: int = 5
    similarity_threshold: float = 0.75
    raw_log_retention_days: int = 7

    # Embedding model (free local via sentence-transformers)
    embed_provider: str = field(default_factory=lambda: os.environ.get("EMBED_PROVIDER", "sentence-transformers"))
    embed_model: str = field(default_factory=lambda: os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    embed_dim: int = int(os.environ.get("EMBED_DIM", "384"))


@dataclass
class SecurityConfig:
    zone: Literal["standard", "lab", "red"] = field(
        default_factory=lambda: os.environ.get("JARVIS_ZONE", "standard")  # type: ignore
    )
    lab_mode: bool = field(default_factory=lambda: os.environ.get("JARVIS_LAB_MODE", "false").lower() == "true")
    sandbox_container: str = "jarvis-sandbox"
    pii_detection: bool = True
    max_file_size_bytes: int = 50_000
    max_cat_bytes: int = 50_000

    # Zones
    green_zone_paths: list[str] = field(default_factory=lambda: [str(WORKSPACE_DIR), str(JARVIS_HOME)])
    yellow_zone_paths: list[str] = field(default_factory=lambda: [str(Path.home())])
    red_zone_paths: list[str] = field(default_factory=lambda: ["/etc", "/var", "/system", "/usr/lib"])

    # Blocked patterns
    blocked_commands: list[str] = field(default_factory=lambda: ["rm -rf /", "mkfs", "dd if="])
    blocked_chmod: list[str] = field(default_factory=lambda: [r"chmod\s+[0-9]*7[0-9]*", r"chmod\s+a\+"])


@dataclass
class TelegramConfig:
    bot_token: str = field(default_factory=lambda: _read_credential("telegram_token", "TELEGRAM_BOT_TOKEN"))
    allowed_user_id: int = field(default_factory=lambda: int(os.environ.get("TELEGRAM_USER_ID", "0")))
    emergency_bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_EMERGENCY_TOKEN", ""))

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token) and self.allowed_user_id != 0


@dataclass
class ServerConfig:
    host: str = field(default_factory=lambda: os.environ.get("JARVIS_HOST", "0.0.0.0"))
    port: int = int(os.environ.get("JARVIS_PORT", "8080"))
    dashboard_port: int = int(os.environ.get("JARVIS_DASHBOARD_PORT", "8081"))
    domain: str = field(default_factory=lambda: os.environ.get("JARVIS_DOMAIN", ""))
    use_https: bool = field(default_factory=lambda: bool(os.environ.get("JARVIS_DOMAIN", "")))
    max_body_size: int = 1_000_000
    ws_heartbeat_s: int = 30
    user_timezone: str = field(default_factory=lambda: os.environ.get("USER_TIMEZONE", "Europe/Athens"))

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.user_timezone)


@dataclass
class KAIROSConfig:
    enabled: bool = field(default_factory=lambda: os.environ.get("KAIROS_ENABLED", "true").lower() == "true")
    poll_interval_s: int = int(os.environ.get("KAIROS_INTERVAL", "300"))
    github_repos: list[str] = field(
        default_factory=lambda: [r for r in os.environ.get("KAIROS_GITHUB_REPOS", "").split(",") if r]
    )
    dream_idle_threshold_s: int = int(os.environ.get("DREAM_IDLE_THRESHOLD", "900"))


@dataclass
class JarvisConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    kairos: KAIROSConfig = field(default_factory=KAIROSConfig)

    def __post_init__(self):
        _ensure_dirs()

    def describe(self) -> str:
        """Human-readable summary of active providers."""
        llm_provider = "Anthropic API" if self.llm.has_anthropic else f"Ollama ({self.llm.ollama_fast_model})"
        stt_provider = "Deepgram Nova-3" if self.voice.has_deepgram else f"Whisper ({self.voice.whisper_model})"
        tts_provider = "ElevenLabs" if self.voice.has_elevenlabs else f"edge-tts ({self.voice.edge_tts_voice})"
        embed_provider = self.memory.embed_provider
        return (
            f"LLM: {llm_provider} | STT: {stt_provider} | TTS: {tts_provider} | "
            f"Embeddings: {embed_provider} | Zone: {self.security.zone}"
        )


# Singleton
_config: JarvisConfig | None = None


def get_config() -> JarvisConfig:
    global _config
    if _config is None:
        _config = JarvisConfig()
    return _config
