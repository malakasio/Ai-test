"""
Speech-to-Text layer.

DEFAULT (FREE): faster-whisper (local OpenAI Whisper)
  - No API key needed
  - Runs on CPU (or CUDA if available)
  - Models: tiny/base/small/medium/large-v3
  - Greek language support

OPTIONAL PAID: Deepgram Nova-3
  - Faster (150-300ms vs 300-500ms)
  - Semantic end-of-speech detection
  - Set DEEPGRAM_API_KEY to enable

v6 fixes:
- CPU-bound transcription offloaded to ThreadPoolExecutor
- spaCy Greek model loaded ONCE at startup
- min sentence len = 3 (not 10, allows 'Ναι.' 'Όχι.')
- Audio format: PCM 16kHz 16-bit mono (Whisper requirement)
"""
from __future__ import annotations

import asyncio
import io
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from jarvis.config import get_config
from jarvis.observability.logger import get_logger
from jarvis.observability.metrics import get_metrics

log = get_logger("stt")

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt")
_whisper_model = None
_deepgram_client = None


def _load_whisper():
    """Load faster-whisper model (blocking, runs in executor)."""
    global _whisper_model
    if _whisper_model is not None:
        return

    cfg = get_config()
    from faster_whisper import WhisperModel

    log.info(f"Loading Whisper model '{cfg.voice.whisper_model}' on {cfg.voice.whisper_device}...")
    _whisper_model = WhisperModel(
        cfg.voice.whisper_model,
        device=cfg.voice.whisper_device,
        compute_type=cfg.voice.whisper_compute_type,
    )
    log.info("Whisper model loaded")


async def ensure_stt_ready():
    """Load STT model in executor so it doesn't block event loop."""
    cfg = get_config()
    if cfg.voice.stt_provider == "whisper" and _whisper_model is None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _load_whisper)


def _transcribe_sync(audio_bytes: bytes, language: str) -> str:
    """Run Whisper transcription synchronously (in executor)."""
    if _whisper_model is None:
        _load_whisper()

    import numpy as np

    # Convert PCM bytes to float32 numpy array
    if len(audio_bytes) % 2 != 0:
        audio_bytes = audio_bytes[:-1]
    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    segments, info = _whisper_model.transcribe(
        pcm,
        language=language or None,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text


async def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes to text.
    Automatically selects provider based on configuration.
    """
    cfg = get_config()
    start_ts = time.time()
    metrics = get_metrics()

    if cfg.voice.has_deepgram and cfg.voice.stt_provider in ("deepgram", "auto"):
        text = await _transcribe_deepgram(audio_bytes)
    else:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            _executor, _transcribe_sync, audio_bytes, cfg.voice.whisper_language
        )

    duration = time.time() - start_ts
    metrics.stt_latency.observe(duration)
    log.debug(f"STT ({duration*1000:.0f}ms): '{text[:80]}'")
    return text


async def _transcribe_deepgram(audio_bytes: bytes) -> str:
    """Deepgram Nova-3 transcription (paid, faster)."""
    global _deepgram_client
    cfg = get_config()

    if _deepgram_client is None:
        from deepgram import DeepgramClient, PrerecordedOptions
        _deepgram_client = DeepgramClient(cfg.voice.deepgram_api_key)

    from deepgram import PrerecordedOptions
    options = PrerecordedOptions(
        model=cfg.voice.deepgram_model,
        language="el",
        smart_format=True,
        utterances=True,
    )

    import aiohttp
    headers = {
        "Authorization": f"Token {cfg.voice.deepgram_api_key}",
        "Content-Type": "audio/wav",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://api.deepgram.com/v1/listen?model={cfg.voice.deepgram_model}&language=el&smart_format=true",
            headers=headers,
            data=audio_bytes,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            data = await resp.json()

    transcript = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
    return transcript.strip()
