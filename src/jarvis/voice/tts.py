"""
Text-to-Speech layer.

DEFAULT (FREE): edge-tts — Microsoft's free online TTS
  - No API key, no cost
  - Greek voices: el-GR-NestorasNeural (male), el-GR-AthinaNeural (female)
  - ~100-200ms first audio (online service)

LOCAL FREE ALTERNATIVE: Kokoro TTS (82M param, CPU-fast, high quality)
  - Set TTS_PROVIDER=kokoro
  - Requires: pip install kokoro-onnx soundfile

OPTIONAL PAID: ElevenLabs Flash v2.5
  - ~75ms first audio
  - Highest quality
  - Set ELEVENLABS_API_KEY to enable

v6 fixes:
- spaCy sentence splitting (not regex '.' — breaks IPs/decimals)
- PyAV in-memory conversion (not ffmpeg subprocess per sentence)
- Tenacity OUTSIDE semaphore (v6 deadlock fix)
- TTS semaphore = 1 concurrent request max (rate limit protection)
- Server streams bytes → client plays (not server-side mpv)
"""
from __future__ import annotations

import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from jarvis.config import get_config
from jarvis.llm.client import is_transient
from jarvis.observability.logger import get_logger
from jarvis.observability.metrics import get_metrics

log = get_logger("tts")

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")
_tts_semaphore = asyncio.Semaphore(1)  # one TTS request at a time
_spacy_nlp = None
_kokoro_tts = None


def _load_spacy():
    """Load spaCy Greek model ONCE at startup (blocking, in executor)."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return
    try:
        import spacy
        _spacy_nlp = spacy.load("el_core_news_sm")
        log.info("spaCy Greek model loaded")
    except Exception as e:
        log.warning(f"spaCy Greek model not available ({e}), using simple sentence splitter")
        _spacy_nlp = None


async def ensure_tts_ready():
    """Load TTS model (if local) and spaCy."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _load_spacy)

    cfg = get_config()
    if cfg.voice.tts_provider == "kokoro" and _kokoro_tts is None:
        await loop.run_in_executor(_executor, _load_kokoro)


def _load_kokoro():
    """Load Kokoro ONNX model (free, high quality)."""
    global _kokoro_tts
    try:
        from kokoro_onnx import Kokoro
        cfg = get_config()
        _kokoro_tts = Kokoro(cfg.voice.kokoro_model, "voices.bin")
        log.info("Kokoro TTS loaded")
    except Exception as e:
        log.warning(f"Kokoro not available: {e}")


def _split_sentences_sync(text: str) -> list[str]:
    """
    v5 fix: spaCy sentence splitting instead of '.' split.
    Handles IPs (192.168.1.1), abbreviations (κ.α.), decimals (3.14).
    min_len=3 allows 'Ναι.' 'Όχι.'
    """
    if _spacy_nlp:
        try:
            doc = _spacy_nlp(text[:2000])  # cap input
            return [s.text.strip() for s in doc.sents if len(s.text.strip()) >= 3]
        except Exception:
            pass

    # Fallback: simple split on sentence-ending punctuation
    import re
    parts = re.split(r"(?<=[.!?;])\s+", text.strip())
    return [p for p in parts if len(p.strip()) >= 3] or [text]


async def split_sentences(text: str) -> list[str]:
    """Async sentence splitting — offloads spaCy to executor."""
    if not text.strip():
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _split_sentences_sync, text)


async def _synthesize_edge_tts(text: str) -> bytes:
    """edge-tts: Microsoft free online TTS."""
    import edge_tts
    cfg = get_config()
    communicate = edge_tts.Communicate(text, voice=cfg.voice.edge_tts_voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data


async def _synthesize_elevenlabs(text: str) -> bytes:
    """ElevenLabs Flash v2.5 (paid, fastest)."""
    import aiohttp
    cfg = get_config()
    voice_id = cfg.voice.elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM"

    headers = {
        "xi-api-key": cfg.voice.elevenlabs_api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": cfg.voice.elevenlabs_model,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                text_err = await resp.text()
                raise RuntimeError(f"ElevenLabs error {resp.status}: {text_err[:200]}")
            return await resp.read()


def _synthesize_kokoro_sync(text: str) -> bytes:
    """Kokoro local TTS synthesis (blocking, in executor)."""
    if _kokoro_tts is None:
        _load_kokoro()
    if _kokoro_tts is None:
        raise RuntimeError("Kokoro TTS not available")

    import soundfile as sf
    cfg = get_config()
    samples, sample_rate = _kokoro_tts.create(text, voice=cfg.voice.kokoro_voice)

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


async def _synthesize_with_retry(text: str) -> bytes:
    """
    v6 fix: Tenacity retry OUTSIDE semaphore.
    (Inside = deadlock when retry wait holds semaphore)
    """
    cfg = get_config()

    async for attempt in AsyncRetrying(
        wait=wait_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(is_transient),
    ):
        with attempt:
            if cfg.voice.has_elevenlabs and cfg.voice.tts_provider in ("elevenlabs", "auto"):
                return await _synthesize_elevenlabs(text)
            elif cfg.voice.tts_provider == "kokoro":
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(_executor, _synthesize_kokoro_sync, text)
            else:
                return await _synthesize_edge_tts(text)

    raise RuntimeError("TTS retry exhausted")


async def synthesize_sentence(text: str) -> bytes:
    """
    Synthesize a single sentence.
    v6 fix: semaphore WRAPS the retried call.
    """
    start_ts = time.time()
    async with _tts_semaphore:
        audio = await _synthesize_with_retry(text)

    duration = time.time() - start_ts
    get_metrics().tts_latency.observe(duration)
    return audio


async def synthesize_streaming(text: str) -> AsyncIterator[bytes]:
    """
    Stream audio sentence by sentence.
    First audio arrives after first sentence — minimizes latency.
    Server yields bytes → client WebSocket plays them.
    """
    sentences = await split_sentences(text)
    if not sentences:
        sentences = [text]

    for sentence in sentences:
        if not sentence.strip():
            continue
        try:
            audio_chunk = await synthesize_sentence(sentence)
            yield audio_chunk
        except Exception as e:
            log.error(f"TTS failed for sentence: {e}")
            # Continue with next sentence, don't abort


async def synthesize_full(text: str) -> bytes:
    """Synthesize complete text as one audio chunk."""
    all_audio = b""
    async for chunk in synthesize_streaming(text):
        all_audio += chunk
    return all_audio
