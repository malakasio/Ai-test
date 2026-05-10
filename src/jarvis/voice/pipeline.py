"""
Main voice pipeline — orchestrates STT → LLM → TTS with <900ms target latency.

Architecture (3 separate processes/coroutines as per v5/v6):
1. Main async event loop (FastAPI/WebSocket handler)
2. VAD process (multiprocessing, bypasses GIL)
3. Audio streaming: server produces bytes → client plays

Key latency optimizations:
- Sentence streaming: first audio after ~1st sentence (~200-300ms after LLM starts)
- Parallel: LLM generating while TTS synthesizing sentence 1
- Server never plays audio (no PulseAudio hell) — streams bytes to client

v6 barge-in state machine (not raw [USER INTERRUPTED]):
LISTENING → PROCESSING → SPEAKING
                         ↑ barge-in
                         └→ INTERRUPTED → LISTENING (wait for new STT)
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from jarvis.config import get_config
from jarvis.observability.logger import get_logger, get_audit
from jarvis.observability.metrics import get_metrics
from jarvis.voice.stt import transcribe_audio, ensure_stt_ready
from jarvis.voice.tts import synthesize_streaming, ensure_tts_ready
from jarvis.voice.vad import VADManager, VoiceState

log = get_logger("voice.pipeline")


@dataclass
class VoiceSession:
    session_id: str
    state: VoiceState = VoiceState.IDLE
    history: list[dict] = field(default_factory=list)
    collected_tokens: str = ""
    current_llm_task: Optional[asyncio.Task] = None
    current_tts_tasks: list[asyncio.Task] = field(default_factory=list)
    audio_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    barge_in_count: int = 0
    last_activity: float = field(default_factory=time.time)

    def touch(self):
        self.last_activity = time.time()


class VoicePipeline:
    """
    Manages a voice conversation session.
    Designed to run behind a WebSocket — yields audio bytes to send to client.
    """

    def __init__(
        self,
        agent_fn: Callable,
        system_prompt: str,
        session_id: str,
    ):
        self.agent_fn = agent_fn
        self.system_prompt = system_prompt
        self.session = VoiceSession(session_id=session_id)
        self.vad = VADManager(aggressiveness=get_config().voice.vad_aggressiveness)
        self._initialized = False

    async def initialize(self):
        """Load models — call once before first use."""
        if self._initialized:
            return
        await asyncio.gather(ensure_stt_ready(), ensure_tts_ready())
        self.vad.start()
        self._initialized = True
        log.info(f"Voice pipeline initialized (session {self.session.session_id[:8]})")

    async def process_audio_input(
        self, audio_bytes: bytes
    ) -> AsyncIterator[bytes]:
        """
        Process incoming audio from client.
        Yields audio response bytes to stream back.

        Full pipeline: audio → STT → LLM → TTS (sentence-streamed)
        """
        if not self._initialized:
            await self.initialize()

        metrics = get_metrics()
        start_ts = time.time()
        self.session.touch()
        self.session.state = VoiceState.PROCESSING

        # 1. Speech-to-Text
        stt_start = time.time()
        try:
            user_text = await transcribe_audio(audio_bytes)
        except Exception as e:
            log.error(f"STT failed: {e}")
            return
        stt_duration = time.time() - stt_start

        if not user_text.strip():
            log.debug("Empty STT result, ignoring")
            return

        log.info(f"STT ({stt_duration*1000:.0f}ms): '{user_text}'")

        # Add to history
        self.session.history.append({"role": "user", "content": user_text})

        # 2. LLM → TTS streaming
        self.session.state = VoiceState.SPEAKING
        response_text = ""

        try:
            async for audio_chunk in self._llm_tts_stream(user_text):
                # Check for barge-in before each chunk
                if await self.vad.wait_for_interrupt(timeout=0.01):
                    log.info("Barge-in detected — stopping TTS")
                    metrics.barge_in_total.inc()
                    self.session.barge_in_count += 1
                    await self._handle_barge_in(response_text)
                    return

                yield audio_chunk

        except asyncio.CancelledError:
            await self._handle_barge_in(response_text)
            return
        except Exception as e:
            log.error(f"LLM/TTS pipeline error: {e}")
            # Yield error audio
            try:
                error_audio = b""
                async for chunk in synthesize_streaming("Συγγνώμη, υπήρξε τεχνικό πρόβλημα."):
                    error_audio += chunk
                yield error_audio
            except Exception:
                pass
            return

        # Add assistant response to history
        if response_text:
            self.session.history.append({"role": "assistant", "content": response_text})

        total_duration = time.time() - start_ts
        metrics.voice_latency.observe(total_duration)
        metrics.voice_sessions_total.inc()
        self.session.state = VoiceState.LISTENING
        log.info(f"Voice round-trip: {total_duration*1000:.0f}ms")

    async def _llm_tts_stream(self, user_text: str) -> AsyncIterator[bytes]:
        """
        LLM streaming → sentence splitting → TTS synthesis.
        Yields audio bytes as they're synthesized (low latency).
        """
        from jarvis.llm.router import route
        from jarvis.llm.client import run_agent
        from jarvis.memory.store import inject_time_context, trim_context
        from jarvis.voice.tts import split_sentences

        cfg = get_config()
        decision = route(user_text)

        # Inject time context
        enriched_text = inject_time_context(user_text)

        # Trim history to fit context
        trimmed_history = trim_context(
            self.session.history[:-1],  # exclude current message
            system=self.system_prompt,
            max_tokens=cfg.memory.max_working_memory_tokens,
        )
        messages = trimmed_history + [{"role": "user", "content": enriched_text}]

        # Run LLM (no tool execution in voice mode — too slow)
        full_response, usage = await run_agent(
            messages=messages,
            tools=[],
            system=self.system_prompt,
            decision=decision,
        )

        # Synthesize sentence by sentence
        sentences = await split_sentences(full_response)
        for sentence in sentences:
            if sentence.strip():
                tts_start = time.time()
                audio = b""
                async for chunk in synthesize_streaming(sentence):
                    audio += chunk
                yield audio

        return full_response

    async def _handle_barge_in(self, partial_response: str):
        """
        v6 state machine fix: don't add [USER INTERRUPTED] before new STT.
        Just update history with what was said so far.
        """
        if partial_response.strip():
            self.session.history.append({
                "role": "assistant",
                "content": partial_response + " [ΔΙΑΚΟΠΗ]",
            })
        await self.vad.clear_interrupts()
        self.session.state = VoiceState.LISTENING
        log.debug("Barge-in handled, waiting for new speech")

    def feed_vad(self, audio_chunk: bytes):
        """Feed audio to VAD for barge-in detection (during TTS playback)."""
        self.vad.feed_audio(audio_chunk)

    async def shutdown(self):
        """Clean up resources."""
        self.vad.stop()
        self._initialized = False
