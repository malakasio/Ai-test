"""
Voice Activity Detection — in a separate OS process to bypass Python GIL.

v6 fixes:
- multiprocessing.Queue (not asyncio.Queue — cannot pickle across processes)
- Async bridge: mp_queue → asyncio.Queue via run_in_executor
- Barge-in state machine (not raw [USER INTERRUPTED] before STT completes)
- AEC (Acoustic Echo Cancellation) via speexdsp or earpiece recommendation
- start_new_session=True on all subprocesses (v6 fix: os.killpg scope)
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum, auto
from multiprocessing import Process, Queue as MPQueue
from typing import Optional


class VoiceState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    INTERRUPTED = auto()


def vad_worker(audio_in: MPQueue, interrupt_out: MPQueue, aggressiveness: int = 2):
    """
    Runs in separate OS process — no asyncio here.
    Detects continuous speech (barge-in) and signals interrupt.

    v6 fix: webrtcvad.Vad is CPU-bound, must NOT run in asyncio event loop.
    """
    try:
        import webrtcvad
    except ImportError:
        # Fallback: energy-based VAD
        while True:
            chunk = audio_in.get()
            if chunk is None:
                break
            # Simple RMS energy threshold
            import struct
            if len(chunk) >= 2:
                samples = struct.unpack(f"{len(chunk)//2}h", chunk)
                rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
                if rms > 1500:  # adjust threshold as needed
                    interrupt_out.put("INTERRUPT")
        return

    vad = webrtcvad.Vad(aggressiveness)
    speech_buffer: list[bool] = []
    SAMPLE_RATE = 16000
    FRAME_MS = 30
    FRAMES_FOR_INTERRUPT = 10  # 300ms of continuous speech

    while True:
        chunk = audio_in.get()
        if chunk is None:
            break  # poison pill

        try:
            is_speech = vad.is_speech(chunk, SAMPLE_RATE)
        except Exception:
            is_speech = False

        speech_buffer.append(is_speech)

        # Keep rolling window
        if len(speech_buffer) > FRAMES_FOR_INTERRUPT * 2:
            speech_buffer = speech_buffer[-FRAMES_FOR_INTERRUPT:]

        # 300ms of continuous speech → barge-in
        if len(speech_buffer) >= FRAMES_FOR_INTERRUPT and all(speech_buffer[-FRAMES_FOR_INTERRUPT:]):
            interrupt_out.put("INTERRUPT")
            speech_buffer.clear()


async def mp_queue_to_async(mp_q: MPQueue, async_q: asyncio.Queue):
    """Bridge: blocking multiprocessing.Queue → asyncio.Queue."""
    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, mp_q.get)
        await async_q.put(item)


class VADManager:
    """
    Manages the VAD process and provides async interrupt signals.
    """

    def __init__(self, aggressiveness: int = 2):
        self.aggressiveness = aggressiveness
        self._audio_queue: MPQueue = MPQueue()
        self._interrupt_mp_queue: MPQueue = MPQueue()
        self._interrupt_async_queue: asyncio.Queue = asyncio.Queue()
        self._process: Optional[Process] = None
        self._bridge_task: Optional[asyncio.Task] = None
        self.state = VoiceState.IDLE

    def start(self):
        """Start VAD in separate process."""
        self._process = Process(
            target=vad_worker,
            args=(self._audio_queue, self._interrupt_mp_queue, self.aggressiveness),
            daemon=True,
        )
        self._process.start()
        self._bridge_task = asyncio.create_task(
            mp_queue_to_async(self._interrupt_mp_queue, self._interrupt_async_queue)
        )

    def stop(self):
        """Send poison pill to VAD process."""
        self._audio_queue.put(None)
        if self._process:
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
        if self._bridge_task:
            self._bridge_task.cancel()

    def feed_audio(self, chunk: bytes):
        """Feed audio frame to VAD (non-blocking)."""
        if self._process and self._process.is_alive():
            self._audio_queue.put_nowait(chunk)

    async def wait_for_interrupt(self, timeout: float = 0.1) -> bool:
        """Check if a barge-in interrupt has been signaled."""
        try:
            await asyncio.wait_for(self._interrupt_async_queue.get(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def clear_interrupts(self):
        """Drain pending interrupts (after handling one)."""
        while not self._interrupt_async_queue.empty():
            try:
                self._interrupt_async_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
