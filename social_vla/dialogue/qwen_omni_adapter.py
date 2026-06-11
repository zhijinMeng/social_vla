"""Qwen-Omni dialogue adapter for SocialVLA-Engage Layer A.

Bridges FrameTick trigger events to the DashScope Qwen-Omni Realtime API.
Handles robot-initiated greetings and user-turn replies via streaming audio.
"""
from __future__ import annotations

import base64
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from social_vla.types import FrameTick, PerPersonScores

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Force INFO level

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
    logger.warning("sounddevice not installed — audio playback disabled")

try:
    import dashscope
    from dashscope.audio.qwen_omni import OmniRealtimeCallback, OmniRealtimeConversation
    _DASHSCOPE_AVAILABLE = True
except ImportError:
    _DASHSCOPE_AVAILABLE = False
    logger.warning("dashscope not installed — QwenOmniAdapter will run in mock mode")


@dataclass
class QwenOmniConfig:
    api_key: str = ""
    model: str = "qwen3.5-omni-flash-realtime"
    url: str = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
    voice: str = "Serena"  # Serena / Cherry / Ethan / longjing / longhua etc.
    system_prompt: str = (
        "你是中国展会现场的服务机器人。"
        "必须用简体中文口语回复，每次不超过两句话，语气简洁友好。"
        "禁止主动使用印尼语、英语或其他语言；仅当用户明确要求换语言时才切换。"
    )
    on_reply: Callable[[str], None] | None = None
    on_partial: Callable[[str], None] | None = None
    mock: bool = False
    # Wait after hardware playback before arming mic (reduces speaker echo)
    echo_guard_ms: float = 800.0
    # Retain this much mic audio while gated so utterance onsets are not clipped
    pcm_preroll_s: float = 1.5
    # Include audio slightly before VAD onset (VAD lags true speech start)
    preroll_lead_s: float = 0.25
    sample_rate: int = 16000
    # Ignore commits shorter than this (16kHz mono int16 ≈ 32000 bytes/s)
    min_user_audio_bytes: int = 9600  # ~0.3s


class _PcmPreroll:
    """Ring buffer of (timestamp, pcm) for audio captured while turn is gated."""

    def __init__(self, max_samples: int):
        self._max_samples = max_samples
        self._chunks: deque[tuple[float, np.ndarray]] = deque()
        self._total = 0

    def push(self, timestamp: float, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        chunk = pcm.astype(np.int16, copy=False)
        self._chunks.append((timestamp, chunk))
        self._total += chunk.size
        while self._total > self._max_samples and self._chunks:
            _, old = self._chunks.popleft()
            self._total -= old.size

    def flush_since(self, since_ts: float) -> list[np.ndarray]:
        return [pcm for ts, pcm in self._chunks if ts >= since_ts]

    def clear(self) -> None:
        self._chunks.clear()
        self._total = 0


class _AudioPlayer:
    """Streams PCM chunks from Qwen-Omni to the speaker in a background thread."""

    SAMPLE_RATE = 16000
    CHANNELS = 1
    _DONE = object()

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._playing = threading.Event()
        self.on_done: Callable[[], None] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="qwen-audio-play")
        self._thread.start()

    def push(self, raw_b64: str) -> None:
        self._playing.set()
        self._q.put_nowait(base64.b64decode(raw_b64))

    def mark_done(self) -> None:
        self._q.put_nowait(self._DONE)

    def is_playing(self) -> bool:
        return self._playing.is_set()

    def stop(self) -> None:
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        # daemon thread — avoid join() blocking on Ctrl+C (sounddevice + PyAudio)

    def _run(self) -> None:
        if not _SD_AVAILABLE:
            while True:
                item = self._q.get()
                if item is None:
                    break
                if item is self._DONE:
                    self._playing.clear()
                    if self.on_done:
                        self.on_done()
            return

        try:
            stream = sd.OutputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype="int16",
            )
            stream.start()
            while True:
                item = self._q.get()
                if item is None:
                    break
                if item is self._DONE:
                    self._playing.clear()
                    if self.on_done:
                        self.on_done()
                    continue
                pcm = np.frombuffer(item, dtype=np.int16)
                stream.write(pcm)
            stream.stop()
            stream.close()
        except Exception as exc:
            logger.warning("[QwenOmni] audio playback error: %s", exc)


class _RealtimeCallback:
    def __init__(self, config: QwenOmniConfig, player: _AudioPlayer):
        self._config = config
        self._player = player
        self._partial: list[str] = []

    def on_event(self, response: dict) -> None:
        event_type = response.get("type", "")

        if event_type == "response.audio.delta":
            delta = response.get("delta", "")
            if delta:
                self._player.push(delta)

        elif event_type == "response.audio_transcript.delta":
            chunk = response.get("delta", "")
            self._partial.append(chunk)
            if self._config.on_partial:
                self._config.on_partial(chunk)

        elif event_type == "response.audio_transcript.done":
            full_text = "".join(self._partial)
            self._partial.clear()
            logger.info("[QwenOmni] reply: %s", full_text)
            if self._config.on_reply:
                self._config.on_reply(full_text)

        elif event_type == "response.done":
            self._player.mark_done()

        elif event_type == "error":
            logger.error("[QwenOmni] API error: %s", response)


class QwenOmniAdapter:
    """Connects Layer A trigger decisions to Qwen-Omni Realtime API."""

    def __init__(self, config: QwenOmniConfig | None = None):
        self.config = config or QwenOmniConfig()
        self._conv: OmniRealtimeConversation | None = None
        self._connected = False
        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self._stream_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._in_turn = False
        self._listening = True
        self._vad_active = False
        self._utterance_start_ts: float | None = None
        self._listening_armed_at: float = 0.0
        self._arm_generation = 0
        self._turn_audio_bytes = 0  # Track bytes enqueued in current turn
        self._preroll = _PcmPreroll(
            max_samples=int(self.config.pcm_preroll_s * self.config.sample_rate)
        )
        self._player = _AudioPlayer()

        if not self.config.mock and not _DASHSCOPE_AVAILABLE:
            logger.warning("dashscope unavailable, forcing mock=True")
            self.config.mock = True

    def start(self) -> None:
        if self.config.mock:
            logger.info("[QwenOmni] mock mode — no API connection")
            self._connected = True
            return

        api_key = self.config.api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set and api_key not provided")

        dashscope.api_key = api_key

        cb = _RealtimeCallback(self.config, self._player)
        self._player.on_done = self._on_robot_speech_done

        class _CBAdapter(OmniRealtimeCallback):
            def on_event(self_, response: dict) -> None:  # noqa: N805
                cb.on_event(response)

        self._conv = OmniRealtimeConversation(
            model=self.config.model,
            callback=_CBAdapter(),
            url=self.config.url,
        )
        self._conv.connect()
        self._connected = True

        from dashscope.audio.qwen_omni.omni_realtime import MultiModality, AudioFormat
        self._conv.update_session(
            output_modalities=[MultiModality.TEXT, MultiModality.AUDIO],
            voice=self.config.voice,
            input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            output_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            enable_turn_detection=False,
            instructions=self.config.system_prompt,
        )

        self._stop_event.clear()
        self._stream_thread = threading.Thread(
            target=self._audio_stream_loop, daemon=True, name="qwen-audio-stream"
        )
        self._stream_thread.start()
        logger.info("[QwenOmni] connected to %s", self.config.model)

    def stop(self) -> None:
        self._stop_event.set()
        self._arm_generation += 1
        if self._conv and self._connected:
            try:
                self._conv.close()
            except Exception:
                pass
        try:
            self._player.stop()
        except Exception:
            pass
        self._connected = False
        logger.info("[QwenOmni] disconnected")

    def gate_vad(self, raw_active: bool) -> bool:
        """Suppress mic VAD while robot speaks or during echo guard.

        Layer A / TriggerRouter must use the gated signal, not raw Silero output.
        """
        with self._lock:
            playing = self._player.is_playing()
            in_turn = self._in_turn
            listening = self._listening
            result = raw_active
            if playing:
                result = False
            elif in_turn:
                result = raw_active
            elif not listening:
                result = False
            logger.info("[QwenOmni] gate_vad: raw=%s → %s (playing=%s in_turn=%s listening=%s)",
                         raw_active, result, playing, in_turn, listening)
            return result

    def is_suppressing_mic(self) -> bool:
        """True when user speech should not be collected or trigger turns."""
        with self._lock:
            return self._player.is_playing() or not self._listening

    def on_frame_tick(
        self,
        tick: FrameTick,
        frame_bgr: np.ndarray | None = None,
        pcm_chunk: np.ndarray | None = None,
    ) -> str | None:
        if not self._connected:
            return None

        trigger = tick.trigger
        if trigger is None:
            return None

        if trigger == "robot_initiated":
            return self._handle_robot_initiated(tick)

        if trigger in ("user_initiated", "turn"):
            return self._handle_user_turn(tick)

        return None

    def _handle_robot_initiated(self, tick: FrameTick) -> str | None:
        score = self._best_score(tick)
        engagement_pct = int((score.engagement if score else 0.5) * 100)

        prompt = (
            f"你好！有人停在你面前，感兴趣度约 {engagement_pct}%。"
            "请用一句话自然地打个招呼，邀请对方了解你。"
        )

        logger.info("[QwenOmni] robot_initiated → sending prompt")

        with self._lock:
            self._listening = False
            self._in_turn = False
            self._vad_active = False
            self._utterance_start_ts = None
        self._preroll.clear()
        self._arm_generation += 1

        if self.config.mock:
            reply = "[MOCK] 你好！我是展会服务机器人，很高兴认识你～"
            if self.config.on_reply:
                self.config.on_reply(reply)
            self._arm_listening_after_guard()
            return reply

        self._send_text_prompt(prompt)
        return None

    def _handle_user_turn(self, tick: FrameTick) -> str | None:
        with self._lock:
            in_turn = self._in_turn
        if not in_turn:
            logger.debug(
                "[QwenOmni] %s ignored — no user turn was started (echo/gated)",
                tick.trigger,
            )
            return None

        logger.info("[QwenOmni] %s → committing user turn", tick.trigger)

        if self.config.mock:
            reply = "[MOCK] 我听到你说话了，请继续。"
            if self.config.on_reply:
                self.config.on_reply(reply)
            with self._lock:
                self._in_turn = False
            self._arm_listening_after_guard()
            return reply

        with self._lock:
            self._in_turn = False
            self._listening = False
            self._vad_active = False
            self._utterance_start_ts = None
        self._preroll.clear()

        nbytes = self._flush_audio_queue()
        if nbytes < self.config.min_user_audio_bytes:
            logger.warning(
                "[QwenOmni] user audio too short (%d bytes < %d), skip commit",
                nbytes,
                self.config.min_user_audio_bytes,
            )
            self._arm_listening_after_guard()
            return None

        logger.info("[QwenOmni] committing %d bytes of user audio", nbytes)
        if self._conv:
            self._conv.commit()
            self._conv.create_response(instructions=self._user_reply_instructions())
        return None

    def _on_robot_speech_done(self) -> None:
        self._arm_listening_after_guard()

    def _arm_listening_after_guard(self) -> None:
        guard_s = self.config.echo_guard_ms / 1000.0
        self._arm_generation += 1
        gen = self._arm_generation

        def _arm() -> None:
            if guard_s > 0:
                time.sleep(guard_s)
            if gen != self._arm_generation:
                return
            armed_at = time.time()
            late_start = False
            with self._lock:
                self._listening = True
                self._listening_armed_at = armed_at
                # Only catch speech that started AFTER arming (not echo during robot tail)
                if (
                    self._vad_active
                    and self._utterance_start_ts is not None
                    and self._utterance_start_ts >= armed_at - self.config.preroll_lead_s
                ):
                    late_start = True
                else:
                    self._vad_active = False
                    self._utterance_start_ts = None
            if late_start:
                self.begin_user_turn()
            else:
                self._preroll.clear()
            logger.info(
                "[QwenOmni] listening armed (echo guard %.0fms) — %s",
                self.config.echo_guard_ms,
                "late turn started" if late_start else "waiting for user",
            )

        threading.Thread(target=_arm, daemon=True, name="qwen-echo-guard").start()

    def begin_user_turn(self) -> None:
        with self._lock:
            logger.info("[QwenOmni] begin_user_turn: listening=%s in_turn=%s", self._listening, self._in_turn)
            if not self._listening or self._in_turn:
                return
            self._in_turn = True
            self._listening = False
            self._turn_audio_bytes = 0  # Reset counter for new turn
            lead = self.config.preroll_lead_s
            if self._utterance_start_ts is not None:
                since_ts = self._utterance_start_ts - lead
            else:
                since_ts = time.time() - min(lead, self.config.pcm_preroll_s)

        preroll = self._preroll.flush_since(since_ts)
        if not self.config.mock and self._conv:
            try:
                self._conv.clear_input_audio_buffer()
            except Exception:
                pass
        for pcm in preroll:
            self._enqueue_audio(pcm)
        if preroll:
            logger.debug("[QwenOmni] flushed %d preroll chunk(s) from %.2fs", len(preroll), since_ts)

    def note_vad(self, active: bool) -> None:
        """Track gated VAD edges for preroll timing (called from layer_a.set_vad)."""
        now = time.time()
        with self._lock:
            if active and not self._vad_active:
                self._utterance_start_ts = now
            if not active:
                self._utterance_start_ts = None
            self._vad_active = active

    def push_user_pcm(self, pcm: np.ndarray, timestamp: float | None = None) -> None:
        if pcm.size == 0:
            return
        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            in_turn = self._in_turn
            suppress = self._player.is_playing() or (not self._in_turn and not self._listening)

        if in_turn and not suppress:
            self._enqueue_audio(pcm)
            logger.debug("[QwenOmni] enqueued %d samples", pcm.size)
        elif not suppress:
            self._preroll.push(ts, pcm)
        # while robot speaks / echo guard: drop mic audio entirely

    def push_pcm(self, pcm: np.ndarray) -> None:
        self._enqueue_audio(pcm)

    def _enqueue_audio(self, pcm: np.ndarray) -> None:
        if pcm.dtype != np.int16:
            pcm = (pcm * 32767).astype(np.int16)
        raw = pcm.tobytes()
        self._audio_q.put_nowait(raw)
        # Track bytes enqueued during turn
        with self._lock:
            if self._in_turn:
                self._turn_audio_bytes += len(raw)

    def _flush_audio_queue(self) -> int:
        """Wait for background thread to send queued audio, return bytes enqueued in this turn."""
        # The _audio_stream_loop background thread is already sending audio continuously.
        # We track how much was enqueued during the turn and wait for the queue to drain.
        qsize = self._audio_q.qsize()

        # Give the background thread time to drain the queue (50ms per chunk)
        if qsize > 0:
            time.sleep(qsize * 0.05)

        # Return the actual bytes enqueued during this turn
        with self._lock:
            nbytes = self._turn_audio_bytes

        logger.info("[QwenOmni] flushed %d bytes from turn (qsize was %d)", nbytes, qsize)
        return nbytes

    def _user_reply_instructions(self) -> str:
        return (
            f"{self.config.system_prompt} "
            "用户刚说完一句话。请听懂其内容后用简体中文直接回答；"
            "不要重复欢迎语，不要再次打招呼。"
        )

    def _audio_stream_loop(self) -> None:
        total_sent = 0
        while not self._stop_event.is_set():
            try:
                raw = self._audio_q.get(timeout=0.05)
                if self._conv:
                    encoded = base64.b64encode(raw).decode()
                    self._conv.append_audio(encoded)
                    total_sent += len(raw)
                    if total_sent % 16000 == 0:  # Log every ~0.5s of audio
                        logger.info("[QwenOmni] streamed %d bytes total to API", total_sent)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.warning("[QwenOmni] audio stream error: %s", exc)

    def _send_text_prompt(self, text: str) -> None:
        if self._conv:
            try:
                self._conv.create_item({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                })
                self._conv.create_response(instructions=self.config.system_prompt)
            except Exception as exc:
                logger.error("[QwenOmni] failed to send text prompt: %s", exc)

    def _best_score(self, tick: FrameTick) -> PerPersonScores | None:
        if not tick.scores:
            return None
        if tick.best_target_id is not None:
            for s in tick.scores:
                if s.track_id == tick.best_target_id:
                    return s
        return max(tick.scores, key=lambda s: s.engagement)
