"""Reuse PERCY live_dialogue Silero VAD (same as record_dialogue_session.sh)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PERCY_VAD_SCRIPTS = (
    Path(__file__).resolve().parents[3] / "percy_ws" / "src" / "percy_dialogue" / "scripts"
)


def _import_silero_stream():
    scripts = str(_PERCY_VAD_SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from silero_vad_stream import SileroVadStream  # type: ignore

    return SileroVadStream


def amplify_pcm_int16(pcm: np.ndarray, gain: float) -> np.ndarray:
    if gain == 1.0 or pcm.size == 0:
        return pcm
    boosted = np.clip(pcm.astype(np.float32) * gain, -32768, 32767)
    return boosted.astype(np.int16)


class PercySileroVAD:
    """
    Streaming VAD aligned with percy_dialogue live_dialogue.py (vad_backend=silero).

    feed() returns whether user is currently in an utterance (between start/end events).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 550,
        speech_pad_ms: int = 100,
        audio_gain: float = 1.0,
    ):
        SileroVadStream = _import_silero_stream()
        self._stream = SileroVadStream(
            sample_rate=16000,
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.audio_gain = audio_gain
        self.speech_active = False
        self.last_event: str | None = None
        self.last_rms = 0.0

    def reset(self) -> None:
        self._stream.reset_states()
        self.speech_active = False
        self.last_event = None

    def feed(self, pcm: np.ndarray) -> bool:
        if pcm.size:
            self.last_rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
        else:
            self.last_rms = 0.0
        pcm = amplify_pcm_int16(pcm, self.audio_gain)
        for event in self._stream.feed(pcm.tobytes()):
            self.last_event = event
            if event == "start":
                self.speech_active = True
            elif event == "end":
                self.speech_active = False
        return self.speech_active
