"""Microphone capture + simple energy VAD for live demos."""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000


def _resample_pcm(pcm: np.ndarray, from_sr: int, to_sr: int = SAMPLE_RATE) -> np.ndarray:
    if from_sr == to_sr or pcm.size == 0:
        return pcm
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(from_sr, to_sr)
    out = resample_poly(pcm.astype(np.float64), to_sr // g, from_sr // g)
    return np.clip(out, -32768, 32767).astype(np.int16)


def _pick_capture_rate(sd, device: int | None) -> int:
    """Pick a sample rate the device actually supports (ALC294 often rejects 16 kHz)."""
    try:
        dev = sd.query_devices(device, "input")
    except Exception:
        dev = sd.query_devices(kind="input")
    candidates: list[int] = []
    default_sr = int(float(dev.get("default_samplerate", 44100)))
    candidates.append(default_sr)
    for sr in (48000, 44100, 32000, 16000):
        if sr not in candidates:
            candidates.append(sr)
    for sr in candidates:
        try:
            sd.check_input_settings(device=device, samplerate=sr, channels=1, dtype="int16")
            return sr
        except Exception:
            continue
    return 44100


class MicCapture:
    def __init__(self, hz: float = 10.0, device: int | None = None, sample_rate: int = SAMPLE_RATE):
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError("pip install sounddevice") from e
        except OSError as e:
            raise OSError(
                "PortAudio 未安装，无法打开麦克风。请运行: sudo apt install libportaudio2 portaudio19-dev"
            ) from e
        self._sd = sd
        self.target_rate = sample_rate
        self.hz = hz
        self.device = device
        self.capture_rate = _pick_capture_rate(sd, device)
        self.capture_chunk = max(1, int(self.capture_rate / hz))
        self.chunk_samples = max(1, int(sample_rate / hz))

        if device is None:
            dev = sd.query_devices(kind="input")
            self.device_name = dev.get("name", "default")
            self.device_index = dev.get("index")
        else:
            dev = sd.query_devices(device)
            self.device_name = dev.get("name", str(device))
            self.device_index = device

    @staticmethod
    def list_devices() -> None:
        import sounddevice as sd

        print(sd.query_devices())
        print("\n默认输入设备:", sd.query_devices(kind="input"))

    def read_chunk(self) -> np.ndarray:
        rec = self._sd.rec(
            self.capture_chunk,
            samplerate=self.capture_rate,
            channels=1,
            dtype="int16",
            device=self.device,
        )
        self._sd.wait()
        pcm = rec[:, 0]
        return _resample_pcm(pcm, self.capture_rate, self.target_rate)


def make_vad(
    backend: str = "silero",
    *,
    threshold: float = 0.5,
    min_silence_ms: int = 550,
    speech_pad_ms: int = 100,
    audio_gain: float = 1.0,
    energy_threshold: float = 450.0,
):
    backend = backend.strip().lower()
    if backend == "silero":
        from social_vla.perception.silero_vad_adapter import PercySileroVAD

        return PercySileroVAD(
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            audio_gain=audio_gain,
        )
    if backend == "energy":
        return EnergyVAD(threshold=energy_threshold)
    raise ValueError(f"unknown vad backend: {backend}")


class EnergyVAD:
    """RMS threshold VAD with short hangover to avoid chopping."""

    def __init__(self, threshold: float = 450.0, hangover_ticks: int = 2):
        self.threshold = threshold
        self.hangover_ticks = hangover_ticks
        self._remain = 0
        self.last_rms = 0.0

    def feed(self, pcm: np.ndarray) -> bool:
        return self.update(pcm)

    def update(self, pcm: np.ndarray) -> bool:
        if pcm.size == 0:
            self.last_rms = 0.0
            if self._remain > 0:
                self._remain -= 1
            return self._remain > 0
        self.last_rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
        if self.last_rms >= self.threshold:
            self._remain = self.hangover_ticks
            return True
        if self._remain > 0:
            self._remain -= 1
            return True
        return False
