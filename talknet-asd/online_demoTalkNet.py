#!/usr/bin/env python3
"""Online-style TalkNet demo.

Modes:
1) File mode (existing behavior): read a video file and output boxed video.
2) Camera mode: webcam + microphone realtime detection and live display.

Examples:
  # File mode
  source .venv-talknet/bin/activate
  python online_demoTalkNet.py --videoFolder demo --videoName 111 \
    --windowSec 0.4 --inferStrideSec 0.2 --scoreThres 0.2 --onlyTopSpeaker

  # Camera realtime mode
  source .venv-talknet/bin/activate
  python online_demoTalkNet.py --camera --cameraId 0 --showWindow \
    --windowSec 0.4 --inferStrideSec 0.2 --scoreThres 0.2 --onlyTopSpeaker
"""

import argparse
import json
import collections
import glob
import math
import os
import socket
import subprocess
import sys
import threading
import time
import warnings
import importlib
import wave
from http import server
from socketserver import ThreadingMixIn
from pathlib import Path
import tempfile

import cv2
import numpy as np
import python_speech_features
import torch
from scipy.io import wavfile

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet

# Reduce noisy third-party warnings in realtime logs.
warnings.filterwarnings("ignore", message=".*Resize.cpp.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")


AUDIO_SR = 16000
TALKNCE_VGGISH_INPUT = None


def _pick_sounddevice_capture_rate(sd, device):
    """Pick a capture rate the device supports (many laptop mics reject 16 kHz)."""
    try:
        dev = sd.query_devices(device, "input")
    except Exception:
        dev = sd.query_devices(kind="input")
    candidates = []
    default_sr = int(float(dev.get("default_samplerate", 44100)))
    candidates.append(default_sr)
    for sr in (48000, 44100, 32000, AUDIO_SR):
        if sr not in candidates:
            candidates.append(sr)
    for sr in candidates:
        try:
            sd.check_input_settings(device=device, samplerate=sr, channels=1, dtype="float32")
            return sr
        except Exception:
            continue
    return 44100


def _resample_audio_float(mono, from_sr, to_sr):
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    if from_sr == to_sr or mono.size == 0:
        return mono
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(from_sr), int(to_sr))
    return resample_poly(mono, int(to_sr) // g, int(from_sr) // g).astype(np.float32)


class SplitChannelWavRecorder:
    def __init__(self, out_dir, sr, channels, prefix="mic", save_denoised_mono=True):
        self.out_dir = Path(out_dir)
        self.sr = int(sr)
        self.channels = max(1, int(channels))
        self.prefix = str(prefix).strip() or "mic"
        self.save_denoised_mono = bool(save_denoised_mono)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.raw_wavs = []
        self.denoised_mono_wav = None

        for ch in range(self.channels):
            path = self.out_dir / f"{self.prefix}_raw_ch{ch}.wav"
            wf = wave.open(str(path), "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            self.raw_wavs.append(wf)

        if self.save_denoised_mono:
            path = self.out_dir / f"{self.prefix}_denoised_mono.wav"
            wf = wave.open(str(path), "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            self.denoised_mono_wav = wf

    def write_raw_channels(self, pcm):
        if pcm.ndim != 2:
            return
        pcm16 = np.clip(np.asarray(pcm), -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16, copy=False)
        for ch, wf in enumerate(self.raw_wavs):
            wf.writeframes(pcm16[:, ch].tobytes())

    def write_denoised_mono(self, mono):
        if self.denoised_mono_wav is None:
            return
        mono16 = np.clip(np.asarray(mono).reshape(-1), -1.0, 1.0)
        mono16 = (mono16 * 32767.0).astype(np.int16, copy=False)
        self.denoised_mono_wav.writeframes(mono16.tobytes())

    def close(self):
        for wf in self.raw_wavs:
            try:
                wf.close()
            except Exception:
                pass
        self.raw_wavs = []
        if self.denoised_mono_wav is not None:
            try:
                self.denoised_mono_wav.close()
            except Exception:
                pass
            self.denoised_mono_wav = None


class MonoWavRecorder:
    def __init__(self, path, sr):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sr = int(sr)
        self.wf = wave.open(str(self.path), "wb")
        self.wf.setnchannels(1)
        self.wf.setsampwidth(2)
        self.wf.setframerate(self.sr)

    def write(self, mono):
        mono16 = np.clip(np.asarray(mono).reshape(-1), -1.0, 1.0)
        mono16 = (mono16 * 32767.0).astype(np.int16, copy=False)
        self.wf.writeframes(mono16.tobytes())

    def close(self):
        if self.wf is not None:
            try:
                self.wf.close()
            except Exception:
                pass
            self.wf = None


class FanNoiseReducer:
    """Lightweight per-block FFT denoiser aimed at stationary fan noise."""

    def __init__(
        self,
        sr,
        channels,
        low_hz=120.0,
        high_hz=4200.0,
        strength=1.2,
        floor_ratio=0.15,
        warmup_sec=0.5,
    ):
        self.sr = int(sr)
        self.channels = max(1, int(channels))
        self.low_hz = float(max(0.0, low_hz))
        self.high_hz = float(max(self.low_hz + 10.0, high_hz))
        self.strength = float(max(0.0, strength))
        self.floor_ratio = float(min(max(floor_ratio, 0.0), 1.0))
        self.warmup_sec = float(max(0.0, warmup_sec))
        self.noise_psd = None
        self.noise_rms = np.full((self.channels,), 1e-3, dtype=np.float32)
        self.processed_samples = 0

    def fit_noise_profile(self, pcm):
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 2 or pcm.shape[0] == 0:
            return
        n, ch = pcm.shape
        self.channels = ch
        spec = np.fft.rfft(pcm, axis=0)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(self.sr))
        band_mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        spec[~band_mask, :] = 0.0
        power = np.abs(spec) ** 2
        self.noise_psd = power.copy()
        self.noise_rms = np.sqrt(np.mean(pcm * pcm, axis=0) + 1e-12).astype(np.float32)
        self.processed_samples = int(round(self.warmup_sec * self.sr))

    def save_noise_profile(self, path):
        if self.noise_psd is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            sr=np.asarray([self.sr], dtype=np.int32),
            channels=np.asarray([self.channels], dtype=np.int32),
            low_hz=np.asarray([self.low_hz], dtype=np.float32),
            high_hz=np.asarray([self.high_hz], dtype=np.float32),
            noise_psd=self.noise_psd.astype(np.float32),
            noise_rms=self.noise_rms.astype(np.float32),
        )

    def load_noise_profile(self, path):
        blob = np.load(str(path))
        noise_psd = np.asarray(blob["noise_psd"], dtype=np.float32)
        noise_rms = np.asarray(blob["noise_rms"], dtype=np.float32).reshape(-1)
        self.noise_psd = noise_psd
        self.channels = int(noise_psd.shape[1])
        self.noise_rms = noise_rms
        self.processed_samples = int(round(self.warmup_sec * self.sr))

    def process_block(self, pcm):
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 2 or pcm.shape[0] == 0:
            return pcm

        n, ch = pcm.shape
        if ch != self.channels:
            self.channels = ch
            self.noise_psd = None
            self.noise_rms = np.full((self.channels,), 1e-3, dtype=np.float32)

        spec = np.fft.rfft(pcm, axis=0)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(self.sr))
        band_mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        spec[~band_mask, :] = 0.0

        power = np.abs(spec) ** 2
        if self.noise_psd is None or self.noise_psd.shape != power.shape:
            self.noise_psd = power.copy()

        frame_rms = np.sqrt(np.mean(pcm * pcm, axis=0) + 1e-12).astype(np.float32)
        warmup = self.processed_samples < int(self.warmup_sec * self.sr)
        update_mask = warmup | (frame_rms <= (self.noise_rms * 2.0))

        for c in range(self.channels):
            alpha = 0.92 if update_mask[c] else 0.995
            self.noise_psd[:, c] = alpha * self.noise_psd[:, c] + (1.0 - alpha) * power[:, c]
            rms_alpha = 0.90 if update_mask[c] else 0.995
            self.noise_rms[c] = rms_alpha * self.noise_rms[c] + (1.0 - rms_alpha) * frame_rms[c]

        mag = np.abs(spec)
        noise_mag = np.sqrt(np.maximum(self.noise_psd, 1e-12))
        clean_mag = np.maximum(mag - self.strength * noise_mag, self.floor_ratio * mag)
        clean_spec = clean_mag * np.exp(1j * np.angle(spec))
        out = np.fft.irfft(clean_spec, n=n, axis=0).astype(np.float32, copy=False)
        self.processed_samples += n
        return np.clip(out, -1.0, 1.0)


def record_raw_alsa_block(card_name, sr, channels, duration_sec):
    duration_sec = float(duration_sec)
    if duration_sec <= 0:
        raise RuntimeError(f"duration_sec must be > 0, got {duration_sec}")
    frame_count = int(round(duration_sec * int(sr)))
    cmd = [
        "arecord",
        "-D",
        f"plughw:{card_name},0",
        "-q",
        "-t",
        "raw",
        "-f",
        "S16_LE",
        "-r",
        str(int(sr)),
        "-c",
        str(int(channels)),
        "--samples",
        str(frame_count),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"arecord failed while sampling noise ({proc.returncode}): {stderr or 'unknown error'}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    frames = pcm.size // int(channels)
    if frames <= 0:
        raise RuntimeError("noise sample capture returned no audio")
    pcm = pcm[: frames * int(channels)].reshape(frames, int(channels)).astype(np.float32) / 32768.0
    return pcm


HTTP_STREAM_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TalkNet Realtime</title>
  <style>
    body {{
      margin: 0;
      background: #111;
      color: #eee;
      font-family: sans-serif;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .meta {{
      margin-bottom: 16px;
      color: #bbb;
      line-height: 1.6;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      background: #000;
      border-radius: 12px;
    }}
    code {{
      background: #1d1d1d;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>TalkNet 实时检测画面</h1>
    <div class="meta">
      页面地址：<code>{root_url}</code><br>
      MJPEG 地址：<code>{stream_url}</code>
    </div>
    <div class="meta">
      帧序号：<code id="frame-id">-</code><br>
      服务端采集时间：<code id="capture-ts">-</code><br>
      网页端估计延迟：<code id="latency-ms">-</code><br>
      网页本地时间：<code id="client-ts">-</code>
    </div>
    <img src="/stream.mjpg" alt="TalkNet stream">
  </div>
  <script>
    async function refreshStats() {{
      try {{
        const resp = await fetch('/stats.json', {{ cache: 'no-store' }});
        if (!resp.ok) return;
        const stats = await resp.json();
        const now = Date.now();
        const latency = (typeof stats.capture_ts_ms === 'number') ? (now - stats.capture_ts_ms) : null;
        document.getElementById('frame-id').textContent = stats.frame_id ?? '-';
        document.getElementById('capture-ts').textContent =
          (typeof stats.capture_ts_ms === 'number')
            ? new Date(stats.capture_ts_ms).toLocaleString() + ' (' + stats.capture_ts_ms + ' ms)'
            : '-';
        document.getElementById('latency-ms').textContent =
          (latency !== null) ? (latency + ' ms') : '-';
        document.getElementById('client-ts').textContent =
          new Date(now).toLocaleString() + ' (' + now + ' ms)';
      }} catch (e) {{}}
    }}
    refreshStats();
    setInterval(refreshStats, 500);
  </script>
</body>
</html>
"""


class ThreadingHTTPServer(ThreadingMixIn, server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class HttpFrameStreamer:
    def __init__(self, host: str, port: int, jpeg_quality: int = 80):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.lock = threading.Lock()
        self.frame_jpeg = None
        self.capture_ts_ms = None
        self.frame_id = 0
        self.httpd = None
        self.thread = None
        self.display_host = self._infer_host_ip() if host == "0.0.0.0" else host

    def _infer_host_ip(self) -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            sock.close()

    def start(self):
        self.httpd = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        print(f"[http] listen      : http://{self.host}:{self.port}/")
        print(f"[http] browser url : http://{self.display_host}:{self.port}/")
        print(f"[http] mjpeg url   : http://{self.display_host}:{self.port}/stream.mjpg")

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def update(self, frame):
        capture_ts_ms = int(time.time() * 1000)
        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"frame={self.frame_id + 1} ts={capture_ts_ms}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(
            ".jpg",
            overlay,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        with self.lock:
            self.frame_id += 1
            self.frame_jpeg = encoded.tobytes()
            self.capture_ts_ms = capture_ts_ms

    def get_frame(self):
        with self.lock:
            return self.frame_jpeg

    def get_stats(self):
        with self.lock:
            return {
                "frame_id": self.frame_id,
                "capture_ts_ms": self.capture_ts_ms,
            }

    def _build_handler(self):
        outer = self

        class Handler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    root_url = f"http://{outer.display_host}:{outer.port}/"
                    stream_url = f"http://{outer.display_host}:{outer.port}/stream.mjpg"
                    body = HTTP_STREAM_HTML.format(root_url=root_url, stream_url=stream_url).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/stats.json":
                    body = json.dumps(outer.get_stats()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/healthz":
                    body = b"ok\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            frame = outer.get_frame()
                            if frame is None:
                                time.sleep(0.05)
                                continue
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(frame)))
                            self.end_headers()
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            time.sleep(0.03)
                    except (BrokenPipeError, ConnectionResetError):
                        return

                self.send_error(404)

            def log_message(self, fmt, *args):
                return

        return Handler


class MediaPipeFaceDetector:
    """Adapter to mimic S3FD detect_faces() output format."""

    def __init__(
        self,
        min_conf=0.5,
        model_selection=0,
        model_path="",
        runtime="auto",
        delegate="auto",
    ):
        try:
            import mediapipe as mp
        except Exception as e:
            raise RuntimeError(
                "MediaPipe backend requires mediapipe. Install with: pip install mediapipe"
            ) from e

        self.mp = mp
        self.min_conf = float(min_conf)
        runtime = str(runtime).lower()
        delegate = str(delegate).lower()
        if runtime not in ("auto", "legacy", "tasks"):
            raise ValueError(f"Unsupported MediaPipe runtime: {runtime}")
        if delegate not in ("auto", "cpu", "gpu"):
            raise ValueError(f"Unsupported MediaPipe delegate: {delegate}")
        self.runtime = runtime
        self.delegate = delegate

        has_legacy = hasattr(mp, "solutions")
        has_tasks = False
        try:
            from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
            from mediapipe.tasks.python.core.base_options import BaseOptions
            has_tasks = True
        except Exception:
            FaceDetector = None
            FaceDetectorOptions = None
            BaseOptions = None

        if runtime == "legacy":
            if not has_legacy:
                raise RuntimeError("MediaPipe legacy runtime is unavailable in this installation")
            self.mode = "legacy"
        elif runtime == "tasks":
            if not has_tasks:
                raise RuntimeError("MediaPipe Tasks runtime is unavailable in this installation")
            self.mode = "tasks"
        elif delegate == "gpu":
            if not has_tasks:
                raise RuntimeError("MediaPipe GPU delegate requires Tasks runtime support")
            self.mode = "tasks"
        else:
            self.mode = "legacy" if has_legacy else "tasks"

        if self.mode == "legacy":
            if delegate == "gpu":
                raise RuntimeError("MediaPipe legacy runtime does not expose a GPU delegate")
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=int(model_selection),
                min_detection_confidence=float(min_conf),
            )
            return

        # mediapipe>=0.10 on some wheels exposes Tasks API only.
        # Use BlazeFace tflite model (auto-download if missing).

        mp_model_path = self._ensure_tasks_model(int(model_selection), model_path)
        base_options_kwargs = {"model_asset_path": mp_model_path}
        if delegate in ("cpu", "gpu"):
            base_options_kwargs["delegate"] = (
                BaseOptions.Delegate.GPU if delegate == "gpu" else BaseOptions.Delegate.CPU
            )
        opts = FaceDetectorOptions(
            base_options=BaseOptions(**base_options_kwargs),
            min_detection_confidence=float(min_conf),
        )
        self.detector = FaceDetector.create_from_options(opts)

    def _ensure_tasks_model(self, model_selection, model_path):
        import urllib.request

        if model_path:
            model_path = os.path.abspath(model_path)
            if not os.path.isfile(model_path):
                raise RuntimeError(f"MediaPipe model not found: {model_path}")
            return model_path

        model_dir = os.path.abspath(os.path.join("model", "faceDetector", "mediapipe"))
        os.makedirs(model_dir, exist_ok=True)

        if model_selection == 1:
            filename = "blaze_face_full_range_sparse.tflite"
            urls = [
                "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range_sparse/float16/1/blaze_face_full_range_sparse.tflite",
                "https://storage.googleapis.com/mediapipe-assets/face_detection_full_range.tflite",
            ]
        else:
            filename = "blaze_face_short_range.tflite"
            urls = [
                "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
                "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite",
            ]

        dst = os.path.join(model_dir, filename)
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            return dst

        last_err = None
        for u in urls:
            try:
                print(f"mediapipe model missing, downloading: {u}")
                urllib.request.urlretrieve(u, dst)
                if os.path.isfile(dst) and os.path.getsize(dst) > 0:
                    print(f"mediapipe model ready: {dst}")
                    return dst
            except Exception as e:
                last_err = e

        raise RuntimeError(
            "Failed to prepare MediaPipe face model. "
            f"Tried: {urls}. "
            f"Please set --mpModelPath /abs/path/to/*.tflite. last_error={last_err}"
        )

    def detect_faces(self, image_rgb, conf_th=0.5, scales=None):
        _ = scales
        h, w = image_rgb.shape[:2]
        out = []
        thr = float(conf_th)

        if self.mode == "legacy":
            res = self.detector.process(image_rgb)
            if not res.detections:
                return out
            for det in res.detections:
                score = float(det.score[0]) if det.score else 0.0
                if score < thr:
                    continue
                rb = det.location_data.relative_bounding_box
                x1 = max(0.0, min(float(w - 1), float(rb.xmin) * w))
                y1 = max(0.0, min(float(h - 1), float(rb.ymin) * h))
                x2 = max(0.0, min(float(w - 1), (float(rb.xmin) + float(rb.width)) * w))
                y2 = max(0.0, min(float(h - 1), (float(rb.ymin) + float(rb.height)) * h))
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append([x1, y1, x2, y2, score])
            return out

        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)
        if not result or not result.detections:
            return out
        for det in result.detections:
            score = 0.0
            if getattr(det, "categories", None):
                score = float(det.categories[0].score)
            if score < thr:
                continue
            bb = det.bounding_box
            x1 = max(0.0, min(float(w - 1), float(bb.origin_x)))
            y1 = max(0.0, min(float(h - 1), float(bb.origin_y)))
            x2 = max(0.0, min(float(w - 1), float(bb.origin_x + bb.width)))
            y2 = max(0.0, min(float(h - 1), float(bb.origin_y + bb.height)))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append([x1, y1, x2, y2, score])
        return out


def build_face_detector(args):
    if args.faceDetectorBackend == "s3fd_trt":
        from trt_support import S3FDTRTDetector

        engine_specs = parse_s3fd_trt_engine_specs(args)
        detector = S3FDTRTDetector(engine_specs=engine_specs, device="cuda")
        scales = ",".join(str(x) for x in sorted(engine_specs.keys()))
        print(f"face detector backend: s3fd_trt (scales={scales}, conf={args.detConf:.2f})")
        return detector

    if args.faceDetectorBackend == "mediapipe":
        detector = MediaPipeFaceDetector(
            min_conf=max(1e-4, float(args.detConf)),
            model_selection=args.mpModelSelection,
            model_path=args.mpModelPath,
            runtime=args.mpRuntime,
            delegate=args.mpDelegate,
        )
        print(
            "face detector backend: mediapipe "
            f"(runtime={detector.mode}, delegate={args.mpDelegate}, "
            f"model_selection={args.mpModelSelection}, conf={args.detConf:.2f})"
        )
        return detector

    detector = S3FD(device="cuda")
    print(f"face detector backend: s3fd (scale={args.facedetScale:.3f}, conf={args.detConf:.2f})")
    return detector


def _latency_triplet(vals):
    if not vals:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(vals, dtype=np.float32)
    return (float(arr.mean()), float(np.percentile(arr, 50)), float(np.percentile(arr, 95)))


def parse_face_det_scales(args):
    if getattr(args, "facedetScales", ""):
        vals = []
        for item in str(args.facedetScales).split(","):
            item = item.strip()
            if not item:
                continue
            vals.append(float(item))
        if vals:
            return vals
    return [float(args.facedetScale)]


def parse_s3fd_trt_engine_specs(args):
    raw = str(getattr(args, "s3fdTrtEngineSpecs", "") or "").strip()
    if not raw:
        raise RuntimeError(
            "s3fd_trt requires --s3fdTrtEngineSpecs, e.g. "
            "--s3fdTrtEngineSpecs 0.5=engines/s3fd_960x540_s0.5.engine,1.0=engines/s3fd_1920x1080_s1.0.engine"
        )
    out = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(f"Invalid --s3fdTrtEngineSpecs item: {item!r}")
        scale_raw, path_raw = item.split("=", 1)
        scale = float(scale_raw.strip())
        path = Path(path_raw.strip()).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parent / path).resolve()
        if not path.exists():
            raise RuntimeError(f"S3FD TRT engine not found for scale {scale}: {path}")
        out[scale] = str(path)
    if not out:
        raise RuntimeError("No valid S3FD TRT engines parsed from --s3fdTrtEngineSpecs")
    return out


def _print_latency_report(lat):
    order = [
        "capture_read",
        "face_detect_track",
        "crop_tracks",
        "talknet_infer",
        "draw_overlay",
        "display",
        "loop_total",
    ]
    print("\n=== Realtime Stage Latency (ms) ===")
    print("stage                 mean    p50    p95")
    for k in order:
        m, p50, p95 = _latency_triplet(lat.get(k, []))
        print(f"{k:20s} {m:7.1f} {p50:6.1f} {p95:6.1f}")


def prepare_temp_alsa_default(card_name: str, channels: int, rate: int):
    card_name = str(card_name).strip()
    if not card_name:
        raise RuntimeError("Empty ALSA card name")
    cfg = "\n".join(
        [
            "pcm.aiui_capture_raw {",
            "    type hw",
            f"    card {card_name}",
            "    device 0",
            "}",
            "",
            "pcm.aiui_capture {",
            "    type plug",
            '    slave.pcm "aiui_capture_raw"',
            "}",
            "",
            "pcm.sysdefault {",
            "    type asym",
            '    capture.pcm "aiui_capture"',
            '    playback.pcm "null"',
            "}",
            "",
            "pcm.!default {",
            "    type asym",
            '    capture.pcm "aiui_capture"',
            '    playback.pcm "null"',
            "}",
            "",
            "ctl.!default {",
            "    type hw",
            f"    card {card_name}",
            "}",
            "",
        ]
    )
    tmp = tempfile.NamedTemporaryFile(prefix="talknet_asound_", suffix=".conf", delete=False)
    tmp.write(cfg.encode("utf-8"))
    tmp.flush()
    tmp.close()
    os.environ["ALSA_CONFIG_PATH"] = tmp.name
    return tmp.name


def choose_sounddevice_input(sd, args):
    if int(getattr(args, "micDevice", -1)) >= 0:
        mic_id = int(args.micDevice)
        return mic_id, f"id={mic_id}"

    try:
        devices = sd.query_devices()
    except Exception as e:
        raise RuntimeError(f"Failed to enumerate sounddevice inputs: {e}") from e

    candidates = []
    prefer_default = bool(getattr(args, "micAlsaCard", ""))
    for idx, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0) or 0)
        if max_in <= 0:
            continue
        name = str(dev.get("name", ""))
        lname = name.lower()
        score = 0
        if prefer_default:
            if lname == "default" or lname.startswith("default "):
                score += 100
            if "sysdefault" in lname:
                score += 90
            if "pulse" in lname:
                score -= 25
        else:
            if lname == "default" or lname.startswith("default "):
                score += 15
            if "pulse" in lname:
                score += 5
        candidates.append((score, idx, name, max_in))

    if not candidates:
        raise RuntimeError("No usable sounddevice input devices found")

    candidates.sort(key=lambda x: (-x[0], x[1]))
    _, idx, name, max_in = candidates[0]
    return idx, f"{idx}:{name} in={max_in}"


class ArecordAudioStream:
    def __init__(
        self,
        card_name,
        channels,
        sr,
        blocksize,
        audio_buf,
        channel_mode="max_energy",
        denoise=False,
        denoise_low_hz=120.0,
        denoise_high_hz=4200.0,
        denoise_strength=1.2,
        denoise_warmup_sec=0.5,
        noise_profile_path="",
        record_dir="",
        record_prefix="mic",
        save_denoised_mono=True,
        mono_recorder=None,
    ):
        self.card_name = str(card_name).strip()
        self.channels = max(1, int(channels))
        self.sr = int(sr)
        self.blocksize = max(256, int(blocksize))
        self.audio_buf = audio_buf
        self.channel_mode = str(channel_mode).strip().lower()
        self.denoise = bool(denoise)
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()
        self.sample_index = 0
        self.mono_recorder = mono_recorder
        self.recorder = None
        if record_dir:
            self.recorder = SplitChannelWavRecorder(
                out_dir=record_dir,
                sr=self.sr,
                channels=self.channels,
                prefix=record_prefix,
                save_denoised_mono=save_denoised_mono,
            )
        self.noise_reducer = None
        if self.denoise:
            self.noise_reducer = FanNoiseReducer(
                sr=self.sr,
                channels=self.channels,
                low_hz=denoise_low_hz,
                high_hz=denoise_high_hz,
                strength=denoise_strength,
                warmup_sec=denoise_warmup_sec,
            )
            if noise_profile_path:
                profile_path = Path(noise_profile_path)
                if profile_path.exists():
                    self.noise_reducer.load_noise_profile(profile_path)

    def start(self):
        if not self.card_name:
            raise RuntimeError("ArecordAudioStream requires a non-empty ALSA card name")
        cmd = [
            "arecord",
            "-D",
            f"plughw:{self.card_name},0",
            "-q",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(self.sr),
            "-c",
            str(self.channels),
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start arecord on {self.card_name}: {e}") from e

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _collapse_to_mono(self, pcm):
        if pcm.ndim != 2 or pcm.shape[1] <= 1:
            return pcm.reshape(-1)
        if self.channel_mode == "first":
            return pcm[:, 0]
        if self.channel_mode == "mean":
            return np.mean(pcm, axis=1)
        energies = np.mean(pcm ** 2, axis=0)
        best_ch = int(np.argmax(energies))
        return pcm[:, best_ch]

    def _reader_loop(self):
        bytes_per_frame = self.channels * 2
        chunk_bytes = self.blocksize * bytes_per_frame
        while not self.stop_event.is_set() and self.proc is not None and self.proc.stdout is not None:
            raw = self.proc.stdout.read(chunk_bytes)
            if not raw:
                break
            n = (len(raw) // bytes_per_frame) * bytes_per_frame
            if n <= 0:
                continue
            raw = raw[:n]
            pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, self.channels).astype(np.float32) / 32768.0
            if self.recorder is not None:
                self.recorder.write_raw_channels(pcm)
            proc_pcm = pcm
            if self.noise_reducer is not None:
                proc_pcm = self.noise_reducer.process_block(proc_pcm)
            mono = self._collapse_to_mono(proc_pcm)
            if self.recorder is not None:
                self.recorder.write_denoised_mono(mono)
            if self.mono_recorder is not None:
                self.mono_recorder.write(mono)
            t0_sec = float(self.sample_index) / float(self.sr)
            self.sample_index += mono.shape[0]
            t1_sec = float(self.sample_index) / float(self.sr)
            self.audio_buf.push_float(mono, t0_sec=t0_sec, t1_sec=t1_sec)

    def stop(self):
        self.stop_event.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def close(self):
        if self.recorder is not None:
            self.recorder.close()
        if self.mono_recorder is not None:
            self.mono_recorder.close()
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=1.0)
            except Exception:
                pass


class RobotHeadController:
    def __init__(self, args, frame_w, frame_h):
        self.args = args
        self.frame_w = max(1, int(frame_w))
        self.frame_h = max(1, int(frame_h))
        self.enabled = bool(args.enableRobotHeadTrack)
        self.rclpy = None
        self.node = None
        self.pub = None
        self.pub_enable = None
        self.pub_cmd = None
        self.pub_head_topic = None
        self.msg_cls = None
        self.joint_cls = None
        self.enable_msg_cls = None
        self.cmd_msg_cls = None
        self.head_topic_msg_cls = None
        self.pub_up = None
        self.pub_down = None
        self.enabled_servos = set()

        if not self.enabled:
            return

        try:
            import rclpy
            self.rclpy = rclpy
            try:
                if not rclpy.ok():
                    rclpy.init(args=None)
            except Exception:
                rclpy.init(args=None)

            self.node = rclpy.create_node("asd_head_tracker")

            if args.robotControlMode == "fd_dm_pos":
                self._init_fd_dm_pos()
            elif args.robotControlMode == "fd_mit":
                self._init_fd_mit()
            elif args.robotControlMode == "fd_head_topic":
                self._init_fd_head_topic()
            else:
                self._init_rosci()
        except Exception as e:
            self.enabled = False
            print(
                f"robot head tracking disabled: failed to init ROS publisher ({e}). "
                "hint: source /opt/ros/humble/setup.bash && source robot workspace install/setup.bash"
            )

    def _init_rosci(self):
        from importlib import import_module
        msg_mod = import_module(self.args.robotMsgModule)
        self.msg_cls = getattr(msg_mod, self.args.robotCmdMsg)
        self.joint_cls = getattr(msg_mod, self.args.robotJointMsg)
        self.pub = self.node.create_publisher(self.msg_cls, self.args.robotCmdTopic, 10)
        print(
            f"robot head tracking enabled (rosci): topic={self.args.robotCmdTopic}, "
            f"msg={self.args.robotMsgModule}.{self.args.robotCmdMsg}"
        )

    def _init_fd_dm_pos(self):
        from importlib import import_module

        msg_mod = import_module(self.args.fdMsgModule)
        self.msg_cls = getattr(msg_mod, self.args.fdPosMsg)
        self.pub = self.node.create_publisher(self.msg_cls, self.args.fdPosTopic, 10)

        # Optional fallback topics for pitch-only movement if servo id not configured.
        try:
            from std_msgs.msg import Float32
            self.pub_up = self.node.create_publisher(Float32, self.args.fdHeadUpTopic, 10)
            self.pub_down = self.node.create_publisher(Float32, self.args.fdHeadDownTopic, 10)
        except Exception:
            self.pub_up = None
            self.pub_down = None

        print(
            "robot head tracking enabled (fd_dm_pos): "
            f"topic={self.args.fdPosTopic}, msg={self.args.fdMsgModule}.{self.args.fdPosMsg}, "
            f"yaw_servo={self.args.fdYawServoId}, pitch_servo={self.args.fdPitchServoId}"
        )

    def _init_fd_mit(self):
        from importlib import import_module

        msg_mod = import_module(self.args.fdMsgModule)
        self.enable_msg_cls = getattr(msg_mod, self.args.fdEnableMsg)
        self.cmd_msg_cls = getattr(msg_mod, self.args.fdMitMsg)
        self.pub_enable = self.node.create_publisher(self.enable_msg_cls, self.args.fdEnableTopic, 10)
        self.pub_cmd = self.node.create_publisher(self.cmd_msg_cls, self.args.fdMitTopic, 10)
        print(
            "robot head tracking enabled (fd_mit): "
            f"enable_topic={self.args.fdEnableTopic}, enable_msg={self.args.fdMsgModule}.{self.args.fdEnableMsg}, "
            f"mit_topic={self.args.fdMitTopic}, mit_msg={self.args.fdMsgModule}.{self.args.fdMitMsg}, "
            f"yaw_servo={self.args.fdYawServoId}, pitch_servo={self.args.fdPitchServoId}"
        )

    def _init_fd_head_topic(self):
        from fd_head_control.msg import HeadAngleCmd

        self.head_topic_msg_cls = HeadAngleCmd
        self.pub_head_topic = self.node.create_publisher(HeadAngleCmd, self.args.fdHeadTopicCmd, 10)
        print(
            "robot head tracking enabled (fd_head_topic): "
            f"topic={self.args.fdHeadTopicCmd}, msg=fd_head_control.msg.HeadAngleCmd"
        )

    def _clamp(self, x, lo, hi):
        return max(lo, min(hi, x))

    def _bbox_to_cmd(self, bbox):
        pos = bbox_face_position(bbox, self.frame_w, self.frame_h, flip=bool(getattr(self.args, "flip", False)))
        ex = pos["offset"]["ex"]
        ey = pos["offset"]["ey"]
        yaw = self._clamp(self.args.robotYawGain * ex, -self.args.robotMaxYawRad, self.args.robotMaxYawRad)
        pitch = self._clamp(-self.args.robotPitchGain * ey, -self.args.robotMaxPitchRad, self.args.robotMaxPitchRad)
        return yaw, pitch, ex, ey

    def _publish_rosci(self, yaw, pitch):
        msg = self.msg_cls()
        if hasattr(msg, "header") and self.node is not None:
            msg.header.stamp = self.node.get_clock().now().to_msg()
            if hasattr(msg.header, "frame_id"):
                msg.header.frame_id = self.args.robotFrameId

        if hasattr(msg, "enable_head"):
            msg.enable_head = [1, 1]
        if hasattr(msg, "enable_waist"):
            msg.enable_waist = [0, 0]

        req_op = getattr(self.msg_cls, "REQUEST_STATE_OPERATION", 4)
        if hasattr(msg, "head_request_state"):
            msg.head_request_state = [req_op, req_op]
        if hasattr(msg, "waist_request_state"):
            msg.waist_request_state = [0, 0]

        if hasattr(msg, "head_joint") and len(msg.head_joint) >= 2:
            # Convention: index 0 = pitch, 1 = yaw
            msg.head_joint[0].position_desired = float(pitch)
            msg.head_joint[0].velocity_gain = float(self.args.robotVelGain)
            msg.head_joint[0].acceleration_gain = float(self.args.robotAccGain)

            msg.head_joint[1].position_desired = float(yaw)
            msg.head_joint[1].velocity_gain = float(self.args.robotVelGain)
            msg.head_joint[1].acceleration_gain = float(self.args.robotAccGain)

        self.pub.publish(msg)

    def _publish_fd_dm_pos(self, yaw, pitch):
        sent = 0
        yaw_cmd = float(self.args.fdYawSign * yaw)
        pitch_cmd = float(self.args.fdPitchSign * pitch)

        if int(self.args.fdYawServoId) >= 0:
            m = self.msg_cls()
            m.servo_id = int(self.args.fdYawServoId)
            m.pose = float(yaw_cmd)
            m.vel = float(self.args.fdCmdVel)
            self.pub.publish(m)
            sent += 1

        if int(self.args.fdPitchServoId) >= 0:
            m = self.msg_cls()
            m.servo_id = int(self.args.fdPitchServoId)
            m.pose = float(pitch_cmd)
            m.vel = float(self.args.fdCmdVel)
            self.pub.publish(m)
            sent += 1
        elif self.pub_up is not None and self.pub_down is not None and abs(pitch_cmd) > 1e-4:
            # Fallback: publish magnitude to up/down topics.
            from std_msgs.msg import Float32
            mm = Float32()
            mm.data = float(abs(pitch_cmd))
            if pitch_cmd > 0:
                self.pub_up.publish(mm)
            else:
                self.pub_down.publish(mm)
            sent += 1

        if sent == 0:
            raise RuntimeError(
                "fd_dm_pos mode has no valid output axis. "
                "Set --fdYawServoId/--fdPitchServoId, or use roscI mode."
            )

    def _publish_fd_enable_once(self, servo_id):
        servo_id = int(servo_id)
        if servo_id < 0 or servo_id in self.enabled_servos:
            return
        if self.pub_enable is None or self.enable_msg_cls is None:
            raise RuntimeError("fd_mit enable publisher is not initialized")
        m = self.enable_msg_cls()
        if not hasattr(m, "servo_id") or not hasattr(m, "enable"):
            raise RuntimeError("EnableMsg missing expected fields servo_id/enable")
        m.servo_id = servo_id
        m.enable = True
        self.pub_enable.publish(m)
        self.enabled_servos.add(servo_id)

    def _publish_fd_mit(self, yaw, pitch):
        sent = 0
        yaw_cmd = float(self.args.fdYawSign * yaw)
        pitch_cmd = float(self.args.fdPitchSign * pitch)

        def _send(servo_id, pose_cmd):
            nonlocal sent
            servo_id = int(servo_id)
            if servo_id < 0:
                return
            self._publish_fd_enable_once(servo_id)
            m = self.cmd_msg_cls()
            required = ("servo_id", "pose", "vel", "kp", "kd", "torque")
            missing = [name for name in required if not hasattr(m, name)]
            if missing:
                raise RuntimeError(f"DmMITControlMsg missing expected fields: {missing}")
            m.servo_id = servo_id
            m.pose = float(pose_cmd)
            m.vel = float(self.args.fdMitVel)
            m.kp = float(self.args.fdMitKp)
            m.kd = float(self.args.fdMitKd)
            m.torque = float(self.args.fdMitTorque)
            self.pub_cmd.publish(m)
            sent += 1

        _send(self.args.fdYawServoId, yaw_cmd)
        _send(self.args.fdPitchServoId, pitch_cmd)

        if sent == 0:
            raise RuntimeError(
                "fd_mit mode has no valid output axis. "
                "Set --fdYawServoId/--fdPitchServoId to valid servo ids."
            )

    def _publish_fd_head_topic(self, yaw, pitch):
        if self.pub_head_topic is None or self.head_topic_msg_cls is None:
            raise RuntimeError("fd_head_topic publisher is not initialized")
        msg = self.head_topic_msg_cls()
        msg.yaw = float(yaw)
        msg.pitch = float(pitch)
        self.pub_head_topic.publish(msg)

    def publish_head_center_cmd(self, bbox):
        if not self.enabled:
            return False, 0.0, 0.0, 0.0, 0.0

        yaw, pitch, ex, ey = self._bbox_to_cmd(bbox)

        try:
            if self.args.robotControlMode == "fd_dm_pos":
                self._publish_fd_dm_pos(yaw, pitch)
            elif self.args.robotControlMode == "fd_mit":
                self._publish_fd_mit(yaw, pitch)
            elif self.args.robotControlMode == "fd_head_topic":
                self._publish_fd_head_topic(yaw, pitch)
            else:
                if self.pub is None:
                    return False, yaw, pitch, ex, ey
                self._publish_rosci(yaw, pitch)

            if self.rclpy is not None and self.node is not None:
                try:
                    self.rclpy.spin_once(self.node, timeout_sec=0.0)
                except Exception:
                    pass
            return True, yaw, pitch, ex, ey
        except Exception as e:
            print(f"robot command publish failed: {e}")
            return False, yaw, pitch, ex, ey

    def shutdown(self):
        if not self.enabled:
            return
        try:
            if self.node is not None:
                self.node.destroy_node()
        except Exception:
            pass
        try:
            if self.rclpy is not None and self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:
            pass


def _to_easydict(d):
    from easydict import EasyDict
    if isinstance(d, dict):
        return EasyDict({k: _to_easydict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_easydict(x) for x in d]
    return d


def _build_talknce_backend(args):
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("TalkNCE backend requires pyyaml. Install: pip install pyyaml") from e
    try:
        from easydict import EasyDict  # noqa: F401
    except Exception as e:
        raise RuntimeError("TalkNCE backend requires easydict. Install: pip install easydict") from e

    root = os.path.abspath(args.talknceRoot)
    if not os.path.isdir(root):
        raise RuntimeError(f"TalkNCE root not found: {root}")

    cfg_path = args.talknceCfg or os.path.join(root, "configs", "test.yaml")
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.abspath(cfg_path)
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f"TalkNCE cfg not found: {cfg_path}")

    ckpt = args.talknceCkpt
    if not ckpt:
        candidates = [
            os.path.join(root, "talknce_ava_pretrained.model"),
            os.path.join(root, "talknce_AVA.model"),
        ]
        candidates += sorted(glob.glob(os.path.join(root, "*.model")))
        ckpt = next((c for c in candidates if os.path.isfile(c)), "")
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.abspath(ckpt)
    if not ckpt or not os.path.isfile(ckpt):
        raise RuntimeError(
            "TalkNCE checkpoint not found. Set --talknceCkpt /abs/path/to/*.model"
        )

    # TalkNCE uses absolute imports like `from model...`. This repo also has a
    # top-level `model` package, so we isolate import resolution while loading TalkNCE.
    repo_root = os.path.abspath(os.path.dirname(__file__))
    old_path = list(sys.path)
    old_model_modules = {
        k: v for k, v in sys.modules.items() if (k == "model" or k.startswith("model."))
    }
    global TALKNCE_VGGISH_INPUT
    try:
        sys.path[:] = [root] + [p for p in old_path if os.path.abspath(p) != repo_root]
        for k in list(sys.modules.keys()):
            if k == "model" or k.startswith("model."):
                sys.modules.pop(k, None)
        importlib.invalidate_caches()
        loconet_mod = importlib.import_module("loconet")
        TALKNCE_VGGISH_INPUT = importlib.import_module("torchvggish.vggish_input")
    finally:
        sys.path[:] = old_path
        # Restore original modules from this repo for downstream detector code.
        for k in list(sys.modules.keys()):
            if k == "model" or k.startswith("model."):
                sys.modules.pop(k, None)
        sys.modules.update(old_model_modules)
    loconet_cls = getattr(loconet_mod, "loconet")

    import yaml
    with open(cfg_path, "r") as f:
        cfg_dict = yaml.full_load(f)

    # Minimal fields used by inference path.
    cfg_dict.setdefault("OUTPUT_DIR", os.path.join(root, "workspace_online"))
    cfg_dict.setdefault("NUM_GPUS", 1)
    cfg_dict.setdefault("LOG_NAME", "online.log")
    cfg = _to_easydict(cfg_dict)
    cfg.WORKSPACE = cfg.OUTPUT_DIR

    backend = loconet_cls(cfg)
    backend.loadParameters(ckpt)
    backend.eval()
    return backend, ckpt

def iou(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter / (area_a + area_b - inter + 1e-9)


def crop_face_gray112(frame, bbox, crop_scale=0.40):
    # Match demoTalkNet.crop_video preprocessing as closely as possible.
    x1, y1, x2, y2 = bbox
    bs = max((x2 - x1), (y2 - y1)) / 2.0
    if bs < 1.0:
        return np.zeros((112, 112), dtype=np.uint8)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bsi = int(bs * (1.0 + 2.0 * crop_scale))

    padded = np.pad(frame, ((bsi, bsi), (bsi, bsi), (0, 0)), mode="constant", constant_values=110)
    mx = cx + bsi
    my = cy + bsi

    ya = int(my - bs)
    yb = int(my + bs * (1.0 + 2.0 * crop_scale))
    xa = int(mx - bs)
    xb = int(mx + bs * (1.0 + crop_scale))

    h, w = padded.shape[:2]
    ya = max(0, min(h - 1, ya))
    yb = max(ya + 1, min(h, yb))
    xa = max(0, min(w - 1, xa))
    xb = max(xa + 1, min(w, xb))

    face = padded[ya:yb, xa:xb]
    if face.size == 0:
        return np.zeros((112, 112), dtype=np.uint8)

    face224 = cv2.resize(face, (224, 224))
    gray = cv2.cvtColor(face224, cv2.COLOR_BGR2GRAY)
    c = 112
    half = 56
    return gray[c - half : c + half, c - half : c + half]


def extract_audio_wav(video_path, wav_path, sr=AUDIO_SR):
    cmd = (
        "ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -ar %d %s -loglevel panic"
        % (video_path, sr, wav_path)
    )
    rc = subprocess.call(cmd, shell=True)
    if rc != 0:
        raise RuntimeError("ffmpeg audio extraction failed")


class Track:
    def __init__(self, tid, bbox, frame_idx, max_hist):
        self.tid = tid
        self.bbox = bbox
        self.bbox_hist = collections.deque(maxlen=10)
        self.bbox_hist.append(np.asarray(bbox, dtype=np.float32))
        self.last_frame = frame_idx
        self.missed = 0
        # None means this track has not produced a valid ASD score yet.
        self.score = None
        self.face_hist = collections.deque(maxlen=max_hist)  # (time_sec, face112)
        self.score_hist = collections.deque(maxlen=16)


class RealtimeAudioBuffer:
    """Ring-buffered microphone PCM with wall-clock aligned timeline."""

    def __init__(self, sr=AUDIO_SR, max_seconds=30.0):
        self.sr = int(sr)
        self.max_samples = int(max(1.0, max_seconds) * self.sr)
        self.lock = threading.Lock()
        self.chunks = collections.deque()  # list[(t0_sec, t1_sec, np.int16 array)]
        self.first_t0 = None
        self.last_t1 = None

    def push_float(self, mono_float, t0_sec=None, t1_sec=None):
        pcm = np.clip(np.asarray(mono_float).reshape(-1), -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        if pcm.size == 0:
            return

        dur = float(pcm.size) / float(self.sr)
        with self.lock:
            if t0_sec is None or t1_sec is None:
                if self.last_t1 is None:
                    t0_sec = 0.0
                    t1_sec = dur
                else:
                    t0_sec = float(self.last_t1)
                    t1_sec = t0_sec + dur
            else:
                t0_sec = float(t0_sec)
                t1_sec = float(t1_sec)
                if t1_sec <= t0_sec:
                    t1_sec = t0_sec + dur

            self.chunks.append((t0_sec, t1_sec, pcm.copy()))
            if self.first_t0 is None:
                self.first_t0 = t0_sec
            self.last_t1 = t1_sec
            min_keep_t = max(self.first_t0 if self.first_t0 is not None else 0.0, self.last_t1 - (self.max_samples / float(self.sr)))

            while self.chunks and self.chunks[0][1] <= min_keep_t:
                self.chunks.popleft()

            if self.chunks and self.chunks[0][0] < min_keep_t:
                c_t0, c_t1, c_arr = self.chunks[0]
                cut = int(round((min_keep_t - c_t0) * self.sr))
                cut = max(0, min(c_arr.size - 1, cut))
                new_t0 = c_t0 + cut / float(self.sr)
                self.chunks[0] = (new_t0, c_t1, c_arr[cut:])

    def get_segment(self, t0, t1):
        t0 = float(t0)
        t1 = float(t1)
        s0 = max(0, int(round(t0 * self.sr)))
        s1 = max(s0 + 1, int(round(t1 * self.sr)))
        need = s1 - s0
        out = np.zeros((need,), dtype=np.int16)

        with self.lock:
            for c_t0, c_t1, arr in self.chunks:
                lo_t = max(t0, c_t0)
                hi_t = min(t1, c_t1)
                if hi_t <= lo_t:
                    continue
                dst_lo = int(round((lo_t - t0) * self.sr))
                dst_hi = int(round((hi_t - t0) * self.sr))
                src_lo = int(round((lo_t - c_t0) * self.sr))
                src_hi = int(round((hi_t - c_t0) * self.sr))
                dst_lo = max(0, min(need, dst_lo))
                dst_hi = max(dst_lo, min(need, dst_hi))
                src_lo = max(0, min(arr.size, src_lo))
                src_hi = max(src_lo, min(arr.size, src_hi))
                copy_n = min(dst_hi - dst_lo, src_hi - src_lo)
                if copy_n <= 0:
                    continue
                out[dst_lo : dst_lo + copy_n] = arr[src_lo : src_lo + copy_n]
        return out


def summarize_audio_segment(audio_seg):
    if audio_seg is None or len(audio_seg) == 0:
        return 0.0, 0.0, 0.0
    arr = np.asarray(audio_seg, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    if np.issubdtype(audio_seg.dtype, np.integer):
        arr = arr / 32768.0
    rms = float(np.sqrt(np.mean(arr * arr)))
    peak = float(np.max(np.abs(arr)))
    mean_abs = float(np.mean(np.abs(arr)))
    return rms, peak, mean_abs


def score_is_ready(score):
    return score is not None and np.isfinite(score)


def score_text(score, precision=2):
    if not score_is_ready(score):
        return "pending"
    return f"{float(score):.{precision}f}"


def greedy_match(tracks, detections, iou_thres):
    matched = []
    used_d = set()
    for ti, tr in enumerate(tracks):
        best_j = -1
        best_iou = 0.0
        for j, det in enumerate(detections):
            if j in used_d:
                continue
            ov = iou(tr.bbox, det)
            if ov > best_iou:
                best_iou = ov
                best_j = j
        if best_j >= 0 and best_iou >= iou_thres:
            matched.append((ti, best_j))
            used_d.add(best_j)
    unmatched_tracks = [i for i in range(len(tracks)) if i not in {x[0] for x in matched}]
    unmatched_dets = [j for j in range(len(detections)) if j not in used_d]
    return matched, unmatched_tracks, unmatched_dets


def resample_visual_hist(hist, target_fps):
    if len(hist) < 2:
        return None, None
    times = np.asarray([x[0] for x in hist], dtype=np.float32)
    frames = [x[1] for x in hist]
    t_start = float(times[0])
    t_end = float(times[-1])
    duration = max(1e-3, t_end - t_start)
    target_len = max(4, int(round(duration * target_fps)))
    if target_len <= 1:
        return None, None
    src_idx = np.linspace(0, len(frames) - 1, num=target_len)
    src_idx = np.clip(np.round(src_idx).astype(np.int32), 0, len(frames) - 1)
    video_feature = np.stack([frames[i] for i in src_idx], axis=0)
    return video_feature, t_start


def get_smoothed_bbox(track):
    if not track.bbox_hist:
        return track.bbox
    arr = np.stack(track.bbox_hist, axis=0)
    return np.median(arr, axis=0).tolist()


def bbox_face_position(bbox, frame_w, frame_h, flip=False):
    """Face center in pixels and normalized offsets from frame center (ex/ey in [-1, 1])."""
    x1, y1, x2, y2 = bbox
    cx = 0.5 * (float(x1) + float(x2))
    cy = 0.5 * (float(y1) + float(y2))
    fw = max(1, int(frame_w))
    fh = max(1, int(frame_h))
    ex = (cx - 0.5 * fw) / (0.5 * fw)
    ey = (cy - 0.5 * fh) / (0.5 * fh)
    if flip:
        ex = -ex
    return {
        "center": [round(cx, 1), round(cy, 1)],
        "norm": {"cx": round(cx / fw, 4), "cy": round(cy / fh, 4)},
        "offset": {"ex": round(ex, 4), "ey": round(ey, 4)},
        "size": {"w": int(x2 - x1), "h": int(y2 - y1)},
    }


def _top_track_tid(tracks, only_top_speaker):
    ready = [tr for tr in tracks if score_is_ready(tr.score)]
    if only_top_speaker and ready:
        return max(ready, key=lambda t: float(t.score)).tid
    return None


def collect_face_position_records(tracks, args, frame_w, frame_h, top_tid=None):
    records = []
    for tr in tracks:
        x1, y1, x2, y2 = [int(v) for v in get_smoothed_bbox(tr)]
        ready = score_is_ready(tr.score)
        if args.onlyTopSpeaker and top_tid is not None:
            active = ready and (tr.tid == top_tid) and (float(tr.score) >= args.scoreThres)
        else:
            active = ready and (float(tr.score) >= args.scoreThres)
        pos = bbox_face_position((x1, y1, x2, y2), frame_w, frame_h, flip=bool(args.flip))
        records.append(
            {
                "id": int(tr.tid),
                "bbox": [x1, y1, x2, y2],
                **pos,
                "score": round(float(tr.score), 4) if ready else None,
                "active": bool(active),
            }
        )
    return records


def filter_face_position_records(records, mode):
    if mode == "all":
        return records
    if mode == "active":
        return [r for r in records if r["active"]]
    if mode == "top":
        if not records:
            return []
        scored = [r for r in records if r["score"] is not None]
        if scored:
            return [max(scored, key=lambda r: r["score"])]
        return [max(records, key=lambda r: r["size"]["w"] * r["size"]["h"])]
    return records


def emit_face_position_log(records, frame_t, args):
    if not records:
        return
    if args.facePositionFormat == "json":
        payload = {"t": round(float(frame_t), 3), "faces": records}
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return
    for r in records:
        ex = r["offset"]["ex"]
        ey = r["offset"]["ey"]
        sc = r["score"]
        sc_txt = f"{sc:.3f}" if sc is not None else "n/a"
        print(
            f"[face] t={frame_t:.3f}s id={r['id']} center={r['center']} "
            f"offset=({ex:+.3f},{ey:+.3f}) norm=({r['norm']['cx']:.3f},{r['norm']['cy']:.3f}) "
            f"bbox={tuple(r['bbox'])} active={r['active']} score={sc_txt}",
            flush=True,
        )


def infer_talknet_offline_style(speaker, audio_seg, video_frames, duration_set):
    if len(video_frames) < 4 or len(audio_seg) < 160:
        return None

    audio_feature = python_speech_features.mfcc(
        audio_seg,
        AUDIO_SR,
        numcep=13,
        winlen=0.025,
        winstep=0.010,
    )

    length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100.0, video_frames.shape[0] / 25.0)
    if length <= 0:
        return None

    audio_feature = audio_feature[: int(round(length * 100)), :]
    video_frames = video_frames[: int(round(length * 25)), :, :]

    all_scores = []
    with torch.no_grad():
        for duration in duration_set:
            batch_size = int(math.ceil(length / duration))
            scores = []
            for i in range(batch_size):
                a = audio_feature[i * duration * 100 : (i + 1) * duration * 100, :]
                v = video_frames[i * duration * 25 : (i + 1) * duration * 25, :, :]
                if a.shape[0] == 0 or v.shape[0] == 0:
                    continue
                input_a = torch.FloatTensor(a).unsqueeze(0).cuda()
                input_v = torch.FloatTensor(v).unsqueeze(0).cuda()
                embed_a = speaker.model.forward_audio_frontend(input_a)
                embed_v = speaker.model.forward_visual_frontend(input_v)
                embed_a, embed_v = speaker.model.forward_cross_attention(embed_a, embed_v)
                out = speaker.model.forward_audio_visual_backend(embed_a, embed_v)
                score = speaker.lossAV.forward(out, labels=None)
                scores.extend(score)
            if scores:
                all_scores.append(scores)

    if not all_scores:
        return None

    min_len = min(len(x) for x in all_scores)
    arr = np.array([x[:min_len] for x in all_scores], dtype=np.float32)
    return np.mean(arr, axis=0)





def infer_talknce_offline_style(speaker, audio_seg, video_frames, duration_set, model_fps=25.0):
    if len(video_frames) < 4 or len(audio_seg) < 160:
        return None

    if TALKNCE_VGGISH_INPUT is None:
        return None
    num_frames = max(1, int(video_frames.shape[0]))
    audio_feature = TALKNCE_VGGISH_INPUT.waveform_to_examples(
        audio_seg, AUDIO_SR, num_frames, int(round(model_fps)), return_tensor=False
    )

    length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100.0, video_frames.shape[0] / 25.0)
    if length <= 0:
        return None

    audio_feature = audio_feature[: int(round(length * 100)), :]
    video_frames = video_frames[: int(round(length * 25)), :, :]

    all_scores = []
    with torch.no_grad():
        for duration in duration_set:
            batch_size = int(math.ceil(length / duration))
            scores = []
            for i in range(batch_size):
                a = audio_feature[i * duration * 100 : (i + 1) * duration * 100, :]
                v = video_frames[i * duration * 25 : (i + 1) * duration * 25, :, :]
                if a.shape[0] == 0 or v.shape[0] == 0:
                    continue
                input_a = torch.FloatTensor(a).unsqueeze(0).unsqueeze(0).cuda()
                input_v = torch.FloatTensor(v).unsqueeze(0).cuda()
                embed_a = speaker.model.forward_audio_frontend(input_a)
                embed_v = speaker.model.forward_visual_frontend(input_v)
                rep_s = 3
                embed_a = embed_a.repeat(rep_s, 1, 1)
                embed_v = embed_v.repeat(rep_s, 1, 1)
                embed_a, embed_v = speaker.model.forward_cross_attention(embed_a, embed_v)
                out = speaker.model.forward_audio_visual_backend(embed_a, embed_v, 1, rep_s)
                score = speaker.lossAV.forward(out, labels=None)
                if len(score) % rep_s == 0:
                    score = np.asarray(score).reshape(rep_s, -1)[0]
                scores.extend(score)
            if scores:
                all_scores.append(scores)

    if not all_scores:
        return None

    min_len = min(len(x) for x in all_scores)
    arr = np.array([x[:min_len] for x in all_scores], dtype=np.float32)
    return np.mean(arr, axis=0)

def process_frame(
    frame,
    frame_idx,
    frame_t,
    tracks,
    next_tid,
    detector,
    speaker,
    duration_set,
    args,
    window_hist_len,
    get_audio_segment,
    run_infer,
    latency_stats=None,
):
    t_pf0 = time.perf_counter()

    t0 = time.perf_counter()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes = detector.detect_faces(
        rgb,
        conf_th=args.detConf,
        scales=parse_face_det_scales(args),
    )
    detections = [list(map(float, b[:4])) for b in bboxes]

    matched, un_t, un_d = greedy_match(tracks, detections, args.iouThres)

    for ti, dj in matched:
        tr = tracks[ti]
        tr.bbox = detections[dj]
        tr.bbox_hist.append(np.asarray(tr.bbox, dtype=np.float32))
        tr.last_frame = frame_idx
        tr.missed = 0

    for ti in un_t:
        tracks[ti].missed += 1

    tracks = [tr for tr in tracks if tr.missed <= args.maxMissedFrames]

    for dj in un_d:
        tr = Track(next_tid, detections[dj], frame_idx, max_hist=window_hist_len)
        tracks.append(tr)
        next_tid += 1
    t1 = time.perf_counter()
    if latency_stats is not None:
        latency_stats["face_detect_track"].append((t1 - t0) * 1000.0)

    t2 = time.perf_counter()
    for tr in tracks:
        smooth_bbox = get_smoothed_bbox(tr)
        face = crop_face_gray112(frame, smooth_bbox, crop_scale=args.cropScale)
        tr.face_hist.append((frame_t, face))
    t3 = time.perf_counter()
    if latency_stats is not None:
        latency_stats["crop_tracks"].append((t3 - t2) * 1000.0)

    if run_infer:
        ti0 = time.perf_counter()
        with torch.no_grad():
            for tr in tracks:
                hist_start_t = max(0.0, frame_t - args.windowSec)
                hist = [x for x in tr.face_hist if x[0] >= hist_start_t]
                if len(hist) < args.minFaceFrames:
                    continue

                frames, t_start = resample_visual_hist(hist, args.modelFps)
                if frames is None:
                    continue

                seg = get_audio_segment(t_start, frame_t)
                if len(seg) < 160:
                    continue

                if args.asdBackend == "talknce":
                    fused = infer_talknce_offline_style(
                        speaker=speaker,
                        audio_seg=seg,
                        video_frames=frames,
                        duration_set=duration_set,
                        model_fps=args.modelFps,
                    )
                elif args.asdBackend == "talknet_trt":
                    fused = speaker.infer_scores(seg, frames)
                else:
                    fused = infer_talknet_offline_style(
                        speaker=speaker,
                        audio_seg=seg,
                        video_frames=frames,
                        duration_set=duration_set,
                    )
                if fused is None or len(fused) == 0:
                    continue

                tr.score_hist.append(float(fused[-1]))
                tr.score = float(np.mean(tr.score_hist))
        ti1 = time.perf_counter()
        if latency_stats is not None:
            latency_stats["talknet_infer"].append((ti1 - ti0) * 1000.0)

    td0 = time.perf_counter()
    has_active = False
    top_tid = None
    ready_tracks = [tr for tr in tracks if score_is_ready(tr.score)]
    if args.onlyTopSpeaker and ready_tracks:
        top_tid = max(ready_tracks, key=lambda t: float(t.score)).tid

    score_samples = []
    active_speakers = []
    for tr in tracks:
        x1, y1, x2, y2 = [int(v) for v in get_smoothed_bbox(tr)]
        ready = score_is_ready(tr.score)
        if args.onlyTopSpeaker:
            active = ready and (tr.tid == top_tid) and (float(tr.score) >= args.scoreThres)
        else:
            active = ready and (float(tr.score) >= args.scoreThres)
        has_active = has_active or active
        if ready:
            score_samples.append(float(tr.score))
        if active:
            active_speakers.append((int(tr.tid), (x1, y1, x2, y2), float(tr.score)))
        color = (0, 255, 0) if active else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        txt = f"id={tr.tid} s={score_text(tr.score, precision=2)}"
        cv2.putText(frame, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    td1 = time.perf_counter()
    if latency_stats is not None:
        latency_stats["draw_overlay"].append((td1 - td0) * 1000.0)
        latency_stats["process_frame_total"].append((td1 - t_pf0) * 1000.0)

    return frame, tracks, next_tid, has_active, score_samples, active_speakers


def run_online_file(args, detector, speaker):
    video_path = args.videoPath
    if not video_path:
        cands = glob.glob(os.path.join(args.videoFolder, args.videoName + ".*"))
        if not cands:
            raise RuntimeError("input video not found")
        video_path = cands[0]

    save_dir = os.path.join(args.videoFolder, args.videoName + "_online")
    os.makedirs(save_dir, exist_ok=True)

    wav_path = os.path.join(save_dir, "audio.wav")
    extract_audio_wav(video_path, wav_path, sr=AUDIO_SR)
    _, audio = wavfile.read(wav_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(save_dir, "video_out.avi")
    vout = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    window_frames = max(int(round(args.windowSec * fps)), 1)
    infer_stride_frames = max(int(round(args.inferStrideSec * fps)), 1)
    duration_set = [int(x) for x in args.durationSet.split(",") if x.strip()]
    if not duration_set:
        duration_set = [1]

    tracks = []
    next_tid = 0
    frame_idx = -1
    t0 = time.perf_counter()
    active_frame_count = 0
    score_all = []
    last_active_print_ts = {}

    def get_audio_seg(t_start, t_end):
        a0 = int(round(t_start * AUDIO_SR))
        a1 = int(round(t_end * AUDIO_SR))
        a0 = max(0, min(a0, len(audio)))
        a1 = max(a0 + 1, min(a1, len(audio)))
        return audio[a0:a1]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame_t = frame_idx / fps

        if args.maxSeconds > 0 and frame_t >= args.maxSeconds:
            break

        run_infer = (frame_idx % infer_stride_frames == 0)
        frame, tracks, next_tid, has_active, score_samples, active_speakers = process_frame(
            frame=frame,
            frame_idx=frame_idx,
            frame_t=frame_t,
            tracks=tracks,
            next_tid=next_tid,
            detector=detector,
            speaker=speaker,
            duration_set=duration_set,
            args=args,
            window_hist_len=max(window_frames * 3, 200),
            get_audio_segment=get_audio_seg,
            run_infer=run_infer,
        )

        if has_active:
            active_frame_count += 1
        score_all.extend(score_samples)
        for tid, bbox, sc in active_speakers:
            prev_t = last_active_print_ts.get(tid, -1e9)
            if (frame_t - prev_t) >= 1.0:
                print(f"[active][file] frame={frame_idx} t={frame_t:.3f}s id={tid} bbox={bbox} score={sc:.3f}")
                last_active_print_ts[tid] = frame_t
        vout.write(frame)

    cap.release()
    vout.release()

    elapsed = time.perf_counter() - t0
    total_frames = max(frame_idx + 1, 1)
    print(f"done. frames={total_frames}, elapsed={elapsed:.2f}s, avg_fps={total_frames/max(elapsed,1e-6):.2f}")
    if score_all:
        arr = np.asarray(score_all, dtype=np.float32)
        print(f"score stats: mean={arr.mean():.3f}, p50={np.percentile(arr,50):.3f}, p95={np.percentile(arr,95):.3f}")
    print(f"active frame ratio (score>={args.scoreThres:.2f}): {active_frame_count/total_frames:.3f}")
    print(f"output: {out_path}")


def _open_camera_source(args):
    def _extract_video_index(src):
        if isinstance(src, str) and src.startswith("/dev/video"):
            suffix = src[len("/dev/video") :]
            if suffix.isdigit():
                return int(suffix)
        return None

    def _try_open_camera(src):
        attempts = []
        if isinstance(src, str) and src.startswith("/dev/"):
            attempts.append((src, cv2.CAP_V4L2))
            attempts.append((src, None))
            dev_index = _extract_video_index(src)
            if dev_index is not None:
                attempts.append((dev_index, cv2.CAP_V4L2))
                attempts.append((dev_index, None))
        else:
            attempts.append((src, None))
            if isinstance(src, int):
                attempts.append((src, cv2.CAP_V4L2))

        for open_src, backend in attempts:
            if backend is None:
                cap = cv2.VideoCapture(open_src)
            else:
                cap = cv2.VideoCapture(open_src, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, _ = cap.read()
            if ok:
                return cap, open_src
            cap.release()
        return None, None

    candidates = []
    explicit_camera_path = str(getattr(args, "cameraPath", "")).strip()
    if explicit_camera_path:
        candidates.append(explicit_camera_path)
    else:
        candidates.append(int(args.cameraId))

    probe = [x.strip() for x in str(getattr(args, "cameraProbeList", "")).split(",") if x.strip()]
    for tok in probe:
        if tok.startswith("/dev/"):
            candidates.append(tok)
        else:
            try:
                candidates.append(int(tok))
            except Exception:
                candidates.append(tok)

    seen = set()
    uniq = []
    for c in candidates:
        k = f"{type(c).__name__}:{c}"
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)

    for src in uniq:
        cap, selected = _try_open_camera(src)
        if cap is not None:
            print(f"camera source selected: {selected}")
            return cap, selected

    raise RuntimeError(f"failed to open any camera source. tried={uniq}")


def run_online_camera(args, detector, speaker):
    use_arecord = bool(getattr(args, "micAlsaCard", ""))
    noise_profile_path = ""
    selected_audio_desc = ""
    model_input_recorder = MonoWavRecorder(args.micInputDumpWav, AUDIO_SR) if args.micInputDumpWav else None
    if not use_arecord:
        try:
            import sounddevice as sd
        except Exception as e:
            raise RuntimeError(
                "Camera mode needs sounddevice for microphone capture. "
                "Install with: pip install sounddevice"
            ) from e

        selected_audio_device, selected_audio_desc = choose_sounddevice_input(sd, args)
        capture_sr = _pick_sounddevice_capture_rate(sd, selected_audio_device)
        if capture_sr != AUDIO_SR:
            selected_audio_desc += f" capture={capture_sr}Hz->model={AUDIO_SR}Hz"
        print(f"[audio] sounddevice input selected: {selected_audio_desc}")
    else:
        if args.micDenoise and args.micDenoiseNoiseFile:
            noise_profile_path = str(Path(args.micDenoiseNoiseFile).expanduser())
            profile_file = Path(noise_profile_path)
            if profile_file.exists():
                print(f"[audio] reuse saved noise profile: {profile_file}")
            elif args.micDenoiseNoiseSec > 0:
                print(
                    f"[audio] capturing noise profile for {args.micDenoiseNoiseSec:.1f}s "
                    f"from plughw:{args.micAlsaCard},0 ..."
                )
                noise_pcm = record_raw_alsa_block(
                    card_name=args.micAlsaCard,
                    sr=AUDIO_SR,
                    channels=args.micAlsaChannels,
                    duration_sec=args.micDenoiseNoiseSec,
                )
                reducer = FanNoiseReducer(
                    sr=AUDIO_SR,
                    channels=args.micAlsaChannels,
                    low_hz=args.micDenoiseLowHz,
                    high_hz=args.micDenoiseHighHz,
                    strength=args.micDenoiseStrength,
                    warmup_sec=args.micDenoiseWarmupSec,
                )
                reducer.fit_noise_profile(noise_pcm)
                reducer.save_noise_profile(profile_file)
                print(f"[audio] saved noise profile: {profile_file}")

        selected_audio_desc = (
            f"arecord:plughw:{args.micAlsaCard},0 ch={int(args.micAlsaChannels)} "
            f"mode={args.micAlsaChannelMode}"
        )
        if args.micDenoise:
            selected_audio_desc += (
                f" denoise=on[{args.micDenoiseLowHz:.0f}-{args.micDenoiseHighHz:.0f}Hz"
                f",strength={args.micDenoiseStrength:.2f}]"
            )
        print(f"[audio] arecord input selected: {selected_audio_desc}")
        if args.micRecordDir:
            print(f"[audio] raw split-channel record dir: {args.micRecordDir}")

    cap, selected_cam = _open_camera_source(args)

    if args.camWidth > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.camWidth))
    if args.camHeight > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.camHeight))
    if args.camFps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(args.camFps))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = float(args.camFps if args.camFps > 0 else 30.0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    robot_ctrl = RobotHeadController(args, width, height)
    http_streamer = None
    if args.httpStream:
        http_streamer = HttpFrameStreamer(
            host=args.httpHost,
            port=args.httpPort,
            jpeg_quality=args.httpJpegQuality,
        )
        http_streamer.start()

    out_path = ""
    vout = None
    if args.saveLiveVideo:
        out_path = args.saveLiveVideo
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        vout = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    duration_set = [int(x) for x in args.durationSet.split(",") if x.strip()]
    if not duration_set:
        duration_set = [1]

    window_frames = max(int(round(args.windowSec * fps)), 1)
    window_hist_len = max(window_frames * 4, 300)

    audio_buf = RealtimeAudioBuffer(sr=AUDIO_SR, max_seconds=max(10.0, args.audioBufferSec))
    audio_time_base = {"offset": None}

    if use_arecord:
        stream = ArecordAudioStream(
            card_name=args.micAlsaCard,
            channels=args.micAlsaChannels,
            sr=AUDIO_SR,
            blocksize=max(256, int(args.audioBlockSize)),
            audio_buf=audio_buf,
            channel_mode=args.micAlsaChannelMode,
            denoise=args.micDenoise,
            denoise_low_hz=args.micDenoiseLowHz,
            denoise_high_hz=args.micDenoiseHighHz,
            denoise_strength=args.micDenoiseStrength,
            denoise_warmup_sec=args.micDenoiseWarmupSec,
            noise_profile_path=noise_profile_path,
            record_dir=args.micRecordDir,
            record_prefix=args.micRecordPrefix,
            save_denoised_mono=args.micRecordDenoisedMono,
            mono_recorder=model_input_recorder,
        )
    else:

        def audio_callback(indata, frames, time_info, status):
            _ = status
            if audio_time_base["offset"] is None:
                audio_time_base["offset"] = float(time_info.inputBufferAdcTime)
            t0_sec = float(time_info.inputBufferAdcTime) - audio_time_base["offset"]
            t1_sec = t0_sec + float(frames) / float(capture_sr)
            mono = np.asarray(indata[:, 0], dtype=np.float32)
            if capture_sr != AUDIO_SR:
                mono = _resample_audio_float(mono, capture_sr, AUDIO_SR)
            if model_input_recorder is not None:
                model_input_recorder.write(mono)
            audio_buf.push_float(mono, t0_sec=t0_sec, t1_sec=t1_sec)

        capture_blocksize = max(256, int(round(args.audioBlockSize * capture_sr / float(AUDIO_SR))))
        stream = sd.InputStream(
            samplerate=capture_sr,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            blocksize=capture_blocksize,
            device=selected_audio_device,
        )

    tracks = []
    next_tid = 0
    frame_idx = -1
    active_frame_count = 0
    score_all = []
    last_active_print_ts = {}

    hold_tid = None
    hold_start_t = 0.0
    hold_bbox = None
    hold_sent = False

    latency_stats = {
        "capture_read": collections.deque(maxlen=max(200, args.latencyPrintEvery * 4)),
        "face_detect_track": collections.deque(maxlen=max(200, args.latencyPrintEvery * 4)),
        "crop_tracks": collections.deque(maxlen=max(200, args.latencyPrintEvery * 4)),
        "talknet_infer": collections.deque(maxlen=max(200, args.latencyPrintEvery * 4)),
        "draw_overlay": collections.deque(maxlen=max(200, args.latencyPrintEvery * 4)),
        "display": collections.deque(maxlen=max(200, args.latencyPrintEvery * 4)),
        "loop_total": collections.deque(maxlen=max(200, args.latencyPrintEvery * 4)),
        "process_frame_total": collections.deque(maxlen=max(200, args.latencyPrintEvery * 4)),
    }

    t0 = time.perf_counter()
    last_infer_t = -1e9
    last_audio_debug_t = -1e9
    last_face_position_t = -1e9
    next_audio_cue_t = float(args.audioDebugCueAtSec)
    cue_count = 0

    stream.start()
    try:
        while True:
            t_loop0 = time.perf_counter()
            t_cap0 = time.perf_counter()
            ok, frame = cap.read()
            t_cap1 = time.perf_counter()
            latency_stats["capture_read"].append((t_cap1 - t_cap0) * 1000.0)
            if not ok:
                break
            frame_idx += 1
            frame_t = time.perf_counter() - t0

            if args.flip:
                frame = cv2.flip(frame, 1)

            if args.maxSeconds > 0 and frame_t >= args.maxSeconds:
                break

            run_infer = (frame_t - last_infer_t) >= args.inferStrideSec
            if run_infer:
                last_infer_t = frame_t

            if args.audioDebug and (frame_t - last_audio_debug_t) >= args.audioDebugEverySec:
                seg = audio_buf.get_segment(max(0.0, frame_t - args.audioDebugWindowSec), frame_t)
                rms, peak, mean_abs = summarize_audio_segment(seg)
                print(
                    f"[audio][debug] t={frame_t:.3f}s dev={selected_audio_desc} "
                    f"window={args.audioDebugWindowSec:.2f}s rms={rms:.6f} peak={peak:.6f} mean_abs={mean_abs:.6f}"
                )
                last_audio_debug_t = frame_t

            if (
                args.audioDebug
                and cue_count < args.audioDebugCueCount
                and next_audio_cue_t >= 0.0
                and frame_t >= next_audio_cue_t
            ):
                print(
                    f"[audio][cue] t={frame_t:.3f}s 现在请连续说话 {args.audioDebugCueSpeakSec:.1f} 秒"
                )
                cue_count += 1
                next_audio_cue_t += max(args.audioDebugCueIntervalSec, args.audioDebugCueSpeakSec)

            frame, tracks, next_tid, has_active, score_samples, active_speakers = process_frame(
                frame=frame,
                frame_idx=frame_idx,
                frame_t=frame_t,
                tracks=tracks,
                next_tid=next_tid,
                detector=detector,
                speaker=speaker,
                duration_set=duration_set,
                args=args,
                window_hist_len=window_hist_len,
                get_audio_segment=audio_buf.get_segment,
                run_infer=run_infer,
                latency_stats=latency_stats,
            )

            if has_active:
                active_frame_count += 1
            score_all.extend(score_samples)
            for tid, bbox, sc in active_speakers:
                prev_t = last_active_print_ts.get(tid, -1e9)
                if (frame_t - prev_t) >= 1.0:
                    print(f"[active][camera] t={frame_t:.3f}s id={tid} bbox={bbox} score={sc:.3f}")
                    last_active_print_ts[tid] = frame_t

            if args.facePositionLog and (frame_t - last_face_position_t) >= args.facePositionEverySec:
                top_tid = _top_track_tid(tracks, args.onlyTopSpeaker)
                face_records = collect_face_position_records(tracks, args, width, height, top_tid)
                face_records = filter_face_position_records(face_records, args.facePositionMode)
                emit_face_position_log(face_records, frame_t, args)
                last_face_position_t = frame_t

            if args.enableRobotHeadTrack and robot_ctrl is not None:
                if len(active_speakers) == 1:
                    tid, bbox, sc = active_speakers[0]
                    if hold_tid != tid:
                        hold_tid = tid
                        hold_start_t = frame_t
                        hold_bbox = bbox
                        hold_sent = False
                    else:
                        hold_bbox = bbox
                        if (not hold_sent) and (frame_t - hold_start_t >= args.robotActiveHoldSec):
                            ok_cmd, yaw_cmd, pitch_cmd, ex, ey = robot_ctrl.publish_head_center_cmd(hold_bbox)
                            if ok_cmd:
                                yaw_deg = math.degrees(yaw_cmd)
                                pitch_deg = math.degrees(pitch_cmd)
                                yaw_txt = f"右转 {abs(yaw_deg):.1f}°" if yaw_deg > 0 else (f"左转 {abs(yaw_deg):.1f}°" if yaw_deg < 0 else "水平不转")
                                pitch_txt = f"抬头 {abs(pitch_deg):.1f}°" if pitch_deg > 0 else (f"低头 {abs(pitch_deg):.1f}°" if pitch_deg < 0 else "俯仰不变")
                                print(
                                    f"[robot][head] trigger id={tid} hold={frame_t-hold_start_t:.2f}s | 控制: {yaw_txt}, {pitch_txt} "
                                    f"| yaw={yaw_cmd:.3f}rad pitch={pitch_cmd:.3f}rad | bbox={hold_bbox}"
                                )
                            hold_sent = hold_sent or ok_cmd
                else:
                    hold_tid = None
                    hold_start_t = 0.0
                    hold_bbox = None
                    hold_sent = False

            if vout is not None:
                vout.write(frame)

            if http_streamer is not None:
                http_streamer.update(frame)

            t_disp0 = time.perf_counter()
            if args.showWindow:
                cv2.imshow("TalkNet Realtime", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
            t_disp1 = time.perf_counter()
            latency_stats["display"].append((t_disp1 - t_disp0) * 1000.0)

            t_loop1 = time.perf_counter()
            latency_stats["loop_total"].append((t_loop1 - t_loop0) * 1000.0)

            if args.latencyPrintEvery > 0 and (frame_idx + 1) % args.latencyPrintEvery == 0 and (frame_idx + 1) >= args.latencyWarmupFrames:
                _print_latency_report(latency_stats)

    finally:
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        if model_input_recorder is not None:
            try:
                model_input_recorder.close()
            except Exception:
                pass
        cap.release()
        if vout is not None:
            vout.release()
        if args.showWindow:
            cv2.destroyAllWindows()
        if robot_ctrl is not None:
            robot_ctrl.shutdown()
        if http_streamer is not None:
            http_streamer.stop()

    elapsed = time.perf_counter() - t0
    total_frames = max(frame_idx + 1, 1)
    print(f"done. frames={total_frames}, elapsed={elapsed:.2f}s, avg_fps={total_frames/max(elapsed,1e-6):.2f}")
    if score_all:
        arr = np.asarray(score_all, dtype=np.float32)
        print(f"score stats: mean={arr.mean():.3f}, p50={np.percentile(arr,50):.3f}, p95={np.percentile(arr,95):.3f}")
    print(f"active frame ratio (score>={args.scoreThres:.2f}): {active_frame_count/total_frames:.3f}")
    _print_latency_report(latency_stats)
    print(f"output: {out_path if out_path else 'N/A (live display only)'}")


def run_online(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by current TalkNet path")

    detector = build_face_detector(args)

    if args.asdBackend == "talknce":
        speaker, ckpt = _build_talknce_backend(args)
        print(f"TalkNCE backend loaded: {ckpt}")
    elif args.asdBackend == "talknet_trt":
        from trt_support import TalkNetTRTBackend

        engine_path = Path(args.talknetTrtEngine).expanduser()
        if not engine_path.is_absolute():
            engine_path = (Path(__file__).resolve().parent / engine_path).resolve()
        if not engine_path.exists():
            raise RuntimeError(f"TalkNet TRT engine not found: {engine_path}")
        speaker = TalkNetTRTBackend(str(engine_path), audio_sr=AUDIO_SR)
        print(f"TalkNet TRT backend loaded: {engine_path}")
    else:
        speaker = talkNet()
        speaker.loadParameters(args.pretrainModel)
        speaker.eval()

    if args.camera:
        run_online_camera(args, detector, speaker)
    else:
        run_online_file(args, detector, speaker)


def main():
    parser = argparse.ArgumentParser("Online-style TalkNet")

    parser.add_argument("--camera", action="store_true", help="Use webcam + microphone realtime mode")
    parser.add_argument("--cameraId", type=int, default=0)
    parser.add_argument("--cameraPath", type=str, default="", help="Optional camera device path, e.g. /dev/video2")
    parser.add_argument("--cameraProbeList", type=str, default="/dev/video2,/dev/video4,0,1,2,3,4,5", help="Fallback camera candidates")
    parser.add_argument("--micDevice", type=int, default=-1, help="sounddevice input id, -1 for default")
    parser.add_argument("--micAlsaCard", type=str, default="", help="Temporarily redirect ALSA default input to this card name, e.g. AIUIUSBMC")
    parser.add_argument("--micAlsaChannels", type=int, default=4, help="Channel count for temporary ALSA default input")
    parser.add_argument(
        "--micAlsaChannelMode",
        type=str,
        default="max_energy",
        choices=["max_energy", "first", "mean"],
        help="How to collapse multi-channel arecord input to mono for ASD",
    )
    parser.add_argument("--micDenoise", action="store_true", help="Enable lightweight fan-noise reduction on arecord multi-mic input")
    parser.add_argument("--micDenoiseLowHz", type=float, default=120.0, help="Lower FFT passband edge for arecord denoise")
    parser.add_argument("--micDenoiseHighHz", type=float, default=4200.0, help="Upper FFT passband edge for arecord denoise")
    parser.add_argument("--micDenoiseStrength", type=float, default=1.2, help="Spectral subtraction strength for arecord denoise")
    parser.add_argument("--micDenoiseWarmupSec", type=float, default=0.5, help="Initial seconds used to estimate stationary fan noise")
    parser.add_argument("--micDenoiseNoiseSec", type=float, default=0.0, help="If >0, capture this many seconds of fan-only noise before online start")
    parser.add_argument("--micDenoiseNoiseFile", type=str, default="", help="Optional .npz noise profile path; reused if already present")
    parser.add_argument("--micRecordDir", type=str, default="", help="Optional directory to save split raw mic wav files when using arecord")
    parser.add_argument("--micRecordPrefix", type=str, default="mic", help="Filename prefix for split raw mic recordings")
    parser.add_argument("--micRecordDenoisedMono", action="store_true", help="Also save the denoised mono stream used for ASD")
    parser.add_argument("--micInputDumpWav", type=str, default="", help="Optional wav path to dump the final mono audio actually fed into the ASD model")
    parser.add_argument("--showWindow", action="store_true", help="Show realtime window (press q to quit)")
    parser.add_argument("--httpStream", action="store_true", help="Expose processed realtime video via MJPEG over HTTP")
    parser.add_argument("--httpHost", type=str, default="0.0.0.0", help="HTTP stream listen host")
    parser.add_argument("--httpPort", type=int, default=8082, help="HTTP stream listen port")
    parser.add_argument("--httpJpegQuality", type=int, default=80, help="HTTP MJPEG JPEG quality")
    parser.add_argument("--saveLiveVideo", type=str, default="", help="Optional output path for camera mode")
    parser.add_argument("--audioBufferSec", type=float, default=30.0)
    parser.add_argument("--audioBlockSize", type=int, default=1024)
    parser.add_argument("--camWidth", type=int, default=0)
    parser.add_argument("--camHeight", type=int, default=0)
    parser.add_argument("--camFps", type=float, default=0.0)
    parser.add_argument("--flip", action="store_true", help="Mirror camera image horizontally")
    parser.add_argument(
        "--facePositionLog",
        action="store_true",
        help="Print tracked face position (bbox/center/offset) to stdout for downstream use",
    )
    parser.add_argument(
        "--facePositionEverySec",
        type=float,
        default=0.1,
        help="Min interval between face position log lines",
    )
    parser.add_argument(
        "--facePositionFormat",
        type=str,
        default="text",
        choices=["text", "json"],
        help="stdout format for --facePositionLog",
    )
    parser.add_argument(
        "--facePositionMode",
        type=str,
        default="top",
        choices=["top", "active", "all"],
        help="top=main face (highest ASD score); active=speaking only; all=all tracks",
    )
    parser.add_argument("--latencyPrintEvery", type=int, default=50, help="Print realtime stage latency every N frames; <=0 to disable")
    parser.add_argument("--latencyWarmupFrames", type=int, default=20, help="Start latency report after this many frames")

    parser.add_argument("--enableRobotHeadTrack", action="store_true", help="Enable ROS2 head tracking command publish")
    parser.add_argument("--robotControlMode", type=str, default="fd_dm_pos", choices=["rosci", "fd_dm_pos", "fd_mit", "fd_head_topic"], help="Robot control message backend")
    parser.add_argument("--robotActiveHoldSec", type=float, default=3.0, help="Require exactly one active speaker for this long before triggering")
    parser.add_argument("--robotCmdRepeatSec", type=float, default=0.1, help="Repeat robot command every N seconds while target stays locked")
    parser.add_argument("--robotCmdTopic", type=str, default="/rosci_head_waist_command")
    parser.add_argument("--robotMsgModule", type=str, default="rosci_robot_message.msg", help="Python message module path")
    parser.add_argument("--robotCmdMsg", type=str, default="HeadWaistCommand")
    parser.add_argument("--robotJointMsg", type=str, default="JointCommand")
    parser.add_argument("--fdPosTopic", type=str, default="/fd_robot/dm_motor/position_cmd")
    parser.add_argument("--fdMsgModule", type=str, default="fd_msgs.msg")
    parser.add_argument("--fdPosMsg", type=str, default="DmPosControlMsg")
    parser.add_argument("--fdEnableTopic", type=str, default="/fd_robot/dm_motor/enable_cmd")
    parser.add_argument("--fdEnableMsg", type=str, default="EnableMsg")
    parser.add_argument("--fdMitTopic", type=str, default="/fd_robot/dm_motor/mit_cmd")
    parser.add_argument("--fdMitMsg", type=str, default="DmMITControlMsg")
    parser.add_argument("--fdHeadTopicCmd", type=str, default="/fd_robot/head_control/cmd_angles")
    parser.add_argument("--fdYawServoId", type=int, default=-1, help="Yaw servo id for fd_dm_pos mode")
    parser.add_argument("--fdPitchServoId", type=int, default=-1, help="Pitch servo id for fd_dm_pos mode")
    parser.add_argument("--fdCmdVel", type=float, default=0.8, help="Command velocity for fd_dm_pos mode")
    parser.add_argument("--fdMitVel", type=float, default=0.0, help="Target velocity for fd_mit mode")
    parser.add_argument("--fdMitKp", type=float, default=2.0, help="Position gain for fd_mit mode")
    parser.add_argument("--fdMitKd", type=float, default=0.15, help="Velocity gain for fd_mit mode")
    parser.add_argument("--fdMitTorque", type=float, default=0.0, help="Feedforward torque for fd_mit mode")
    parser.add_argument("--fdYawSign", type=float, default=1.0, help="Sign correction for yaw command")
    parser.add_argument("--fdPitchSign", type=float, default=1.0, help="Sign correction for pitch command")
    parser.add_argument("--fdHeadUpTopic", type=str, default="/fd_robot/head_up_move")
    parser.add_argument("--fdHeadDownTopic", type=str, default="/fd_robot/head_down_move")
    parser.add_argument("--robotYawGain", type=float, default=0.8, help="yaw command gain from normalized x error")
    parser.add_argument("--robotPitchGain", type=float, default=0.6, help="pitch command gain from normalized y error")
    parser.add_argument("--robotMaxYawRad", type=float, default=0.8)
    parser.add_argument("--robotMaxPitchRad", type=float, default=0.6)
    parser.add_argument("--robotVelGain", type=float, default=0.6)
    parser.add_argument("--robotAccGain", type=float, default=0.6)
    parser.add_argument("--robotFrameId", type=str, default="asd_tracker")

    parser.add_argument("--videoName", type=str, default="001")
    parser.add_argument("--videoFolder", type=str, default="demo")
    parser.add_argument("--videoPath", type=str, default="")
    parser.add_argument("--pretrainModel", type=str, default="pretrain_TalkSet.model")
    parser.add_argument("--asdBackend", type=str, default="talknet", choices=["talknet", "talknet_trt", "talknce"], help="ASD backend")
    parser.add_argument("--talknceRoot", type=str, default="TalkNCE-main", help="TalkNCE repo root")
    parser.add_argument("--talknceCfg", type=str, default="", help="TalkNCE config yaml path")
    parser.add_argument("--talknceCkpt", type=str, default="", help="TalkNCE checkpoint .model path")
    parser.add_argument("--talknetTrtEngine", type=str, default="", help="TensorRT engine path for TalkNet ASD backend")

    parser.add_argument("--windowSec", type=float, default=1.0, help="Sliding window seconds used for ASD")
    parser.add_argument("--inferStrideSec", type=float, default=0.2, help="Run ASD update every N seconds")
    parser.add_argument("--modelFps", type=float, default=25.0, help="Target visual FPS for TalkNet frontend")
    parser.add_argument("--minFaceFrames", type=int, default=6, help="Minimum faces in window before ASD")
    parser.add_argument("--scoreThres", type=float, default=0.0, help="Speaking threshold on smoothed score")
    parser.add_argument("--onlyTopSpeaker", action="store_true", help="Only mark the top-scoring face per frame as speaking")
    parser.add_argument("--durationSet", type=str, default="1,1,1,2,2,2,3,3,4,5,6", help="Offline-style duration ensemble")

    parser.add_argument("--facedetScale", type=float, default=0.25)
    parser.add_argument(
        "--facedetScales",
        type=str,
        default="",
        help="Optional comma-separated multi-scale face detection list, e.g. 0.5,1.0,1.5",
    )
    parser.add_argument("--detConf", type=float, default=0.9)
    parser.add_argument(
        "--faceDetectorBackend",
        type=str,
        default="s3fd",
        choices=["s3fd", "s3fd_trt", "mediapipe"],
        help="Face detector backend for online pipeline",
    )
    parser.add_argument(
        "--s3fdTrtEngineSpecs",
        type=str,
        default="",
        help="Comma-separated TensorRT S3FD engines, e.g. 0.5=engines/s3fd_960x540_s0.5.engine,1.0=engines/s3fd_1920x1080_s1.0.engine",
    )
    parser.add_argument(
        "--mpModelSelection",
        type=int,
        default=0,
        choices=[0, 1],
        help="MediaPipe face detector model_selection (0 short-range, 1 full-range)",
    )
    parser.add_argument(
        "--mpModelPath",
        type=str,
        default="",
        help="Optional MediaPipe face model path (.tflite) for tasks backend",
    )
    parser.add_argument(
        "--mpRuntime",
        type=str,
        default="auto",
        choices=["auto", "legacy", "tasks"],
        help="MediaPipe runtime selection. Use tasks for explicit CPU/GPU delegate support.",
    )
    parser.add_argument(
        "--mpDelegate",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="MediaPipe Tasks delegate selection. GPU is only used with tasks runtime.",
    )
    parser.add_argument("--iouThres", type=float, default=0.5)
    parser.add_argument("--maxMissedFrames", type=int, default=10)
    parser.add_argument("--cropScale", type=float, default=0.40)

    parser.add_argument("--maxSeconds", type=float, default=0.0, help="0 means full stream")
    parser.add_argument("--audioDebug", action="store_true", help="Print realtime microphone energy stats")
    parser.add_argument("--audioDebugEverySec", type=float, default=1.0, help="Audio debug print interval in seconds")
    parser.add_argument("--audioDebugWindowSec", type=float, default=1.0, help="Audio debug analysis window in seconds")
    parser.add_argument("--audioDebugCueAtSec", type=float, default=3.0, help="First speak-now cue time in seconds")
    parser.add_argument("--audioDebugCueIntervalSec", type=float, default=6.0, help="Seconds between speak-now cues")
    parser.add_argument("--audioDebugCueSpeakSec", type=float, default=3.0, help="Suggested speaking duration for each cue")
    parser.add_argument("--audioDebugCueCount", type=int, default=3, help="How many speak-now cues to print")

    args = parser.parse_args()
    run_online(args)


if __name__ == "__main__":
    main()
