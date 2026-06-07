#!/usr/bin/env python3
"""Mock Layer A demo: synthetic video + optional webcam."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running as script without install
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from social_vla.pipeline.layer_a import (
    LayerAConfig,
    LayerAPipeline,
    default_yolo_weights,
    open_video_source,
)

SAMPLE_RATE = 16000


def synthetic_frame(tick: int, w: int = 640, h: int = 480) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (32, 32, 40)
    cx = int(w * (0.5 + 0.1 * np.sin(tick * 0.08)))
    cy = int(h * 0.45)
    bw, bh = int(w * 0.2), int(h * 0.5)
    cv2.rectangle(frame, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), (180, 140, 120), -1)
    cv2.putText(frame, f"mock tick {tick}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def synthetic_pcm_chunk(tick: int, hz: float, vad_active: bool) -> np.ndarray:
    """~100ms PCM for TalkNet MFCC when VAD is on."""
    n = max(1, int(SAMPLE_RATE / hz))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    if not vad_active:
        return np.zeros(n, dtype=np.int16)
    freq = 180.0 + 40.0 * np.sin(tick * 0.12)
    amp = 0.25 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 8.0 * t))
    wave = amp * np.sin(2.0 * np.pi * freq * t)
    return (wave * 32767.0).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description="SocialVLA Layer A mock demo")
    parser.add_argument("--video", default="", help="Video path or camera index (default: synthetic)")
    parser.add_argument("--ticks", type=int, default=120, help="Max ticks (synthetic mode)")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--real-weights",
        action="store_true",
        help="Load TalkNet/LAM checkpoints (detector still mock on synthetic frames)",
    )
    parser.add_argument(
        "--real-detector",
        action="store_true",
        help="Use YOLO on synthetic frames (usually useless; prefer live_demo.py)",
    )
    parser.add_argument(
        "--mock-detector",
        action="store_true",
        help="Force mock bbox even when --video is set",
    )
    parser.add_argument("--vad-mock", action="store_true", help="Toggle mock VAD during demo")
    parser.add_argument("--mock-audio", action="store_true", help="Feed synthetic PCM when VAD on (for TalkNet)")
    args = parser.parse_args()

    synthetic = not args.video
    if synthetic:
        mock_detector = not args.real_detector
    else:
        mock_detector = args.mock_detector

    cfg = LayerAConfig(
        hz=args.hz,
        mock_detector=mock_detector,
        mock_models=not args.real_weights,
        device=args.device,
        yolo_model=default_yolo_weights(),
        vad_mock=args.vad_mock,
    )
    pipe = LayerAPipeline(cfg)

    print(
        f"mode: detector={'mock' if cfg.mock_detector else 'yolo'} | "
        f"models={'mock' if cfg.mock_models else 'real'} | "
        f"source={'synthetic' if synthetic else args.video}"
    )
    if synthetic and args.real_detector:
        print("warn: --real-detector on synthetic frames usually detects nothing (not a real person)")
    if not synthetic and not mock_detector:
        print("tip: for live person detection with preview, use: python social_vla/pipeline/live_demo.py")

    if args.video:
        cap = open_video_source(int(args.video) if args.video.isdigit() else args.video)
        pipe.run_period(cap, max_ticks=None if args.ticks == 120 else args.ticks)
        cap.release()
        return

    period = 1.0 / cfg.hz
    for tick in range(args.ticks):
        t0 = time.time()
        frame = synthetic_frame(tick)
        vad_on = (tick // 25) % 2 == 1 if cfg.vad_mock else False
        if cfg.vad_mock:
            pipe.set_vad(vad_on, t0)
        if args.mock_audio or cfg.vad_mock:
            pipe.push_audio_pcm(synthetic_pcm_chunk(tick, cfg.hz, vad_on))
        out = pipe.process_frame(frame, t0)
        pipe._print_tick(out)
        time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
