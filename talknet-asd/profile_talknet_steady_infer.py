#!/usr/bin/env python3
"""Steady-state TalkNet inference profiler (model loaded once).

Example:
  source .venv-talknet/bin/activate
  python profile_talknet_steady_infer.py \
    --videoFolder demo --videoName 111 \
    --iters 10 --warmup-iters 2
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import python_speech_features
import torch
from scipy.io import wavfile

from talkNet import talkNet


@dataclass
class Sample:
    t: float
    rss_mb: float
    gpu_mb: float


def read_rss_mb(pid: int) -> float:
    p = Path(f"/proc/{pid}/status")
    if not p.exists():
        return 0.0
    try:
        for line in p.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except Exception:
        return 0.0
    return 0.0


def read_gpu_mb(pid: int) -> float:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            return 0.0
        for line in r.stdout.splitlines():
            cols = [x.strip() for x in line.split(",")]
            if len(cols) >= 2 and int(cols[0]) == int(pid):
                return float(cols[1])
    except Exception:
        return 0.0
    return 0.0


def load_clip_features(clip_avi: str, pycrop_path: str) -> tuple[np.ndarray, np.ndarray]:
    file_name = os.path.splitext(os.path.basename(clip_avi))[0]
    wav_path = os.path.join(pycrop_path, file_name + ".wav")

    _, audio = wavfile.read(wav_path)
    audio_feature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

    cap = cv2.VideoCapture(clip_avi)
    video_feature = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (224, 224))
        face = face[int(112 - 56):int(112 + 56), int(112 - 56):int(112 + 56)]
        video_feature.append(face)
    cap.release()

    video_feature = np.array(video_feature)
    length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0] / 25)
    audio_feature = audio_feature[: int(round(length * 100)), :]
    video_feature = video_feature[: int(round(length * 25)), :, :]
    return audio_feature, video_feature


def infer_one_pass(model: talkNet, files: list[str], pycrop_path: str, duration_set: list[int]) -> dict[str, float]:
    t_pre0 = time.perf_counter()
    feats = [load_clip_features(f, pycrop_path) for f in files]
    t_pre1 = time.perf_counter()

    t_inf0 = time.perf_counter()
    with torch.no_grad():
        for audio_feature, video_feature in feats:
            length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0] / 25)
            for duration in duration_set:
                batch_size = int(math.ceil(length / duration))
                for i in range(batch_size):
                    input_a = torch.FloatTensor(audio_feature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(0).cuda()
                    input_v = torch.FloatTensor(video_feature[i * duration * 25:(i + 1) * duration * 25, :, :]).unsqueeze(0).cuda()
                    embed_a = model.model.forward_audio_frontend(input_a)
                    embed_v = model.model.forward_visual_frontend(input_v)
                    embed_a, embed_v = model.model.forward_cross_attention(embed_a, embed_v)
                    out = model.model.forward_audio_visual_backend(embed_a, embed_v)
                    _ = model.lossAV.forward(out, labels=None)
    torch.cuda.synchronize()
    t_inf1 = time.perf_counter()

    return {
        "preprocess_ms": (t_pre1 - t_pre0) * 1000.0,
        "inference_ms": (t_inf1 - t_inf0) * 1000.0,
        "total_ms": (t_inf1 - t_pre0) * 1000.0,
    }


def stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(vals)
    return {
        "mean": float(sum(s) / len(s)),
        "p50": float(statistics.median(s)),
        "p95": float(s[int(0.95 * (len(s) - 1))]),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser("Steady TalkNet inference profiler")
    ap.add_argument("--videoFolder", type=str, default="demo")
    ap.add_argument("--videoName", type=str, default="111")
    ap.add_argument("--pycrop-path", type=str, default="", help="If set, use this clip folder directly")
    ap.add_argument("--pretrainModel", type=str, default="pretrain_TalkSet.model")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup-iters", type=int, default=2)
    ap.add_argument("--sample-ms", type=int, default=20)
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    pycrop_path = args.pycrop_path or os.path.join(args.videoFolder, args.videoName, "pycrop")
    files = sorted(glob.glob(os.path.join(pycrop_path, "*.avi")))
    if not files:
        raise RuntimeError(f"No clip avi files found in {pycrop_path}. Run demoTalkNet once first.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this steady-state profiler (current TalkNet code uses .cuda()).")

    # Load model ONCE (not included in steady-state inference stats).
    t_load0 = time.perf_counter()
    model = talkNet()
    model.loadParameters(args.pretrainModel)
    model.eval()
    torch.cuda.synchronize()
    t_load1 = time.perf_counter()
    load_ms = (t_load1 - t_load0) * 1000.0

    duration_set = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]

    pid = os.getpid()
    samples: list[Sample] = []
    stop_evt = threading.Event()

    def sampler() -> None:
        while not stop_evt.is_set():
            samples.append(Sample(time.perf_counter(), read_rss_mb(pid), read_gpu_mb(pid)))
            time.sleep(max(args.sample_ms, 5) / 1000.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    iter_results = []
    t_all0 = time.perf_counter()
    for i in range(args.iters):
        r = infer_one_pass(model, files, pycrop_path, duration_set)
        iter_results.append(r)
        print(
            f"[Iter {i+1}/{args.iters}] "
            f"pre={r['preprocess_ms']:.1f} ms, inf={r['inference_ms']:.1f} ms, total={r['total_ms']:.1f} ms"
        )
    t_all1 = time.perf_counter()

    stop_evt.set()
    th.join(timeout=1.0)

    steady = iter_results[min(args.warmup_iters, len(iter_results)):]
    steady_pre = stats([x["preprocess_ms"] for x in steady])
    steady_inf = stats([x["inference_ms"] for x in steady])
    steady_total = stats([x["total_ms"] for x in steady])

    if samples:
        rss_peak = max(s.rss_mb for s in samples)
        rss_avg = sum(s.rss_mb for s in samples) / len(samples)
        gpu_peak = max(s.gpu_mb for s in samples)
        gpu_avg = sum(s.gpu_mb for s in samples) / len(samples)
    else:
        rss_peak = rss_avg = gpu_peak = gpu_avg = 0.0

    out = {
        "video_name": args.videoName,
        "pycrop_path": pycrop_path,
        "num_clips": len(files),
        "model_load_once_ms": load_ms,
        "iters": args.iters,
        "warmup_iters": args.warmup_iters,
        "iter_results_ms": iter_results,
        "steady_preprocess_ms": steady_pre,
        "steady_inference_ms": steady_inf,
        "steady_total_ms": steady_total,
        "process_memory_mb": {
            "rss_peak": rss_peak,
            "rss_avg": rss_avg,
            "gpu_peak": gpu_peak,
            "gpu_avg": gpu_avg,
        },
        "wall_total_ms_including_all_iters": (t_all1 - t_all0) * 1000.0,
    }

    out_path = Path(args.out_json) if args.out_json else Path(args.videoFolder) / args.videoName / "pywork" / "profile_talknet_steady_infer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n=== Model Load (once) ===")
    print(f"load_ms: {load_ms:.1f}")

    print("\n=== Steady-State (model kept loaded) ===")
    print("inference_only_ms:", json.dumps(steady_inf, ensure_ascii=False))
    print("preprocess_ms:", json.dumps(steady_pre, ensure_ascii=False))
    print("total_ms:", json.dumps(steady_total, ensure_ascii=False))

    print("\n=== Process Memory During Profiling (MB) ===")
    print(f"rss_peak={rss_peak:.1f}, rss_avg={rss_avg:.1f}, gpu_peak={gpu_peak:.1f}, gpu_avg={gpu_avg:.1f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
