#!/usr/bin/env python3
"""Steady-state profiler for online_demoTalkNet.py.

It runs online_demoTalkNet.py multiple times, skips warmup runs,
and reports steady-state latency + process RSS/GPU memory.

Example:
  source .venv-talknet/bin/activate
  python profile_online_demoTalkNet_steady.py \
    --videoFolder demo --videoName 111 \
    --runs 6 --warmup-runs 1 \
    --windowSec 0.4 --inferStrideSec 0.2 \
    --scoreThres 0.2 --onlyTopSpeaker
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DONE_RE = re.compile(r"done\. frames=(\d+), elapsed=([0-9.]+)s, avg_fps=([0-9.]+)")
SCORE_RE = re.compile(r"score stats: mean=([-0-9.]+), p50=([-0-9.]+), p95=([-0-9.]+)")
ACTIVE_RE = re.compile(r"active frame ratio \(score>=([-0-9.]+)\): ([0-9.]+)")


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
            if len(cols) >= 2 and cols[0].isdigit() and int(cols[0]) == int(pid):
                return float(cols[1])
    except Exception:
        return 0.0
    return 0.0


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


def run_once(args: argparse.Namespace, run_idx: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        args.script,
        "--videoFolder",
        args.videoFolder,
        "--videoName",
        args.videoName,
        "--windowSec",
        str(args.windowSec),
        "--inferStrideSec",
        str(args.inferStrideSec),
        "--scoreThres",
        str(args.scoreThres),
    ]
    if args.onlyTopSpeaker:
        cmd.append("--onlyTopSpeaker")
    if args.maxSeconds > 0:
        cmd += ["--maxSeconds", str(args.maxSeconds)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    samples: list[Sample] = []
    stop_evt = threading.Event()

    def sampler() -> None:
        while not stop_evt.is_set():
            if proc.poll() is not None:
                break
            samples.append(Sample(time.perf_counter(), read_rss_mb(proc.pid), read_gpu_mb(proc.pid)))
            time.sleep(max(args.sample_ms, 5) / 1000.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    out_lines: list[str] = []
    t0 = time.perf_counter()
    done_frames = 0
    done_elapsed = 0.0
    done_avg_fps = 0.0
    score_mean = score_p50 = score_p95 = 0.0
    score_thres = args.scoreThres
    active_ratio = 0.0

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        out_lines.append(line)
        m = DONE_RE.search(line)
        if m:
            done_frames = int(m.group(1))
            done_elapsed = float(m.group(2))
            done_avg_fps = float(m.group(3))
        m = SCORE_RE.search(line)
        if m:
            score_mean = float(m.group(1))
            score_p50 = float(m.group(2))
            score_p95 = float(m.group(3))
        m = ACTIVE_RE.search(line)
        if m:
            score_thres = float(m.group(1))
            active_ratio = float(m.group(2))

    rc = proc.wait()
    t1 = time.perf_counter()
    stop_evt.set()
    th.join(timeout=1.0)

    if samples:
        rss_peak = max(s.rss_mb for s in samples)
        rss_avg = sum(s.rss_mb for s in samples) / len(samples)
        gpu_peak = max(s.gpu_mb for s in samples)
        gpu_avg = sum(s.gpu_mb for s in samples) / len(samples)
    else:
        rss_peak = rss_avg = gpu_peak = gpu_avg = 0.0

    if rc != 0:
        tail = "\n".join(out_lines[-40:])
        raise RuntimeError(f"Run {run_idx} failed rc={rc}\n{tail}")

    wall_ms = (t1 - t0) * 1000.0
    per_frame_ms = 1000.0 / done_avg_fps if done_avg_fps > 0 else 0.0
    decision_latency_ms_est = args.windowSec * 1000.0 + args.inferStrideSec * 1000.0 + per_frame_ms

    return {
        "run": run_idx,
        "rc": rc,
        "wall_ms": wall_ms,
        "frames": done_frames,
        "elapsed_s_reported": done_elapsed,
        "avg_fps": done_avg_fps,
        "per_frame_latency_ms": per_frame_ms,
        "decision_latency_ms_est": decision_latency_ms_est,
        "score": {
            "mean": score_mean,
            "p50": score_p50,
            "p95": score_p95,
            "threshold": score_thres,
            "active_ratio": active_ratio,
        },
        "memory_mb": {
            "rss_peak": rss_peak,
            "rss_avg": rss_avg,
            "gpu_peak": gpu_peak,
            "gpu_avg": gpu_avg,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser("Steady profiler for online_demoTalkNet")
    ap.add_argument("--script", type=str, default="online_demoTalkNet.py")
    ap.add_argument("--videoFolder", type=str, default="demo")
    ap.add_argument("--videoName", type=str, default="111")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--sample-ms", type=int, default=20)

    ap.add_argument("--windowSec", type=float, default=0.4)
    ap.add_argument("--inferStrideSec", type=float, default=0.2)
    ap.add_argument("--scoreThres", type=float, default=0.2)
    ap.add_argument("--onlyTopSpeaker", action="store_true")
    ap.add_argument("--maxSeconds", type=float, default=0.0, help="0 means full video")

    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    results: list[dict[str, Any]] = []
    for i in range(1, args.runs + 1):
        print(f"[Run {i}/{args.runs}] start")
        r = run_once(args, i)
        results.append(r)
        print(
            f"[Run {i}/{args.runs}] "
            f"fps={r['avg_fps']:.2f}, frame_ms={r['per_frame_latency_ms']:.1f}, "
            f"decision_ms~{r['decision_latency_ms_est']:.1f}, "
            f"rss_peak={r['memory_mb']['rss_peak']:.1f}MB, gpu_peak={r['memory_mb']['gpu_peak']:.1f}MB"
        )

    steady = results[min(args.warmup_runs, len(results)):]

    s_fps = stats([x["avg_fps"] for x in steady])
    s_frame_ms = stats([x["per_frame_latency_ms"] for x in steady])
    s_decision_ms = stats([x["decision_latency_ms_est"] for x in steady])
    s_active_ratio = stats([x["score"]["active_ratio"] for x in steady])

    s_rss_peak = stats([x["memory_mb"]["rss_peak"] for x in steady])
    s_rss_avg = stats([x["memory_mb"]["rss_avg"] for x in steady])
    s_gpu_peak = stats([x["memory_mb"]["gpu_peak"] for x in steady])
    s_gpu_avg = stats([x["memory_mb"]["gpu_avg"] for x in steady])

    out = {
        "config": {
            "script": args.script,
            "videoFolder": args.videoFolder,
            "videoName": args.videoName,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "sample_ms": args.sample_ms,
            "windowSec": args.windowSec,
            "inferStrideSec": args.inferStrideSec,
            "scoreThres": args.scoreThres,
            "onlyTopSpeaker": args.onlyTopSpeaker,
            "maxSeconds": args.maxSeconds,
        },
        "all_runs": results,
        "steady_stats": {
            "avg_fps": s_fps,
            "per_frame_latency_ms": s_frame_ms,
            "decision_latency_ms_est": s_decision_ms,
            "active_ratio": s_active_ratio,
            "memory_mb": {
                "rss_peak": s_rss_peak,
                "rss_avg": s_rss_avg,
                "gpu_peak": s_gpu_peak,
                "gpu_avg": s_gpu_avg,
            },
        },
    }

    out_path = Path(args.out_json) if args.out_json else Path(args.videoFolder) / args.videoName / "pywork" / "profile_online_demoTalkNet_steady.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n=== Steady-State Latency ===")
    print("avg_fps:", json.dumps(s_fps, ensure_ascii=False))
    print("per_frame_latency_ms:", json.dumps(s_frame_ms, ensure_ascii=False))
    print("decision_latency_ms_est:", json.dumps(s_decision_ms, ensure_ascii=False))

    print("\n=== Steady-State Memory (MB) ===")
    print("rss_peak:", json.dumps(s_rss_peak, ensure_ascii=False))
    print("rss_avg:", json.dumps(s_rss_avg, ensure_ascii=False))
    print("gpu_peak:", json.dumps(s_gpu_peak, ensure_ascii=False))
    print("gpu_avg:", json.dumps(s_gpu_avg, ensure_ascii=False))

    print("\n=== Steady-State Quality Signal ===")
    print("active_ratio:", json.dumps(s_active_ratio, ensure_ascii=False))

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
