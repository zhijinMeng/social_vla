#!/usr/bin/env python3
"""Profile demoTalkNet.py stage latency + RAM/VRAM usage.

Example:
  source .venv-talknet/bin/activate
  python profile_demoTalkNet_memory.py --videoFolder demo --videoName 111 --runs 5 --warmup-runs 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MARKERS = [
    ("extract_video", "Extract the video and save in"),
    ("extract_audio", "Extract the audio and save in"),
    ("extract_frames", "Extract the frames and save in"),
    ("scene_detect", "Scene detection and save in"),
    ("face_detection", "Face detection and save in"),
    ("face_tracking", "Face track and detected"),
    ("face_crop", "Face Crop and saved in"),
    ("asd_inference", "Scores extracted and saved in"),
]


@dataclass
class Sample:
    t: float
    rss_mb: float
    gpu_mb: float


def _read_rss_mb(pid: int) -> float:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return 0.0
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                kb = float(line.split()[1])
                return kb / 1024.0
    except Exception:
        return 0.0
    return 0.0


def _read_gpu_mb(pid: int) -> float:
    try:
        p = subprocess.run(
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
        if p.returncode != 0:
            return 0.0
        for line in p.stdout.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 2:
                continue
            if int(cols[0]) == int(pid):
                return float(cols[1])
    except Exception:
        return 0.0
    return 0.0


def _profile_one_run(cmd: list[str], sample_ms: int) -> dict[str, Any]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    samples: list[Sample] = []
    lines: list[str] = []
    marker_times: dict[str, float] = {}
    t0 = time.perf_counter()
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            now = time.perf_counter()
            rss_mb = _read_rss_mb(proc.pid)
            gpu_mb = _read_gpu_mb(proc.pid)
            samples.append(Sample(t=now, rss_mb=rss_mb, gpu_mb=gpu_mb))
            time.sleep(max(sample_ms, 5) / 1000.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    marker_idx = 0
    assert proc.stdout is not None
    for raw in proc.stdout:
        for chunk in raw.replace("\r", "\n").split("\n"):
            if not chunk:
                continue
            lines.append(chunk)
            if marker_idx < len(MARKERS):
                stage_name, token = MARKERS[marker_idx]
                if token in chunk:
                    marker_times[stage_name] = time.perf_counter()
                    marker_idx += 1

    rc = proc.wait()
    t1 = time.perf_counter()
    stop.set()
    th.join(timeout=1.0)

    def stats_between(t_start: float, t_end: float) -> dict[str, float]:
        xs = [s for s in samples if t_start <= s.t <= t_end]
        if not xs:
            return {"rss_peak_mb": 0.0, "rss_avg_mb": 0.0, "gpu_peak_mb": 0.0, "gpu_avg_mb": 0.0}
        rss = [s.rss_mb for s in xs]
        gpu = [s.gpu_mb for s in xs]
        return {
            "rss_peak_mb": float(max(rss)),
            "rss_avg_mb": float(sum(rss) / len(rss)),
            "gpu_peak_mb": float(max(gpu)),
            "gpu_avg_mb": float(sum(gpu) / len(gpu)),
        }

    stages = []
    prev_t = t0
    for stage_name, _ in MARKERS:
        if stage_name in marker_times:
            end_t = marker_times[stage_name]
            m = stats_between(prev_t, end_t)
            stages.append({"stage": stage_name, "elapsed_ms": (end_t - prev_t) * 1000.0, **m})
            prev_t = end_t
        else:
            break

    tail_m = stats_between(prev_t, t1)
    stages.append(
        {
            "stage": "post_asd_to_exit",
            "elapsed_ms": (t1 - prev_t) * 1000.0,
            **tail_m,
        }
    )

    return {
        "return_code": rc,
        "total_elapsed_ms": (t1 - t0) * 1000.0,
        "stages": stages,
        "stdout_tail": lines[-80:],
    }


def _steady_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    vs = sorted(values)
    return {
        "mean_ms": float(sum(vs) / len(vs)),
        "p50_ms": float(statistics.median(vs)),
        "p95_ms": float(vs[int(0.95 * (len(vs) - 1))]),
        "min_ms": float(vs[0]),
        "max_ms": float(vs[-1]),
    }


def _print_stage_table(stages: list[dict[str, Any]], title: str) -> None:
    print(f"\n=== {title} ===")
    print("stage                 lat_ms   rss_peak  rss_avg   gpu_peak  gpu_avg")
    for s in stages:
        print(
            f"{s['stage']:<20} {s['elapsed_ms']:>8.1f} "
            f"{s.get('rss_peak_mb', 0.0):>9.1f} {s.get('rss_avg_mb', 0.0):>8.1f} "
            f"{s.get('gpu_peak_mb', 0.0):>9.1f} {s.get('gpu_avg_mb', 0.0):>8.1f}"
        )


def _steady_mem_by_stage(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stage_names = sorted({s["stage"] for r in runs for s in r["stages"]})
    out: dict[str, dict[str, float]] = {}
    for name in stage_names:
        sel = [s for r in runs for s in r["stages"] if s["stage"] == name]
        if not sel:
            out[name] = {"rss_peak_max_mb": 0.0, "rss_avg_mean_mb": 0.0, "gpu_peak_max_mb": 0.0, "gpu_avg_mean_mb": 0.0}
            continue
        out[name] = {
            "rss_peak_max_mb": float(max(x.get("rss_peak_mb", 0.0) for x in sel)),
            "rss_avg_mean_mb": float(sum(x.get("rss_avg_mb", 0.0) for x in sel) / len(sel)),
            "gpu_peak_max_mb": float(max(x.get("gpu_peak_mb", 0.0) for x in sel)),
            "gpu_avg_mean_mb": float(sum(x.get("gpu_avg_mb", 0.0) for x in sel) / len(sel)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser("Profile demoTalkNet stage memory/latency")
    ap.add_argument("--videoFolder", type=str, default="demo")
    ap.add_argument("--videoName", type=str, default="001")
    ap.add_argument("--runs", type=int, default=3, help="Total repeated runs")
    ap.add_argument("--warmup-runs", type=int, default=1, help="Exclude first N runs for steady-state")
    ap.add_argument("--sample-ms", type=int, default=20, help="Sampling interval in ms")
    ap.add_argument("--out-json", type=str, default="", help="Output JSON path")
    args = ap.parse_args()

    cmd = [
        "python",
        "demoTalkNet.py",
        "--videoFolder",
        args.videoFolder,
        "--videoName",
        args.videoName,
    ]

    all_runs = []
    for i in range(args.runs):
        print(f"[Run {i+1}/{args.runs}] start")
        res = _profile_one_run(cmd, sample_ms=args.sample_ms)
        all_runs.append(res)
        print(f"[Run {i+1}/{args.runs}] rc={res['return_code']} total={res['total_elapsed_ms']:.1f} ms")
        _print_stage_table(res["stages"], f"Run {i+1} Stage Memory/Latency")
        if res["return_code"] != 0:
            break

    warm = max(0, min(args.warmup_runs, len(all_runs)))
    steady_runs = all_runs[warm:]

    steady_total = _steady_stats([r["total_elapsed_ms"] for r in steady_runs])

    stage_names = sorted({s["stage"] for r in steady_runs for s in r["stages"]})
    steady_stage: dict[str, dict[str, float]] = {}
    for name in stage_names:
        vals = [s["elapsed_ms"] for r in steady_runs for s in r["stages"] if s["stage"] == name]
        steady_stage[name] = _steady_stats(vals)

    steady_mem = _steady_mem_by_stage(steady_runs)

    out = {
        "cmd": cmd,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "sample_ms": args.sample_ms,
        "results": all_runs,
        "steady_total_latency_ms": steady_total,
        "steady_stage_latency_ms": steady_stage,
        "steady_stage_memory_mb": steady_mem,
    }

    out_path = Path(args.out_json) if args.out_json else Path(args.videoFolder) / args.videoName / "pywork" / "profile_demoTalkNet_memory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n=== Steady-state total latency (ms) ===")
    print(json.dumps(steady_total, ensure_ascii=False, indent=2))
    print("\n=== Steady-state stage latency (ms) ===")
    print(json.dumps(steady_stage, ensure_ascii=False, indent=2))

    print("\n=== Steady-state stage memory (MB) ===")
    print("stage                 rss_peak_max  rss_avg_mean  gpu_peak_max  gpu_avg_mean")
    for name in sorted(steady_mem):
        m = steady_mem[name]
        print(
            f"{name:<20} {m['rss_peak_max_mb']:>12.1f} {m['rss_avg_mean_mb']:>13.1f} "
            f"{m['gpu_peak_max_mb']:>12.1f} {m['gpu_avg_mean_mb']:>12.1f}"
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
