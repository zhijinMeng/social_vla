#!/usr/bin/env python3
"""Warm end-to-end profiler for demoTalkNet pipeline.

Goal:
- Load models once (S3FD + TalkNet).
- Process multiple video segments continuously.
- Report per-stage latency + RSS/GPU usage.
- Steady stats use runs from #2 onward by default.

Example:
  source .venv-talknet/bin/activate
  python profile_demoTalkNet_warm_e2e.py \
    --videoFolder demo --videoName 111 \
    --runs 5 --duration-sec 2 --warmup-runs 1
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import shutil
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
import tqdm
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager

from model.faceDetector.s3fd import S3FD
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


def video_duration_sec(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n / fps) if fps > 0 else 0.0


def scene_detect(video_file_path: str, pywork_path: str):
    video_manager = VideoManager([video_file_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)
    if scene_list == []:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
    with open(os.path.join(pywork_path, "scene.pckl"), "wb") as f:
        pickle.dump(scene_list, f)
    return scene_list


def inference_video(detector: S3FD, pyframes_path: str, pywork_path: str, video_file_path: str, facedet_scale: float):
    flist = sorted(glob.glob(os.path.join(pyframes_path, "*.jpg")))
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        image_numpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = detector.detect_faces(image_numpy, conf_th=0.9, scales=[facedet_scale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]})
    with open(os.path.join(pywork_path, "faces.pckl"), "wb") as f:
        pickle.dump(dets, f)
    return dets


def iou(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    areaB = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / float(areaA + areaB - inter + 1e-9)


def track_shot(scene_faces, num_failed_det: int, min_track: int, min_face_size: int):
    tracks = []
    iou_thres = 0.5
    while True:
        track = []
        for frame_faces in scene_faces:
            for face in frame_faces:
                if track == []:
                    track.append(face)
                    frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= num_failed_det:
                    if iou(face["bbox"], track[-1]["bbox"]) > iou_thres:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        if len(track) > min_track:
            frame_num = np.array([x["frame"] for x in track])
            bboxes = np.array([np.array(x["bbox"]) for x in track])
            frame_i = np.arange(frame_num[0], frame_num[-1] + 1)
            bboxes_i = []
            for ij in range(4):
                fn = interp1d(frame_num, bboxes[:, ij])
                bboxes_i.append(fn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)
            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > min_face_size:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})
    return tracks


def crop_video(track, pyframes_path: str, audio_file_path: str, crop_file: str, crop_scale: float, threads: int):
    flist = sorted(glob.glob(os.path.join(pyframes_path, "*.jpg")))
    vout = cv2.VideoWriter(crop_file + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224))
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

    for fidx, frame in enumerate(track["frame"]):
        bs = dets["s"][fidx]
        bsi = int(bs * (1 + 2 * crop_scale))
        image = cv2.imread(flist[frame])
        frame_pad = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my = dets["y"][fidx] + bsi
        mx = dets["x"][fidx] + bsi
        face = frame_pad[int(my - bs): int(my + bs * (1 + 2 * crop_scale)), int(mx - bs * (1 + crop_scale)): int(mx + bs * (1 + crop_scale))]
        vout.write(cv2.resize(face, (224, 224)))
    vout.release()

    audio_tmp = crop_file + ".wav"
    audio_start = (track["frame"][0]) / 25
    audio_end = (track["frame"][-1] + 1) / 25

    cmd = (
        "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 "
        "-threads %d -ss %.3f -to %.3f %s -loglevel panic"
        % (audio_file_path, threads, audio_start, audio_end, audio_tmp)
    )
    subprocess.call(cmd, shell=True, stdout=None)

    cmd = (
        "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic"
        % (crop_file, audio_tmp, threads, crop_file)
    )
    subprocess.call(cmd, shell=True, stdout=None)
    os.remove(crop_file + "t.avi")
    return {"track": track, "proc_track": dets}


def evaluate_network_loaded(model: talkNet, files: list[str], pycrop_path: str):
    all_scores = []
    duration_set = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        _, audio = wavfile.read(os.path.join(pycrop_path, file_name + ".wav"))
        audio_feature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        cap = cv2.VideoCapture(os.path.join(pycrop_path, file_name + ".avi"))
        video_feature = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            face = face[int(112 - 56): int(112 + 56), int(112 - 56): int(112 + 56)]
            video_feature.append(face)
        cap.release()
        video_feature = np.array(video_feature)

        length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0] / 25)
        audio_feature = audio_feature[: int(round(length * 100)), :]
        video_feature = video_feature[: int(round(length * 25)), :, :]

        all_score = []
        with torch.no_grad():
            for duration in duration_set:
                batch_size = int(math.ceil(length / duration))
                scores = []
                for i in range(batch_size):
                    input_a = torch.FloatTensor(audio_feature[i * duration * 100: (i + 1) * duration * 100, :]).unsqueeze(0).cuda()
                    input_v = torch.FloatTensor(video_feature[i * duration * 25: (i + 1) * duration * 25, :, :]).unsqueeze(0).cuda()
                    embed_a = model.model.forward_audio_frontend(input_a)
                    embed_v = model.model.forward_visual_frontend(input_v)
                    embed_a, embed_v = model.model.forward_cross_attention(embed_a, embed_v)
                    out = model.model.forward_audio_visual_backend(embed_a, embed_v)
                    score = model.lossAV.forward(out, labels=None)
                    scores.extend(score)
                all_score.append(scores)
        all_score = np.round(np.mean(np.array(all_score), axis=0), 1).astype(float)
        all_scores.append(all_score)
    return all_scores


def visualization(tracks, scores, pyframes_path: str, pyavi_path: str, threads: int):
    flist = sorted(glob.glob(os.path.join(pyframes_path, "*.jpg")))
    faces = [[] for _ in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]
            s = np.mean(s)
            faces[frame].append(
                {
                    "score": float(s),
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                }
            )

    first = cv2.imread(flist[0])
    fw, fh = first.shape[1], first.shape[0]
    vout = cv2.VideoWriter(os.path.join(pyavi_path, "video_only.avi"), cv2.VideoWriter_fourcc(*"XVID"), 25, (fw, fh))
    color_dict = {0: 0, 1: 255}

    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            clr = color_dict[int((face["score"] >= 0))]
            txt = round(face["score"], 1)
            cv2.rectangle(
                image,
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                (0, clr, 255 - clr),
                6,
            )
            cv2.putText(
                image,
                f"{txt}",
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, clr, 255 - clr),
                3,
            )
        vout.write(image)
    vout.release()

    cmd = (
        "ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic"
        % (
            os.path.join(pyavi_path, "video_only.avi"),
            os.path.join(pyavi_path, "audio.wav"),
            threads,
            os.path.join(pyavi_path, "video_out.avi"),
        )
    )
    subprocess.call(cmd, shell=True, stdout=None)


def phase_mem(samples: list[Sample], t0: float, t1: float) -> dict[str, float]:
    xs = [s for s in samples if t0 <= s.t <= t1]
    if not xs:
        return {"rss_peak_mb": 0.0, "rss_avg_mb": 0.0, "gpu_peak_mb": 0.0, "gpu_avg_mb": 0.0}
    rss = [x.rss_mb for x in xs]
    gpu = [x.gpu_mb for x in xs]
    return {
        "rss_peak_mb": float(max(rss)),
        "rss_avg_mb": float(sum(rss) / len(rss)),
        "gpu_peak_mb": float(max(gpu)),
        "gpu_avg_mb": float(sum(gpu) / len(gpu)),
    }


def print_run_table(run_idx: int, phases: list[dict[str, Any]]) -> None:
    print(f"\n=== Run {run_idx} Stage Latency/Memory ===")
    print("stage                 lat_ms   rss_peak  rss_avg   gpu_peak  gpu_avg")
    for s in phases:
        print(
            f"{s['stage']:<20} {s['elapsed_ms']:>8.1f} "
            f"{s['rss_peak_mb']:>9.1f} {s['rss_avg_mb']:>8.1f} "
            f"{s['gpu_peak_mb']:>9.1f} {s['gpu_avg_mb']:>8.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser("Warm E2E profiler for demoTalkNet pipeline")
    ap.add_argument("--videoFolder", type=str, default="demo")
    ap.add_argument("--videoName", type=str, default="111")
    ap.add_argument("--videoPath", type=str, default="", help="Optional explicit video path")
    ap.add_argument("--pretrainModel", type=str, default="pretrain_TalkSet.model")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--duration-sec", type=float, default=2.0, help="Per-run input clip duration in seconds")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--stride-sec", type=float, default=2.0, help="Start time stride between runs")
    ap.add_argument("--sample-ms", type=int, default=20)
    ap.add_argument("--nDataLoaderThread", type=int, default=10)
    ap.add_argument("--facedetScale", type=float, default=0.25)
    ap.add_argument("--minTrack", type=int, default=10)
    ap.add_argument("--numFailedDet", type=int, default=10)
    ap.add_argument("--minFaceSize", type=int, default=1)
    ap.add_argument("--cropScale", type=float, default=0.40)
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    if args.videoPath:
        src_video = args.videoPath
    else:
        cands = glob.glob(os.path.join(args.videoFolder, args.videoName + ".*"))
        if not cands:
            raise RuntimeError("Input video not found. Please provide --videoPath or put file in videoFolder/videoName.*")
        src_video = cands[0]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by current TalkNet/S3FD path.")

    # One-time load (excluded from steady e2e stats by warmup-runs).
    t_load0 = time.perf_counter()
    detector = S3FD(device="cuda")
    model = talkNet()
    model.loadParameters(args.pretrainModel)
    model.eval()
    torch.cuda.synchronize()
    t_load1 = time.perf_counter()
    model_load_ms = (t_load1 - t_load0) * 1000.0
    print(f"Model+Detector loaded once: {model_load_ms:.1f} ms")

    pid = os.getpid()
    samples: list[Sample] = []
    stop_evt = threading.Event()

    def sampler() -> None:
        while not stop_evt.is_set():
            samples.append(Sample(time.perf_counter(), read_rss_mb(pid), read_gpu_mb(pid)))
            time.sleep(max(args.sample_ms, 5) / 1000.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    base_save = os.path.join(args.videoFolder, args.videoName, "warm_profile_runs")
    os.makedirs(base_save, exist_ok=True)

    vd = video_duration_sec(src_video)
    run_results = []

    try:
        for i in range(args.runs):
            run_idx = i + 1
            run_dir = os.path.join(base_save, f"run_{run_idx:03d}")
            if os.path.isdir(run_dir):
                shutil.rmtree(run_dir)
            os.makedirs(run_dir, exist_ok=True)

            pyavi = os.path.join(run_dir, "pyavi")
            pyframes = os.path.join(run_dir, "pyframes")
            pywork = os.path.join(run_dir, "pywork")
            pycrop = os.path.join(run_dir, "pycrop")
            for p in [pyavi, pyframes, pywork, pycrop]:
                os.makedirs(p, exist_ok=True)

            start = args.start_sec + i * args.stride_sec
            dur = args.duration_sec
            if vd > 0 and dur > 0:
                max_start = max(0.0, vd - dur)
                if start > max_start:
                    start = start % (max_start + 1e-6) if max_start > 0 else 0.0

            print(f"\n[Run {run_idx}/{args.runs}] start={start:.3f}s duration={dur:.3f}s")

            phase_rows = []
            run_t0 = time.perf_counter()

            # 1) extract video/audio/frames
            p0 = time.perf_counter()
            video_file = os.path.join(pyavi, "video.avi")
            cmd = (
                "ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic"
                % (src_video, args.nDataLoaderThread, start, start + dur, video_file)
            )
            subprocess.call(cmd, shell=True, stdout=None)

            audio_file = os.path.join(pyavi, "audio.wav")
            cmd = (
                "ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic"
                % (video_file, args.nDataLoaderThread, audio_file)
            )
            subprocess.call(cmd, shell=True, stdout=None)

            cmd = (
                "ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic"
                % (video_file, args.nDataLoaderThread, os.path.join(pyframes, "%06d.jpg"))
            )
            subprocess.call(cmd, shell=True, stdout=None)
            p1 = time.perf_counter()
            phase_rows.append({"stage": "extract_media", "t0": p0, "t1": p1})

            # 2) scene detect
            p0 = time.perf_counter()
            scenes = scene_detect(video_file, pywork)
            p1 = time.perf_counter()
            phase_rows.append({"stage": "scene_detect", "t0": p0, "t1": p1})

            # 3) face detect + track
            p0 = time.perf_counter()
            faces = inference_video(detector, pyframes, pywork, video_file, args.facedetScale)
            all_tracks = []
            for shot in scenes:
                if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
                    all_tracks.extend(
                        track_shot(
                            faces[shot[0].frame_num: shot[1].frame_num],
                            num_failed_det=args.numFailedDet,
                            min_track=args.minTrack,
                            min_face_size=args.minFaceSize,
                        )
                    )
            p1 = time.perf_counter()
            phase_rows.append({"stage": "face_detect_track", "t0": p0, "t1": p1, "num_tracks": len(all_tracks)})

            # 4) crop tracks
            p0 = time.perf_counter()
            vid_tracks = []
            for ii, track in enumerate(all_tracks):
                vid_tracks.append(
                    crop_video(
                        track,
                        pyframes_path=pyframes,
                        audio_file_path=audio_file,
                        crop_file=os.path.join(pycrop, f"{ii:05d}"),
                        crop_scale=args.cropScale,
                        threads=args.nDataLoaderThread,
                    )
                )
            with open(os.path.join(pywork, "tracks.pckl"), "wb") as f:
                pickle.dump(vid_tracks, f)
            p1 = time.perf_counter()
            phase_rows.append({"stage": "crop_tracks", "t0": p0, "t1": p1})

            # 5) TalkNet score
            p0 = time.perf_counter()
            files = sorted(glob.glob(os.path.join(pycrop, "*.avi")))
            scores = evaluate_network_loaded(model, files, pycrop)
            with open(os.path.join(pywork, "scores.pckl"), "wb") as f:
                pickle.dump(scores, f)
            p1 = time.perf_counter()
            phase_rows.append({"stage": "talknet_score", "t0": p0, "t1": p1, "num_clips": len(files)})

            # 6) visualize + write video
            p0 = time.perf_counter()
            visualization(vid_tracks, scores, pyframes, pyavi, args.nDataLoaderThread)
            p1 = time.perf_counter()
            phase_rows.append({"stage": "visualize_write", "t0": p0, "t1": p1})

            run_t1 = time.perf_counter()

            phases_out = []
            for row in phase_rows:
                mem = phase_mem(samples, row["t0"], row["t1"])
                item = {
                    "stage": row["stage"],
                    "elapsed_ms": (row["t1"] - row["t0"]) * 1000.0,
                    **mem,
                }
                for k in ["num_tracks", "num_clips"]:
                    if k in row:
                        item[k] = row[k]
                phases_out.append(item)

            total_mem = phase_mem(samples, run_t0, run_t1)
            run_item = {
                "run_idx": run_idx,
                "start_sec": start,
                "duration_sec": dur,
                "input_to_output_ms": (run_t1 - run_t0) * 1000.0,
                "phases": phases_out,
                "run_memory": total_mem,
                "output_video": os.path.join(pyavi, "video_out.avi"),
            }
            run_results.append(run_item)

            print(f"[Run {run_idx}] input->output: {run_item['input_to_output_ms']:.1f} ms")
            print_run_table(run_idx, phases_out)
    finally:
        stop_evt.set()
        th.join(timeout=1.0)

    warm = min(max(args.warmup_runs, 0), len(run_results))
    steady = run_results[warm:]

    steady_e2e = stats([r["input_to_output_ms"] for r in steady])

    stage_names = sorted({p["stage"] for r in steady for p in r["phases"]})
    steady_stage_latency = {}
    steady_stage_mem = {}
    for name in stage_names:
        rows = [p for r in steady for p in r["phases"] if p["stage"] == name]
        steady_stage_latency[name] = stats([x["elapsed_ms"] for x in rows])
        steady_stage_mem[name] = {
            "rss_peak_max_mb": float(max(x["rss_peak_mb"] for x in rows)) if rows else 0.0,
            "rss_avg_mean_mb": float(sum(x["rss_avg_mb"] for x in rows) / len(rows)) if rows else 0.0,
            "gpu_peak_max_mb": float(max(x["gpu_peak_mb"] for x in rows)) if rows else 0.0,
            "gpu_avg_mean_mb": float(sum(x["gpu_avg_mb"] for x in rows) / len(rows)) if rows else 0.0,
        }

    out = {
        "video": src_video,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "duration_sec": args.duration_sec,
        "stride_sec": args.stride_sec,
        "model_load_once_ms": model_load_ms,
        "run_results": run_results,
        "steady_input_to_output_ms": steady_e2e,
        "steady_stage_latency_ms": steady_stage_latency,
        "steady_stage_memory_mb": steady_stage_mem,
    }

    out_path = Path(args.out_json) if args.out_json else Path(args.videoFolder) / args.videoName / "pywork" / "profile_demoTalkNet_warm_e2e.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n=== One-time Load (excluded from steady by warmup-runs) ===")
    print(f"model_load_once_ms: {model_load_ms:.1f}")

    print("\n=== Steady input->output latency (from run #2 by default) ===")
    print(json.dumps(steady_e2e, ensure_ascii=False, indent=2))

    print("\n=== Steady stage memory (MB) ===")
    print("stage                 rss_peak_max  rss_avg_mean  gpu_peak_max  gpu_avg_mean")
    for name in sorted(steady_stage_mem):
        m = steady_stage_mem[name]
        print(
            f"{name:<20} {m['rss_peak_max_mb']:>12.1f} {m['rss_avg_mean_mb']:>13.1f} "
            f"{m['gpu_peak_max_mb']:>12.1f} {m['gpu_avg_mean_mb']:>12.1f}"
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
