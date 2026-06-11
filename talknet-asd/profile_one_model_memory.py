#!/usr/bin/env python3
"""单模型显存/内存剖析脚本：定位推理各阶段占用。

cd /home/CNF2026900120/Downloads/AVSBench
python3 profile_one_model_memory.py ~/Downloads/111.qt \
  --model avis_r50 \
  --device cuda \
  --use-trt-avism \
  --trt-engine avis-main/trt_artifacts/r50_test/avism_r50_module_fp16.engine \
  --out-dir memory_profiles_trt

cd /home/CNF2026900120/Downloads/AVSBench
python3 profile_one_model_memory.py ~/Downloads/111.qt \
  --model avis_r50 \
  --device cuda \
  --out-dir memory_profiles_torch


cd /home/CNF2026900120/Downloads/AVSBench
python3 profile_one_model_memory.py ~/Downloads/111.qt \
  --model avis_r50 \
  --device cuda \
  --use-trt-avism \
  --trt-engine avis-main/trt_artifacts/r50_t1_int8/avism_r50_module_int8.engine \
  --avis-chunk-size 1 \
  --out-dir memory_profiles_trt_int8_t1_chunk1

python3 profile_one_model_memory.py ~/Downloads/111.qt \
  --model avis_r50 \
  --device cuda \
  --use-trt-avism \
  --trt-engine avis-main/trt_artifacts/r50_t1_int8/avism_r50_module_int8.engine \
  --avis-chunk-size 1 \
  --avis-min-size-test 160 \
  --out-dir memory_profiles_trt_int8_t1_chunk1

cd /home/CNF2026900120/Downloads/AVSBench
python3 profile_one_model_memory.py ~/Downloads/111.qt \
  --model avis_r50 \
  --device cuda \
  --use-trt-avism \
  --trt-engine avis-main/trt_artifacts/r50_t1_int8/avism_r50_module_int8.engine \
  --avis-max-frames 1 \
  --avis-chunk-size 1 \
  --avis-min-size-test 160 \
  --out-dir memory_profiles_trt_int8_t1_chunk1_f1

cd /home/CNF2026900120/Downloads/AVSBench
python3 profile_one_model_memory.py ~/Downloads/111.qt \
  --model avis_r50 \
  --device cuda \
  --use-trt-avism \
  --trt-engine avis-main/trt_artifacts/r50_t1_int8/avism_r50_module_int8_single_stream.engine \
  --avis-max-frames 1 \
  --avis-chunk-size 1 \
  --avis-min-size-test 160 \
  --out-dir memory_profiles_trt_int8_t1_chunk1_f1_single_stream


cd /home/CNF2026900120/Downloads/AVSBench
python3 profile_one_model_memory.py ~/Downloads/111.qt \
  --model avis_r50 \
  --device cuda \
  --use-trt-avism \
  --trt-engine avis-main/trt_artifacts/r50_t1_int8/avism_r50_module_int8_single_stream.engine \
  --use-trt-audio-vggish \
  --audio-trt-engine avis-main/trt_artifacts/audio_vggish_t1/simple_vggish_int8.engine \
  --avis-max-frames 1 \
  --avis-chunk-size 1 \
  --avis-min-size-test 160 \
  --out-dir memory_profiles_trt_dual_int8_t1_single_stream

cd /home/CNF2026900120/Downloads/AVSBench
python3 profile_one_model_memory.py ~/Downloads/111.qt \
  --model avis_r50 \
  --device cuda \
  --use-trt-avism \
  --trt-engine avis-main/trt_artifacts/r50_t1_int8/avism_r50_module_int8_single_stream.engine \
  --use-trt-audio-vggish \
  --audio-trt-engine avis-main/trt_artifacts/audio_vggish_t1/simple_vggish_int8.engine \
  --use-trt-mask-features \
  --mask-features-trt-engine avis-main/trt_artifacts/mask_features_size160/avism_mask_features_int8.engine \
  --avis-max-frames 1 \
  --avis-chunk-size 1 \
  --avis-min-size-test 160 \
  --out-dir memory_profiles_triple_trt_int8_t1_single_stream





"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def ensure_exists(path: Path, desc: str) -> None:
    """中文说明：统一路径存在性检查。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到{desc}: {path}")


def run_ffmpeg_extract_wav(video_path: Path, wav_path: Path, channels: int = 1) -> None:
    """中文说明：从视频中提取 wav，供后续音频特征使用。"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        str(channels),
        "-ar",
        "16000",
        str(wav_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 提取音频失败: {p.stderr[-800:]}")


def sample_frames_uniform(video_path: Path, num_frames: int) -> list[np.ndarray]:
    """中文说明：均匀采样固定帧数，和基准脚本保持一致。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("无法读取视频帧数")

    idxs = np.linspace(0, max(total - 1, 0), num=num_frames).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if not ok:
            if frames:
                f = frames[-1].copy()
            else:
                cap.release()
                raise RuntimeError(f"读取帧失败: idx={idx}")
        frames.append(f)
    cap.release()
    return frames


def to_exec_path(root: Path, p: str) -> Path:
    """中文说明：保留 venv 路径本身，不 resolve 到系统 python。"""
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    return root / pp


def mem_snapshot_torch(torch_mod: Any) -> dict[str, float]:
    """中文说明：记录当前 CUDA 分配/保留/峰值显存。单位 MB。"""
    if not torch_mod.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
        }
    return {
        "allocated_mb": float(torch_mod.cuda.memory_allocated()) / (1024.0 * 1024.0),
        "reserved_mb": float(torch_mod.cuda.memory_reserved()) / (1024.0 * 1024.0),
        "max_allocated_mb": float(torch_mod.cuda.max_memory_allocated()) / (1024.0 * 1024.0),
    }


def release_cuda_cache_torch(torch_mod: Any) -> None:
    """中文说明：主动触发 GC + CUDA cache 回收，用于观察释放后驻留显存。"""
    if not torch_mod.cuda.is_available():
        return
    import gc

    gc.collect()
    torch_mod.cuda.synchronize()
    torch_mod.cuda.empty_cache()
    try:
        torch_mod.cuda.ipc_collect()
    except Exception:
        pass


def _query_gpu_process_mb(pid: int) -> float:
    """中文说明：查询当前进程在 nvidia-smi 中的显存占用（MiB）。"""
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
            ss = [x.strip() for x in line.split(",")]
            if len(ss) < 2:
                continue
            if int(ss[0]) == int(pid):
                return float(ss[1])
    except Exception:
        pass
    return 0.0


def _run_with_gpu_process_monitor(forward_fn, enable_monitor: bool) -> tuple[float, dict[str, float]]:
    """中文说明：执行前向并采样进程级显存，返回耗时与均值/峰值。"""
    if not enable_monitor:
        t0 = time.perf_counter()
        forward_fn()
        return (time.perf_counter() - t0) * 1000.0, {"avg_mb": 0.0, "peak_mb": 0.0}

    samples: list[float] = []
    stop_evt = threading.Event()
    pid = os.getpid()

    def _worker():
        while not stop_evt.is_set():
            samples.append(_query_gpu_process_mb(pid))
            time.sleep(0.02)

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    t0 = time.perf_counter()
    try:
        forward_fn()
    finally:
        stop_evt.set()
        th.join(timeout=1.0)

    elapsed = (time.perf_counter() - t0) * 1000.0
    if not samples:
        return elapsed, {"avg_mb": 0.0, "peak_mb": 0.0}
    return elapsed, {"avg_mb": float(sum(samples) / len(samples)), "peak_mb": float(max(samples))}


def profile_avseg(
    root: Path,
    model_key: str,
    video: Path,
    device: str,
    out_dir: Path,
    with_op_profiler: bool,
    img_size: int,
) -> dict[str, Any]:
    """中文说明：AVSegFormer 单模型阶段剖析（加载/预处理/前向）。"""
    avseg_repo = root / "AVSegFormer-master"
    sys.path.insert(0, str(avseg_repo))

    import torch
    from mmcv import Config
    from torchvision import transforms

    from model import build_model
    from model.vggish import vggish_input

    def to_tensor(frames_bgr: list[np.ndarray], size: int = 224):
        tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        xs = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            xs.append(tfm(rgb))
        return torch.stack(xs, dim=0)

    def pad_or_trim(audio_lm, T: int):
        n = audio_lm.shape[0]
        if n == T:
            return audio_lm
        if n > T:
            return audio_lm[:T]
        if n == 0:
            return torch.zeros(T, 1, 96, 64)
        pad = audio_lm[-1:].repeat(T - n, 1, 1, 1)
        return torch.cat([audio_lm, pad], dim=0)

    cfg_map = {
        "avseg_s4": ("config/s4/AVSegFormer_pvt2_s4_eval_bs1.py", "work_dir/AVSegFormer_pvt2_s4/S4_best.pth", 5, "s4"),
        "avseg_ms3": ("config/ms3/AVSegFormer_pvt2_ms3_eval_bs1.py", "work_dir/AVSegFormer_pvt2_ms3/MS3_best.pth", 5, "ms3"),
        "avseg_avss": ("config/avss/AVSegFormer_pvt2_avss_eval_bs1.py", "work_dir/AVSegFormer_pvt2_avss/AVSS_best.pth", 10, "avss"),
    }
    cfg_rel, ckpt_rel, T, task = cfg_map[model_key]
    cfg_path = (avseg_repo / cfg_rel).resolve()
    ckpt_path = (avseg_repo / ckpt_rel).resolve()
    ensure_exists(cfg_path, "配置")
    ensure_exists(ckpt_path, "权重")

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    work = out_dir / model_key
    work.mkdir(parents=True, exist_ok=True)
    wav_path = work / "audio.wav"
    run_ffmpeg_extract_wav(video, wav_path, channels=1)

    phase = {}

    # 中文说明：阶段0，空闲基线。
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    phase["baseline"] = mem_snapshot_torch(torch)

    # 中文说明：阶段1，模型加载。
    t0 = time.perf_counter()
    cfg = Config.fromfile(str(cfg_path))

    # 中文说明：AVSeg 配置里常见是相对路径（例如 pretrained/pvt_v2_b5.pth），
    # 这里统一转绝对路径，避免从外层目录启动脚本时找不到文件。
    def _abs_in_repo(path_like: Any) -> Any:
        if not path_like or not isinstance(path_like, str):
            return path_like
        p = Path(path_like)
        if p.is_absolute():
            return path_like
        return str((avseg_repo / p).resolve())

    if "backbone" in cfg.model and "init_weights_path" in cfg.model.backbone:
        cfg.model.backbone.init_weights_path = _abs_in_repo(cfg.model.backbone.init_weights_path)
    if "vggish" in cfg.model and "pretrained_vggish_model_path" in cfg.model.vggish:
        cfg.model.vggish.pretrained_vggish_model_path = _abs_in_repo(cfg.model.vggish.pretrained_vggish_model_path)
    if "vggish" in cfg.model and "pretrained_pca_params_path" in cfg.model.vggish:
        cfg.model.vggish.pretrained_pca_params_path = _abs_in_repo(cfg.model.vggish.pretrained_pca_params_path)

    model = build_model(**cfg.model)
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model = model.to(device)
    model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t0) * 1000.0
    phase["after_model_load"] = mem_snapshot_torch(torch)

    # 中文说明：阶段2，输入预处理。
    t1 = time.perf_counter()
    frames = sample_frames_uniform(video, T)
    # 中文说明：支持外部传入 AVSeg 输入分辨率，便于测试降分辨率对显存/时延影响。
    imgs = to_tensor(frames, img_size).to(device)
    if task == "avss":
        vid_flag = torch.ones(T, device=device)
        audio_input = [str(wav_path)]
        model_inputs = (audio_input, imgs, vid_flag)
    else:
        audio_lm = vggish_input.wavfile_to_examples(str(wav_path))
        audio_lm = pad_or_trim(audio_lm, T=T).to(device)
        model_inputs = (audio_lm, imgs)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    prep_ms = (time.perf_counter() - t1) * 1000.0
    phase["after_preprocess"] = mem_snapshot_torch(torch)

    # 中文说明：阶段3，单次前向（统计前向峰值）。
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    def _forward_once():
        with torch.no_grad():
            _ = model(*model_inputs)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    fwd_ms, gpu_proc = _run_with_gpu_process_monitor(
        _forward_once,
        enable_monitor=device.startswith("cuda"),
    )
    phase["after_forward"] = mem_snapshot_torch(torch)
    # 中文说明：补充释放缓存后的快照，便于判断 reserved 是否可回收。
    if device.startswith("cuda") and torch.cuda.is_available():
        release_cuda_cache_torch(torch)
        phase["after_forward_released"] = mem_snapshot_torch(torch)

    op_profile_txt = ""
    if with_op_profiler:
        # 中文说明：可选算子级显存剖析，帮助定位高占用操作。
        try:
            import torch.profiler as tprof

            with tprof.profile(
                activities=[tprof.ProfilerActivity.CPU] + ([tprof.ProfilerActivity.CUDA] if device.startswith("cuda") and torch.cuda.is_available() else []),
                profile_memory=True,
                record_shapes=True,
            ) as prof:
                with torch.no_grad():
                    _ = model(*model_inputs)
            op_profile_txt = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=25)
        except Exception as e:
            op_profile_txt = f"op profiler failed: {e}"

    result = {
        "model": model_key,
        "device": device,
        "input_settings": {"avseg_img_size": int(img_size)},
        "phase_memory": phase,
        "phase_time_ms": {
            "model_load_ms": load_ms,
            "preprocess_ms": prep_ms,
            "forward_once_ms": fwd_ms,
            "forward_gpu_process_avg_mb": gpu_proc["avg_mb"],
            "forward_gpu_process_peak_mb": gpu_proc["peak_mb"],
        },
    }
    if op_profile_txt:
        result["op_memory_top"] = op_profile_txt
    return result



def _get_trt_expected_t(engine_path: Path) -> int | None:
    """中文说明：读取 TRT engine 的 frame_query 静态 T 维，失败返回 None。"""
    try:
        import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            return None
        shp = tuple(engine.get_tensor_shape("frame_query"))
        if len(shp) >= 2 and int(shp[1]) > 0:
            return int(shp[1])
    except Exception:
        return None
    return None


def _pad_or_trim_frames(frames: list[np.ndarray], target_t: int) -> list[np.ndarray]:
    """中文说明：将视频帧数对齐到 TRT 期望 T（不足补最后一帧，过长截断）。"""
    if target_t <= 0:
        return frames
    if len(frames) == target_t:
        return frames
    if len(frames) > target_t:
        return frames[:target_t]
    if not frames:
        raise RuntimeError("抽帧结果为空，无法对齐到 TRT 期望帧数")
    out = list(frames)
    while len(out) < target_t:
        out.append(out[-1].copy())
    return out


def profile_avis(
    root: Path,
    model_key: str,
    video: Path,
    device: str,
    out_dir: Path,
    with_op_profiler: bool,
    avis_min_size_test: int | None,
    avis_num_queries: int | None,
    avis_max_frames: int,
    avis_chunk_size: int,
    use_trt_backbone: bool,
    backbone_trt_engine: Path | None,
    use_trt_avism: bool,
    trt_engine: Path | None,
    use_trt_audio_vggish: bool,
    audio_trt_engine: Path | None,
    use_trt_mask_features: bool,
    mask_features_trt_engine: Path | None,
    use_trt_mask_pred: bool,
    mask_pred_trt_engine: Path | None,
) -> dict[str, Any]:
    """中文说明：AVIS 单模型阶段剖析（加载/预处理/前向）。"""
    avis_repo = root / "avis-main"
    sys.path.insert(0, str(avis_repo))

    import torch

    # 中文说明：复用现有推理脚本中的函数，减少重复逻辑。
    import scripts.infer_one_mp4_avis as avis_infer

    cfg_map = {
        "avis_r50": ("configs/avism/R50/avism_R50_IN.yaml", "checkpoints/AVISM_R50_IN.pth"),
        "avis_swinl_coco": ("configs/avism/SwinL/avism_SwinL_COCO.yaml", "checkpoints/AVISM_SwinL_COCO.pth"),
    }
    cfg_rel, ckpt_rel = cfg_map[model_key]
    cfg_path = (avis_repo / cfg_rel).resolve()
    ckpt_path = (avis_repo / ckpt_rel).resolve()
    vggish_path = (root / "AVSegFormer-master/pretrained/vggish-10086976.pth").resolve()
    ensure_exists(cfg_path, "配置")
    ensure_exists(ckpt_path, "权重")
    ensure_exists(vggish_path, "VGGish 权重")
    if use_trt_backbone:
        if backbone_trt_engine is None:
            raise ValueError("use_trt_backbone=True 时必须提供 backbone_trt_engine")
        ensure_exists(backbone_trt_engine, "backbone TRT engine")
    if use_trt_avism:
        if trt_engine is None:
            raise ValueError("use_trt_avism=True 时必须提供 trt_engine")
        ensure_exists(trt_engine, "TRT engine")
    if use_trt_audio_vggish:
        if audio_trt_engine is None:
            raise ValueError("use_trt_audio_vggish=True 时必须提供 audio_trt_engine")
        ensure_exists(audio_trt_engine, "音频 TRT engine")
    if use_trt_mask_features:
        if mask_features_trt_engine is None:
            raise ValueError("use_trt_mask_features=True 时必须提供 mask_features_trt_engine")
        ensure_exists(mask_features_trt_engine, "mask_features TRT engine")
    if use_trt_mask_pred:
        if mask_pred_trt_engine is None:
            raise ValueError("use_trt_mask_pred=True 时必须提供 mask_pred_trt_engine")
        ensure_exists(mask_pred_trt_engine, "mask_pred TRT engine")

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    work = out_dir / model_key
    work.mkdir(parents=True, exist_ok=True)
    wav_path = work / "audio.wav"

    phase = {}
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    phase["baseline"] = mem_snapshot_torch(torch)

    # 中文说明：阶段1，提取/预处理输入（AVIS 这里包含音频embedding和抽帧）。
    t0 = time.perf_counter()
    avis_infer.extract_wav(video, wav_path)

    trt_expected_t = None
    if use_trt_avism and trt_engine is not None:
        trt_expected_t = _get_trt_expected_t(trt_engine)

    # 中文说明：可通过 avis_max_frames 控制 AVIS 预处理帧数，便于评估显存与速度折中。
    frames = avis_infer.sample_video_frames(video, sample_fps=1.0, max_frames=int(avis_max_frames))
    if trt_expected_t is not None:
        frames = _pad_or_trim_frames(frames, trt_expected_t)

    audio_feats = avis_infer.build_audio_embeddings(
        repo_root=avis_repo,
        wav_path=wav_path,
        num_steps=len(frames),
        vggish_weight_path=vggish_path,
        device=device,
        use_trt_audio_vggish=bool(use_trt_audio_vggish),
        audio_trt_engine_path=audio_trt_engine,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    prep_ms = (time.perf_counter() - t0) * 1000.0
    phase["after_preprocess"] = mem_snapshot_torch(torch)

    # 中文说明：阶段2，模型加载。
    t1 = time.perf_counter()
    sys.path.insert(0, str(avis_repo / "demo_video"))
    from predictor import VisualizationDemo

    cfg = avis_infer.setup_cfg(config_file=cfg_path, model_weight=ckpt_path, device=device, num_queries=avis_num_queries)
    # 中文说明：可选覆盖 AVIS 测试分辨率（短边），用于降分辨率实验。
    if avis_min_size_test is not None:
        # 中文说明：Detectron2 配置默认是冻结状态，需先解冻再改写。
        cfg.defrost()
        cfg.INPUT.MIN_SIZE_TEST = int(avis_min_size_test)
        cfg.freeze()
    demo = VisualizationDemo(cfg, conf_thres=0.3)
    if use_trt_backbone:
        if not device.startswith("cuda"):
            raise RuntimeError("use_trt_backbone 仅支持 CUDA")
        demo.predictor.model.backbone = avis_infer.TRTBackboneModule(
            backbone_trt_engine,
        ).to(device)
    if use_trt_avism:
        if not device.startswith("cuda"):
            raise RuntimeError("use_trt_avism 仅支持 CUDA")
        # 中文说明：仅替换 avism_module，保持 AVIS 其余流程不变。
        demo.predictor.model.avism_module = avis_infer.TRTAvismModule(
            demo.predictor.model.avism_module,
            trt_engine,
        ).to(device)

    if use_trt_mask_features:
        if not device.startswith("cuda"):
            raise RuntimeError("use_trt_mask_features 仅支持 CUDA")
        demo.predictor.model.avism_module.avism_mask_features = avis_infer.TRTMaskFeaturesModule(
            mask_features_trt_engine,
        ).to(device)

    if use_trt_mask_pred:
        if not device.startswith("cuda"):
            raise RuntimeError("use_trt_mask_pred 仅支持 CUDA")
        demo.predictor.model.trt_mask_pred_module = avis_infer.TRTMaskPredModule(
            mask_pred_trt_engine,
        ).to(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t1) * 1000.0
    phase["after_model_load"] = mem_snapshot_torch(torch)

    # 中文说明：阶段3，单次前向（统计前向峰值）。
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    def _forward_once():
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            _, _ = avis_infer.run_on_video_chunked(demo, frames, audio_feats, avis_chunk_size)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    fwd_ms, gpu_proc = _run_with_gpu_process_monitor(
        _forward_once,
        enable_monitor=device.startswith("cuda"),
    )
    phase["after_forward"] = mem_snapshot_torch(torch)
    # 中文说明：补充释放缓存后的快照，便于判断 reserved 是否可回收。
    if device.startswith("cuda") and torch.cuda.is_available():
        release_cuda_cache_torch(torch)
        phase["after_forward_released"] = mem_snapshot_torch(torch)

    op_profile_txt = ""
    if with_op_profiler:
        try:
            import torch.profiler as tprof

            with tprof.profile(
                activities=[tprof.ProfilerActivity.CPU] + ([tprof.ProfilerActivity.CUDA] if device.startswith("cuda") and torch.cuda.is_available() else []),
                profile_memory=True,
                record_shapes=True,
            ) as prof:
                with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                    _, _ = avis_infer.run_on_video_chunked(demo, frames, audio_feats, avis_chunk_size)
            op_profile_txt = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=25)
        except Exception as e:
            op_profile_txt = f"op profiler failed: {e}"

    result = {
        "model": model_key,
        "device": device,
        "input_settings": {
            "avis_min_size_test": (None if avis_min_size_test is None else int(avis_min_size_test)),
            "avis_num_queries": (None if avis_num_queries is None else int(avis_num_queries)),
            "avis_max_frames": int(avis_max_frames),
            "avis_chunk_size": int(avis_chunk_size),
            "use_trt_backbone": bool(use_trt_backbone),
            "backbone_trt_engine": (None if backbone_trt_engine is None else str(backbone_trt_engine)),
            "use_trt_avism": bool(use_trt_avism),
            "trt_engine": (None if trt_engine is None else str(trt_engine)),
            "trt_expected_t": (None if (not use_trt_avism or trt_engine is None) else _get_trt_expected_t(trt_engine)),
            "use_trt_audio_vggish": bool(use_trt_audio_vggish),
            "audio_trt_engine": (None if audio_trt_engine is None else str(audio_trt_engine)),
            "use_trt_mask_features": bool(use_trt_mask_features),
            "mask_features_trt_engine": (None if mask_features_trt_engine is None else str(mask_features_trt_engine)),
            "use_trt_mask_pred": bool(use_trt_mask_pred),
            "mask_pred_trt_engine": (None if mask_pred_trt_engine is None else str(mask_pred_trt_engine)),
        },
        "phase_memory": phase,
        "phase_time_ms": {
            "preprocess_ms": prep_ms,
            "model_load_ms": load_ms,
            "forward_once_ms": fwd_ms,
            "forward_gpu_process_avg_mb": gpu_proc["avg_mb"],
            "forward_gpu_process_peak_mb": gpu_proc["peak_mb"],
        },
    }
    if op_profile_txt:
        result["op_memory_top"] = op_profile_txt
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="单模型显存剖析：定位推理各阶段占用")
    parser.add_argument("video", type=str, help="输入视频路径")
    parser.add_argument("--model", type=str, required=True, choices=[
        "avseg_s4", "avseg_ms3", "avseg_avss", "avis_r50", "avis_swinl_coco"
    ], help="待剖析模型")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("--out-dir", type=str, default="memory_profiles", help="输出目录")
    parser.add_argument("--with-op-profiler", action="store_true", help="是否输出算子级显存 Top")
    parser.add_argument("--avseg-img-size", type=int, default=224, help="AVSeg 输入分辨率（正方形边长）")
    parser.add_argument("--avis-min-size-test", type=int, default=None, help="AVIS 测试短边分辨率，默认用原配置")
    parser.add_argument("--avis-num-queries", type=int, default=None, help="AVIS NUM_OBJECT_QUERIES 覆盖值（如 50/75）")
    parser.add_argument("--avis-max-frames", type=int, default=10, help="AVIS 预处理抽帧上限，用于控制推理输入时长")
    parser.add_argument("--avis-chunk-size", type=int, default=0, help="AVIS 按时间维分块推理；0 表示不分块")
    parser.add_argument("--use-trt-backbone", action="store_true", help="AVIS 侧使用 TRT engine 替代 backbone 前向")
    parser.add_argument("--backbone-trt-engine", type=str, default="avis-main/trt_artifacts/backbone_r50/avism_r50_backbone_int8.engine", help="backbone TRT engine 路径（相对 AVSBench 根目录）")
    parser.add_argument("--use-trt-avism", action="store_true", help="AVIS 侧使用 TRT engine 替代 avism_module 前向")
    parser.add_argument("--trt-engine", type=str, default="avis-main/trt_artifacts/r50_test/avism_r50_module_fp16.engine", help="AVIS TRT engine 路径（相对 AVSBench 根目录）")
    parser.add_argument("--use-trt-audio-vggish", action="store_true", help="AVIS 侧使用 TRT engine 替代音频 SimpleVGGish 前向")
    parser.add_argument("--audio-trt-engine", type=str, default="avis-main/trt_artifacts/audio_vggish/simple_vggish_int8.engine", help="音频 TRT engine 路径（相对 AVSBench 根目录）")
    parser.add_argument("--use-trt-mask-features", action="store_true", help="AVIS 侧使用 TRT engine 替代 avism_mask_features 前向")
    parser.add_argument("--mask-features-trt-engine", type=str, default="avis-main/trt_artifacts/mask_features/avism_mask_features_int8.engine", help="mask_features TRT engine 路径（相对 AVSBench 根目录）")
    parser.add_argument("--use-trt-mask-pred", action="store_true", help="AVIS 侧使用 TRT engine 替代 mask_pred(einsum) 前向")
    parser.add_argument("--mask-pred-trt-engine", type=str, default="avis-main/trt_artifacts/mask_pred/avism_mask_pred_int8.engine", help="mask_pred TRT engine 路径（相对 AVSBench 根目录）")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    video = Path(args.video).expanduser().resolve()
    ensure_exists(video, "输入视频")

    # 中文说明：按模型自动切换到对应虚拟环境解释器，避免缺少依赖（如 torch/mmcv/detectron2）。
    if args.model.startswith("avseg_"):
        target_py = root / "AVSegFormer-master/.venv-avseg/bin/python"
    else:
        target_py = root / "avis-main/.venv-avis/bin/python"

    if os.environ.get("PROFILE_MEMORY_REEXEC") != "1":
        cur = Path(sys.executable)
        tgt = target_py
        # 中文说明：这里不做 resolve，避免软链接都解到 /usr/bin/python3 导致误判。
        if target_py.exists() and cur != tgt:
            env = os.environ.copy()
            env["PROFILE_MEMORY_REEXEC"] = "1"
            cmd = [str(target_py), str(Path(__file__).resolve())] + sys.argv[1:]
            p_run = subprocess.run(cmd, env=env)
            sys.exit(p_run.returncode)

    out_dir = (root / args.out_dir / video.stem).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model.startswith("avseg_"):
        result = profile_avseg(root, args.model, video, args.device, out_dir, args.with_op_profiler, args.avseg_img_size)
    else:
        backbone_trt_engine = None
        if args.use_trt_backbone:
            backbone_trt_engine = Path(args.backbone_trt_engine).expanduser()
            if not backbone_trt_engine.is_absolute():
                backbone_trt_engine = (root / backbone_trt_engine).resolve()

        trt_engine = None
        if args.use_trt_avism:
            trt_engine = Path(args.trt_engine).expanduser()
            if not trt_engine.is_absolute():
                trt_engine = (root / trt_engine).resolve()
        audio_trt_engine = None
        if args.use_trt_audio_vggish:
            audio_trt_engine = Path(args.audio_trt_engine).expanduser()
            if not audio_trt_engine.is_absolute():
                audio_trt_engine = (root / audio_trt_engine).resolve()

        mask_features_trt_engine = None
        if args.use_trt_mask_features:
            mask_features_trt_engine = Path(args.mask_features_trt_engine).expanduser()
            if not mask_features_trt_engine.is_absolute():
                mask_features_trt_engine = (root / mask_features_trt_engine).resolve()

        mask_pred_trt_engine = None
        if args.use_trt_mask_pred:
            mask_pred_trt_engine = Path(args.mask_pred_trt_engine).expanduser()
            if not mask_pred_trt_engine.is_absolute():
                mask_pred_trt_engine = (root / mask_pred_trt_engine).resolve()

        result = profile_avis(
            root,
            args.model,
            video,
            args.device,
            out_dir,
            args.with_op_profiler,
            args.avis_min_size_test,
            args.avis_num_queries,
            args.avis_max_frames,
            args.avis_chunk_size,
            args.use_trt_backbone,
            backbone_trt_engine,
            args.use_trt_avism,
            trt_engine,
            args.use_trt_audio_vggish,
            audio_trt_engine,
            args.use_trt_mask_features,
            mask_features_trt_engine,
            args.use_trt_mask_pred,
            mask_pred_trt_engine,
        )

    out_json = out_dir / f"{args.model}_memory_profile.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {out_json}")


if __name__ == "__main__":
    main()
