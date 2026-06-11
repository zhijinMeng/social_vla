#!/usr/bin/env python3
"""Record 4-mic audio and compare raw mono vs RNNoise output."""

import argparse
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np


def record_raw_alsa(card_name, sr, channels, duration_sec):
    duration_sec = float(duration_sec)
    frame_count = int(round(duration_sec * int(sr)))
    if frame_count <= 0:
        raise RuntimeError(f"duration_sec too small: {duration_sec}")
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
        raise RuntimeError(f"arecord failed ({proc.returncode}): {stderr or 'unknown error'}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    frames = pcm.size // int(channels)
    if frames <= 0:
        raise RuntimeError("arecord returned no audio")
    return pcm[: frames * int(channels)].reshape(frames, int(channels)).astype(np.float32) / 32768.0


def save_mono_wav(path, sr, audio):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio16 = np.clip(np.asarray(audio).reshape(-1), -1.0, 1.0)
    audio16 = (audio16 * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(audio16.tobytes())


def save_split_wavs(out_dir, prefix, sr, audio):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ch in range(audio.shape[1]):
        save_mono_wav(out_dir / f"{prefix}_ch{ch}.wav", sr, audio[:, ch])


def collapse_to_mono(audio, mode):
    audio = np.asarray(audio, dtype=np.float32)
    mode = str(mode).strip().lower()
    if mode == "first":
        return audio[:, 0]
    if mode == "mean":
        return np.mean(audio, axis=1)
    energies = np.mean(audio * audio, axis=0)
    return audio[:, int(np.argmax(energies))]


def resample_linear(x, src_sr, dst_sr):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if int(src_sr) == int(dst_sr):
        return x.copy()
    if x.size <= 1:
        return x.copy()
    src_n = x.size
    dst_n = int(round(src_n * float(dst_sr) / float(src_sr)))
    dst_n = max(dst_n, 1)
    src_t = np.linspace(0.0, 1.0, src_n, endpoint=False)
    dst_t = np.linspace(0.0, 1.0, dst_n, endpoint=False)
    return np.interp(dst_t, src_t, x).astype(np.float32)


def run_rnnoise_demo(rnnoise_bin, mono_48k):
    mono_48k = np.asarray(mono_48k, dtype=np.float32).reshape(-1)
    raw_in = tempfile.NamedTemporaryFile(prefix="rnnoise_in_", suffix=".pcm", delete=False)
    raw_out = tempfile.NamedTemporaryFile(prefix="rnnoise_out_", suffix=".pcm", delete=False)
    try:
        raw_in.write((np.clip(mono_48k, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        raw_in.flush()
        raw_in.close()
        raw_out.close()
        cmd = [rnnoise_bin, raw_in.name, raw_out.name]
        env = dict(**__import__("os").environ)
        rnnoise_root = Path(rnnoise_bin).resolve().parents[2]
        lib_dir = str(rnnoise_root / ".libs")
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib_dir if not old_ld else f"{lib_dir}:{old_ld}"
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, env=env)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"rnnoise_demo failed ({proc.returncode}): {stderr or 'unknown error'}")
        out = np.fromfile(raw_out.name, dtype=np.int16).astype(np.float32) / 32768.0
        return out
    finally:
        Path(raw_in.name).unlink(missing_ok=True)
        Path(raw_out.name).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Record 4-mic audio and compare raw mono vs RNNoise output")
    ap.add_argument("--alsaCard", type=str, default="AIUIUSBMC")
    ap.add_argument("--sampleRate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=4)
    ap.add_argument("--recordSec", type=float, default=5.0)
    ap.add_argument("--outDir", type=str, default="/tmp/rnnoise_compare_4mic")
    ap.add_argument("--prefix", type=str, default="take")
    ap.add_argument("--monoMode", type=str, default="max_energy", choices=["max_energy", "first", "mean"])
    ap.add_argument("--rnnoiseBin", type=str, default="", help="Optional rnnoise_demo path")
    args = ap.parse_args()

    rnnoise_bin = args.rnnoiseBin.strip() or shutil.which("rnnoise_demo")
    if not rnnoise_bin:
        raise RuntimeError(
            "rnnoise_demo not found. Install/build RNNoise first, or pass --rnnoiseBin /path/to/rnnoise_demo"
        )

    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"out_dir={out_dir}")
    print(f"alsa_card={args.alsaCard} sample_rate={args.sampleRate} channels={args.channels}")
    print(f"record_sec={args.recordSec} mono_mode={args.monoMode}")
    print(f"rnnoise_bin={rnnoise_bin}")
    print("")
    print(f"Recording starts in 2s, duration={args.recordSec:.1f}s ...")
    time.sleep(2.0)
    audio4 = record_raw_alsa(args.alsaCard, args.sampleRate, args.channels, args.recordSec)
    print("capture done.")

    save_split_wavs(out_dir, f"{args.prefix}_raw", args.sampleRate, audio4)

    raw_mono = collapse_to_mono(audio4, args.monoMode)
    save_mono_wav(out_dir / f"{args.prefix}_raw_mono_{args.monoMode}.wav", args.sampleRate, raw_mono)

    raw_48k = resample_linear(raw_mono, args.sampleRate, 48000)
    rn_48k = run_rnnoise_demo(rnnoise_bin, raw_48k)
    rn_out = resample_linear(rn_48k, 48000, args.sampleRate)
    save_mono_wav(out_dir / f"{args.prefix}_rnnoise_mono_{args.monoMode}.wav", args.sampleRate, rn_out)

    print("")
    print("saved_files:")
    for path in sorted(out_dir.glob(f"{args.prefix}_*.wav")):
        print(path)


if __name__ == "__main__":
    main()
