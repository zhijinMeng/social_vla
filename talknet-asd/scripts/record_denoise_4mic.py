#!/usr/bin/env python3
"""Record 4-mic audio, learn fan noise, and export raw/denoised wavs."""

import argparse
import subprocess
import time
import wave
from pathlib import Path

import numpy as np


def record_raw_alsa(card_name, sr, channels, duration_sec):
    duration_sec = float(duration_sec)
    if duration_sec <= 0:
        raise RuntimeError(f"duration_sec must be > 0, got {duration_sec}")
    frame_count = int(round(duration_sec * int(sr)))
    if frame_count <= 0:
        raise RuntimeError(f"duration_sec too small, got {duration_sec}")
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
        raise RuntimeError("arecord returned no audio frames")
    pcm = pcm[: frames * int(channels)].reshape(frames, int(channels)).astype(np.float32) / 32768.0
    return pcm


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


def stft_denoise_channel(
    noisy,
    noise_ref,
    sr,
    low_hz=120.0,
    high_hz=4200.0,
    strength=1.8,
    floor_ratio=0.08,
    win_sec=0.032,
    hop_sec=0.016,
):
    noisy = np.asarray(noisy, dtype=np.float32).reshape(-1)
    noise_ref = np.asarray(noise_ref, dtype=np.float32).reshape(-1)
    n_fft = max(256, int(round(win_sec * sr)))
    hop = max(64, int(round(hop_sec * sr)))
    if n_fft % 2 == 1:
        n_fft += 1
    window = np.hanning(n_fft).astype(np.float32)
    eps = 1e-8

    def frame_signal(x):
        if x.size < n_fft:
            x = np.pad(x, (0, n_fft - x.size))
        extra = (hop - ((x.size - n_fft) % hop)) % hop
        x = np.pad(x, (0, extra + n_fft))
        frames = []
        for start in range(0, x.size - n_fft + 1, hop):
            frames.append(x[start : start + n_fft])
        return np.stack(frames, axis=0), x.size

    noise_frames, _ = frame_signal(noise_ref)
    noise_spec = np.fft.rfft(noise_frames * window[None, :], axis=1)
    noise_psd = np.mean(np.abs(noise_spec) ** 2, axis=0)

    noisy_frames, padded_len = frame_signal(noisy)
    noisy_spec = np.fft.rfft(noisy_frames * window[None, :], axis=1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    band_mask = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    mag = np.abs(noisy_spec)
    phase = np.angle(noisy_spec)
    noise_mag = np.sqrt(np.maximum(noise_psd, eps))[None, :]
    clean_mag = np.maximum(mag - float(strength) * noise_mag, float(floor_ratio) * mag)
    clean_mag[:, ~band_mask] *= 0.05
    clean_spec = clean_mag * np.exp(1j * phase)
    clean_frames = np.fft.irfft(clean_spec, n=n_fft, axis=1).astype(np.float32)

    out = np.zeros((padded_len,), dtype=np.float32)
    norm = np.zeros((padded_len,), dtype=np.float32)
    for i, frame in enumerate(clean_frames):
        start = i * hop
        out[start : start + n_fft] += frame * window
        norm[start : start + n_fft] += window ** 2
    norm[norm < eps] = 1.0
    out = out / norm
    return np.clip(out[: noisy.size], -1.0, 1.0)


def collapse_to_mono(audio, mode):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] == 1:
        return audio.reshape(-1)
    mode = str(mode).strip().lower()
    if mode == "first":
        return audio[:, 0]
    if mode == "mean":
        return np.mean(audio, axis=1)
    energies = np.mean(audio * audio, axis=0)
    return audio[:, int(np.argmax(energies))]


def main():
    ap = argparse.ArgumentParser(description="Record 4-mic audio, learn fan noise, and export denoised comparisons")
    ap.add_argument("--alsaCard", type=str, default="AIUIUSBMC")
    ap.add_argument("--sampleRate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=4)
    ap.add_argument("--noiseSec", type=float, default=2.0, help="Record this much fan-only noise first")
    ap.add_argument("--recordSec", type=float, default=5.0, help="Then record this much target audio")
    ap.add_argument("--outDir", type=str, default="/tmp/denoise_4mic")
    ap.add_argument("--prefix", type=str, default="take")
    ap.add_argument("--lowHz", type=float, default=120.0)
    ap.add_argument("--highHz", type=float, default=4200.0)
    ap.add_argument("--strength", type=float, default=1.8)
    ap.add_argument("--floorRatio", type=float, default=0.08)
    ap.add_argument("--monoMode", type=str, default="max_energy", choices=["max_energy", "first", "mean"])
    args = ap.parse_args()

    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"out_dir={out_dir}")
    print(f"alsa_card={args.alsaCard} sample_rate={args.sampleRate} channels={args.channels}")
    print(f"noise_sec={args.noiseSec} record_sec={args.recordSec}")
    print(f"denoise_band_hz=({args.lowHz}, {args.highHz}) strength={args.strength} floor_ratio={args.floorRatio}")
    print("")
    print(f"1/2 Keep the robot running, but DO NOT speak for {args.noiseSec:.1f}s. Starting in 2s...")
    time.sleep(2.0)
    noise_audio = record_raw_alsa(args.alsaCard, args.sampleRate, args.channels, args.noiseSec)
    print("noise profile recorded.")
    print("")
    print(f"2/2 Now speak for {args.recordSec:.1f}s. Starting in 1s...")
    time.sleep(1.0)
    noisy_audio = record_raw_alsa(args.alsaCard, args.sampleRate, args.channels, args.recordSec)
    print("target recording captured.")

    save_split_wavs(out_dir, f"{args.prefix}_raw", args.sampleRate, noisy_audio)
    save_split_wavs(out_dir, f"{args.prefix}_noise", args.sampleRate, noise_audio)

    denoised = np.zeros_like(noisy_audio, dtype=np.float32)
    for ch in range(noisy_audio.shape[1]):
        denoised[:, ch] = stft_denoise_channel(
            noisy_audio[:, ch],
            noise_audio[:, min(ch, noise_audio.shape[1] - 1)],
            sr=args.sampleRate,
            low_hz=args.lowHz,
            high_hz=args.highHz,
            strength=args.strength,
            floor_ratio=args.floorRatio,
        )
    save_split_wavs(out_dir, f"{args.prefix}_denoised", args.sampleRate, denoised)

    raw_mono = collapse_to_mono(noisy_audio, args.monoMode)
    denoised_mono = collapse_to_mono(denoised, args.monoMode)
    save_mono_wav(out_dir / f"{args.prefix}_raw_mono_{args.monoMode}.wav", args.sampleRate, raw_mono)
    save_mono_wav(out_dir / f"{args.prefix}_denoised_mono_{args.monoMode}.wav", args.sampleRate, denoised_mono)

    print("")
    print("saved_files:")
    for path in sorted(out_dir.glob(f"{args.prefix}_*.wav")):
        print(path)


if __name__ == "__main__":
    main()
