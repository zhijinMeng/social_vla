#!/usr/bin/env python3
"""Minimal 4-mic rough DOA estimator using GCC-PHAT.

Supports:
- reading a multichannel wav file
- recording a short chunk from a sounddevice input device

Assumptions:
- 4 effective microphones arranged approximately in a horizontal line
- channels are ordered left->right (customizable with --channels)
- output is a rough horizontal angle in degrees
"""

import argparse
import math
import time
import tempfile
import wave

import numpy as np


def gcc_phat(sig, refsig, fs, max_tau=None, interp=8):
    sig = np.asarray(sig, dtype=np.float64)
    refsig = np.asarray(refsig, dtype=np.float64)
    n = sig.shape[0] + refsig.shape[0]

    sig -= np.mean(sig)
    refsig -= np.mean(refsig)

    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REF)
    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    cc = np.fft.irfft(R / denom, n=interp * n)

    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc)) - max_shift)
    tau = shift / float(interp * fs)
    peak = float(np.max(np.abs(cc)))
    return tau, peak


def read_wav_channels(path: str):
    path = str(path)
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width != 2:
        raise RuntimeError(f"Only 16-bit PCM wav is supported, got sample_width={sample_width}")
    audio = np.frombuffer(raw, dtype=np.int16)
    audio = audio.reshape(-1, channels).astype(np.float32) / 32768.0
    return sr, audio


def record_from_device(device, sr, channels, duration):
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError("sounddevice is required for live capture: pip install sounddevice") from exc

    audio = sd.rec(
        int(round(duration * sr)),
        samplerate=sr,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return sr, np.asarray(audio, dtype=np.float32)


def parse_channels(raw: str):
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) != 4:
        raise RuntimeError(f"--channels must contain exactly 4 indices, got: {raw!r}")
    return vals


def _fft_bandpass(sig, sr, low_hz, high_hz):
    sig = np.asarray(sig, dtype=np.float64)
    n = sig.shape[0]
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    spec[~mask] = 0
    out = np.fft.irfft(spec, n=n)
    return out.astype(np.float32)


def bandpass_multich(audio, sr, low_hz, high_hz):
    out = np.zeros_like(audio, dtype=np.float32)
    for ch in range(audio.shape[1]):
        out[:, ch] = _fft_bandpass(audio[:, ch], sr=sr, low_hz=low_hz, high_hz=high_hz)
    return out


def rms_level(audio):
    audio = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(audio ** 2) + 1e-12))


def prepare_temp_alsa_default(card_name: str, channels: int, rate: int):
    import os

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
    tmp = tempfile.NamedTemporaryFile(prefix="rough_doa_asound_", suffix=".conf", delete=False)
    tmp.write(cfg.encode("utf-8"))
    tmp.flush()
    tmp.close()
    os.environ["ALSA_CONFIG_PATH"] = tmp.name
    return tmp.name


def select_active_window(audio, sr, window_sec, speech_low_hz, speech_high_hz):
    win = max(1, int(round(window_sec * sr)))
    if audio.shape[0] <= win:
        return audio

    speech_audio = bandpass_multich(audio[:, :4], sr=sr, low_hz=speech_low_hz, high_hz=speech_high_hz)
    speech_energy = np.mean(speech_audio ** 2, axis=1)
    full_energy = np.mean(audio[:, :4] ** 2, axis=1) + 1e-9
    score = speech_energy / full_energy

    kernel = np.ones(win, dtype=np.float64)
    smooth = np.convolve(score, kernel, mode="valid")
    start = int(np.argmax(smooth))
    end = start + win
    return audio[start:end]


def estimate_angle_deg(audio4, sr, spacing_m, sound_speed):
    # Use outer pair for primary estimate, adjacent pairs for sanity/weighting.
    max_tau_outer = (3.0 * spacing_m) / sound_speed
    max_tau_adj = spacing_m / sound_speed

    pair_specs = [
        ((0, 3), 3.0 * spacing_m, max_tau_outer, 1.0),
        ((0, 1), 1.0 * spacing_m, max_tau_adj, 0.35),
        ((1, 2), 1.0 * spacing_m, max_tau_adj, 0.35),
        ((2, 3), 1.0 * spacing_m, max_tau_adj, 0.35),
    ]

    angle_votes = []
    debug_rows = []
    for (i, j), baseline, max_tau, weight in pair_specs:
        tau, peak = gcc_phat(audio4[:, j], audio4[:, i], fs=sr, max_tau=max_tau, interp=8)
        x = np.clip((sound_speed * tau) / baseline, -1.0, 1.0)
        angle = math.degrees(math.asin(x))
        conf = max(1e-6, peak) * weight
        angle_votes.append((angle, conf))
        debug_rows.append((i, j, tau, peak, angle))

    angle_num = sum(a * w for a, w in angle_votes)
    angle_den = sum(w for _, w in angle_votes)
    angle_deg = angle_num / angle_den if angle_den > 0 else 0.0
    return angle_deg, debug_rows


def classify_lr_center(angle_deg, center_threshold_deg):
    angle_deg = float(angle_deg)
    thr = abs(float(center_threshold_deg))
    if angle_deg > thr:
        return "right"
    if angle_deg < -thr:
        return "left"
    return "center"


def classify_with_level(angle_deg, active_audio, center_threshold_deg, speech_rms_threshold):
    level = rms_level(active_audio)
    if level < float(speech_rms_threshold):
        return "silence", level
    return classify_lr_center(angle_deg, center_threshold_deg), level


def main():
    ap = argparse.ArgumentParser("4-mic rough horizontal DOA")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", type=str, default="", help="Path to multichannel wav")
    src.add_argument("--device", type=int, default=None, help="sounddevice input id")
    ap.add_argument("--sampleRate", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=2.0, help="Record duration / refresh period for --device mode")
    ap.add_argument("--alsaCard", type=str, default="", help="Temporarily redirect ALSA default input to this card name, e.g. AIUIUSBMC")
    ap.add_argument("--channels", type=str, default="0,1,2,3", help="4 effective channel indices, left->right")
    ap.add_argument("--activeWindowSec", type=float, default=0.4, help="Use the loudest short window for DOA")
    ap.add_argument("--micSpacingM", type=float, default=0.04, help="Adjacent mic spacing in meters")
    ap.add_argument("--soundSpeed", type=float, default=343.0)
    ap.add_argument("--centerThresholdDeg", type=float, default=10.0, help="Abs(angle)<=threshold => center")
    ap.add_argument("--speechLowHz", type=float, default=300.0, help="Speech band low cutoff for anti-noise windowing")
    ap.add_argument("--speechHighHz", type=float, default=3400.0, help="Speech band high cutoff for anti-noise windowing")
    ap.add_argument("--speechRmsThreshold", type=float, default=0.006, help="Only report direction when speech-band RMS exceeds this threshold")
    ap.add_argument("--printSilence", action="store_true", help="Also print silence windows below threshold")
    args = ap.parse_args()

    ch_idx = parse_channels(args.channels)

    if args.alsaCard:
        cfg_path = prepare_temp_alsa_default(
            card_name=args.alsaCard,
            channels=max(ch_idx) + 1,
            rate=args.sampleRate,
        )
        print(f"[audio] ALSA default redirected via {cfg_path} -> hw:{args.alsaCard},0")

    def run_once(sr, audio):
        if audio.ndim != 2:
            raise RuntimeError(f"Expected 2D audio array [samples, channels], got shape={audio.shape}")
        if audio.shape[1] <= max(ch_idx):
            raise RuntimeError(f"Audio has only {audio.shape[1]} channels, but --channels={ch_idx}")

        audio4 = audio[:, ch_idx]
        active_raw = select_active_window(
            audio4,
            sr=sr,
            window_sec=args.activeWindowSec,
            speech_low_hz=args.speechLowHz,
            speech_high_hz=args.speechHighHz,
        )
        active = bandpass_multich(active_raw, sr=sr, low_hz=args.speechLowHz, high_hz=args.speechHighHz)
        angle_deg, debug_rows = estimate_angle_deg(
            active,
            sr=sr,
            spacing_m=float(args.micSpacingM),
            sound_speed=float(args.soundSpeed),
        )
        return angle_deg, debug_rows, active

    if args.wav:
        sr, audio = read_wav_channels(args.wav)
        angle_deg, debug_rows, active = run_once(sr, audio)
        direction, level = classify_with_level(
            angle_deg,
            active,
            args.centerThresholdDeg,
            args.speechRmsThreshold,
        )
        print(f"sample_rate={sr}")
        print(f"used_channels={ch_idx}")
        print(f"adjacent_spacing_m={args.micSpacingM}")
        print(f"speech_rms={level:.6f}")
        print(f"direction={direction}")
        print(f"angle_deg={angle_deg:.1f}")
        print("pair_debug:")
        for i, j, tau, peak, angle in debug_rows:
            print(f"  ch{i}-ch{j}: tau={tau:+.6f}s peak={peak:.4f} angle={angle:+.1f}deg")
        return

    print(f"sample_rate={args.sampleRate}")
    print(f"used_channels={ch_idx}")
    print(f"adjacent_spacing_m={args.micSpacingM}")
    print(f"refresh_every_sec={args.duration}")
    print(f"center_threshold_deg={args.centerThresholdDeg}")
    print(f"speech_band_hz=({args.speechLowHz}, {args.speechHighHz})")
    print(f"speech_rms_threshold={args.speechRmsThreshold}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            t0 = time.strftime("%H:%M:%S")
            sr, audio = record_from_device(args.device, args.sampleRate, max(ch_idx) + 1, args.duration)
            angle_deg, debug_rows, active = run_once(sr, audio)
            direction, level = classify_with_level(
                angle_deg,
                active,
                args.centerThresholdDeg,
                args.speechRmsThreshold,
            )
            if direction == "silence" and not args.printSilence:
                continue
            print(f"[{t0}] direction={direction} angle_deg={angle_deg:.1f} speech_rms={level:.6f}")
            for i, j, tau, peak, angle in debug_rows:
                print(f"  ch{i}-ch{j}: tau={tau:+.6f}s peak={peak:.4f} angle={angle:+.1f}deg")
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
