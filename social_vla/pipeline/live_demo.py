#!/usr/bin/env python3
"""Live Layer A demo: webcam + mic + YOLO real-person detection."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_env_local() -> None:
    """Load KEY=VALUE lines from Research/.env.local into os.environ (no overwrite)."""
    import os

    for path in (_ROOT.parent / ".env.local", _ROOT / ".env.local"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("'\"")
            os.environ.setdefault(key, val)
        break


_load_env_local()

from social_vla.perception.audio_io import MicCapture, make_vad
from social_vla.pipeline.layer_a import LayerAConfig, LayerAPipeline, default_yolo_weights, open_video_source
from social_vla.types import FrameTick

def _engagement_color(eng: float) -> tuple[int, int, int]:
    """BGR: low=orange, high=green."""
    return (0, int(180 + 75 * eng), int(255 - 200 * eng))


def draw_overlay(frame: np.ndarray, tick: FrameTick, rms: float = 0.0) -> np.ndarray:
    vis = frame.copy()
    score_by_id = {s.track_id: s for s in tick.scores}
    for person in tick.persons:
        sc = score_by_id.get(person.track_id)
        eng = sc.engagement if sc else 0.0
        body_color = _engagement_color(eng)

        bx1, by1, bx2, by2 = person.bbox.as_xyxy_int()
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), body_color, 2)
        body_label = f"id{person.track_id} eng={eng:.2f}"
        if sc:
            body_label += f" dwell={sc.dwell_time:.1f}"
        cv2.putText(
            vis, body_label, (bx1, max(20, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, body_color, 2,
        )

        if sc and sc.face_bbox is not None:
            fx1, fy1, fx2, fy2 = sc.face_bbox.as_xyxy_int()
            face_color = (255, 200, 0)  # cyan-ish in BGR
            cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), face_color, 2)
            face_label = f"face lam={sc.lam_prob:.2f} talk={sc.talknet_prob:.2f}"
            if sc.ready_lam:
                face_label += " L"
            if sc.ready_talknet:
                face_label += " T"
            cv2.putText(
                vis, face_label, (fx1, min(vis.shape[0] - 8, fy2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2,
            )

    mic_color = (0, 0, 255) if tick.vad_active else (120, 120, 120)
    header = f"state={tick.session_state.value} vad={int(tick.vad_active)} rms={rms:.0f} persons={len(tick.persons)}"
    if tick.trigger:
        label = "WOULD_TRIGGER" if tick.debug.get("perception_only") else "TRIGGER"
        header += f" | {label}={tick.trigger}"
    cv2.putText(vis, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(
        vis, "body=engagement  face=lam/talk  L/T=buffer ready",
        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
    )
    cv2.circle(vis, (vis.shape[1] - 28, 28), 12, mic_color, -1 if tick.vad_active else 2)
    cv2.putText(vis, "q=quit r=reset session", (12, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return vis


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SocialVLA live person + speech detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--video", default="", help="Video file path (overrides --camera)")
    parser.add_argument("--mic", type=int, default=None, help="Input device index (default system mic)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    model_w = parser.add_mutually_exclusive_group()
    model_w.add_argument("--mock-models", action="store_true", help="Skip TalkNet/LAM weights (mock scores)")
    model_w.add_argument(
        "--real-weights",
        action="store_true",
        help="Load TalkNet/LAM checkpoints (default when --mock-models is not set)",
    )
    parser.add_argument("--no-mic", action="store_true", help="Disable microphone (vision only)")
    parser.add_argument(
        "--vad-backend",
        default="silero",
        choices=("silero", "energy"),
        help="silero=PERCY同款; energy=简单RMS门限",
    )
    parser.add_argument("--silero-threshold", type=float, default=0.5, help="同 live_session.launch")
    parser.add_argument("--end-silence-ms", type=int, default=550, help="同 PERCY end_silence_sec 默认 0.55s")
    parser.add_argument("--audio-gain", type=float, default=1.2, help="同 PERCY audio_gain")
    parser.add_argument("--vad-threshold", type=float, default=450.0, help="仅 energy 后端")
    parser.add_argument(
        "--perception-only",
        action="store_true",
        help="Only scores, no auto session/trigger (good for testing speech)",
    )
    parser.add_argument("--no-preview", action="store_true", help="Terminal only, no OpenCV window")
    parser.add_argument("--width", type=int, default=0, help="Capture width (0=default)")
    parser.add_argument("--height", type=int, default=0, help="Capture height (0=default)")
    parser.add_argument("--score-threshold", type=float, default=0.55, help="Engagement trigger threshold (default 0.55)")
    # Qwen-Omni dialogue
    parser.add_argument("--qwen", action="store_true", help="Enable Qwen-Omni real dialogue (requires DASHSCOPE_API_KEY)")
    parser.add_argument("--qwen-voice", default="Serena", help="Qwen-Omni voice (default: Serena)")
    parser.add_argument("--qwen-mock", action="store_true", help="Use mock Qwen replies (no API key needed)")
    args = parser.parse_args(argv)

    if args.list_devices:
        MicCapture.list_devices()
        return

    try:
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            print("warn: CUDA unavailable, falling back to cpu")
            args.device = "cpu"
    except ImportError:
        args.device = "cpu"

    cfg = LayerAConfig(
        hz=args.hz,
        mock_detector=False,
        mock_models=args.mock_models,
        device=args.device,
        yolo_model=default_yolo_weights(),
        perception_only=args.perception_only,
        score_threshold=args.score_threshold,
    )

    # Qwen-Omni dialogue adapter
    if args.qwen or args.qwen_mock:
        import os
        from social_vla.dialogue.qwen_omni_adapter import QwenOmniConfig

        def _on_reply(text: str) -> None:
            print(f"\n  [🤖 Robot says] {text}\n")

        def _on_partial(chunk: str) -> None:
            print(chunk, end="", flush=True)

        cfg.qwen_omni = QwenOmniConfig(
            mock=args.qwen_mock,
            voice=args.qwen_voice,
            on_reply=_on_reply,
            on_partial=_on_partial,
        )
    pipe = LayerAPipeline(cfg)

    mic = None
    vad = None
    if not args.no_mic and not args.video:
        try:
            mic = MicCapture(hz=args.hz, device=args.mic)
            try:
                vad = make_vad(
                    args.vad_backend,
                    threshold=args.silero_threshold,
                    min_silence_ms=args.end_silence_ms,
                    audio_gain=args.audio_gain,
                    energy_threshold=args.vad_threshold,
                )
            except Exception as e:
                print(f"warn: {args.vad_backend} VAD failed ({e}), fallback to energy VAD")
                vad = make_vad("energy", energy_threshold=args.vad_threshold)
            vad_name = "silero" if vad.__class__.__name__ == "PercySileroVAD" else "energy"
            print(
                f"mic: [{mic.device_index}] {mic.device_name} | "
                f"capture={mic.capture_rate}Hz -> 16kHz | vad={vad_name} | gain={args.audio_gain}"
            )
            print("  若无声音: python social_vla/pipeline/live_demo.py --list-devices  然后 --mic <编号>")
        except ImportError as e:
            print(f"warn: mic unavailable ({e}) → pip install sounddevice silero-vad")
        except Exception as e:
            print(f"warn: mic open failed ({e})")

    if args.video:
        cap = open_video_source(args.video)
        source_name = args.video
    else:
        cap = open_video_source(args.camera)
        source_name = f"camera:{args.camera}"
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {source_name}")

    print(
        f"live: yolo={cfg.yolo_model} | device={cfg.device} | "
        f"models={'mock' if cfg.mock_models else 'real'} | source={source_name}"
    )
    print("Stand in front of camera and speak. Preview: q=quit, r=reset session.")

    if cfg.perception_only:
        print("perception-only: 只打分，WOULD_TRIGGER 不会改 state（试完整流程请去掉 --perception-only）")

    period = 1.0 / cfg.hz
    silent_frames = 0
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed, exiting.")
                break

            rms = 0.0
            if mic is not None and vad is not None:
                pcm = mic.read_chunk()
                if pcm.size:
                    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
                speech = vad.feed(pcm)
                if hasattr(vad, "last_rms") and vad.last_rms > 0:
                    rms = max(rms, vad.last_rms)
                pipe.set_vad(speech, t0)
                pipe.push_audio_pcm(pcm)
                if rms < 5.0:
                    silent_frames += 1
                    if silent_frames == 50:
                        print(
                            "warn: 连续 50 帧 rms≈0，麦克风可能未选对或未授权。"
                            " 运行: python social_vla/pipeline/live_demo.py --list-devices"
                        )
                else:
                    silent_frames = 0
            elif not args.no_mic and not args.video:
                silent_frames += 1

            tick = pipe.process_frame(frame, t0)
            tick.debug["audio_rms"] = rms
            pipe._print_tick(tick)

            if not args.no_preview:
                vis = draw_overlay(frame, tick, rms=rms)
                cv2.imshow("SocialVLA Layer A", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    pipe.reset_session()
                    print("session reset -> idle")

            elapsed = time.time() - t0
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            pipe.close()
        except (KeyboardInterrupt, Exception):
            pass


if __name__ == "__main__":
    main()
