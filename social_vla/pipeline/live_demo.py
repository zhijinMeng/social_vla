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

from social_vla.perception.audio_io import MicCapture, make_vad
from social_vla.pipeline.layer_a import LayerAConfig, LayerAPipeline, default_yolo_weights, open_video_source
from social_vla.types import FrameTick


def draw_overlay(frame: np.ndarray, tick: FrameTick, rms: float = 0.0) -> np.ndarray:
    vis = frame.copy()
    score_by_id = {s.track_id: s for s in tick.scores}
    for person in tick.persons:
        x1, y1, x2, y2 = person.bbox.as_xyxy_int()
        sc = score_by_id.get(person.track_id)
        eng = sc.engagement if sc else 0.0
        color = (0, int(180 + 75 * eng), int(255 - 200 * eng))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"id{person.track_id} eng={eng:.2f}"
        if sc:
            label += f" lam={sc.lam_prob:.2f} talk={sc.talknet_prob:.2f}"
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    mic_color = (0, 0, 255) if tick.vad_active else (120, 120, 120)
    header = f"state={tick.session_state.value} vad={int(tick.vad_active)} rms={rms:.0f} persons={len(tick.persons)}"
    if tick.trigger:
        header += f" | {tick.trigger}"
    cv2.putText(vis, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.circle(vis, (vis.shape[1] - 28, 28), 12, mic_color, -1 if tick.vad_active else 2)
    cv2.putText(vis, "q=quit r=reset session", (12, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="SocialVLA live person + speech detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--video", default="", help="Video file path (overrides --camera)")
    parser.add_argument("--mic", type=int, default=None, help="Input device index (default system mic)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--mock-models", action="store_true", help="Skip TalkNet/LAM weights")
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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
