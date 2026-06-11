#!/usr/bin/env python3
"""一键启动：摄像头真人检测 + Qwen-Omni 真实对话。

用法:
    python run_live.py              # 默认摄像头 0、麦克风 4（ALC294）
    python run_live.py --mic 7      # 换麦克风
    python run_live.py --list-devices
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings

warnings.filterwarnings("ignore", message=".*was resized since it had shape.*")


def main() -> None:
    parser = argparse.ArgumentParser(description="SocialVLA 真人对话（一键启动）")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号（默认 0）")
    parser.add_argument("--mic", type=int, default=4, help="麦克风编号（默认 4 = ALC294）")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备后退出")
    args = parser.parse_args()

    from social_vla.pipeline.live_demo import main as live_main

    live_argv = [
        "--camera",
        str(args.camera),
        "--mic",
        str(args.mic),
        "--device",
        "cuda",
        "--real-weights",
        "--vad-backend",
        "silero",
        "--audio-gain",
        "2.0",
        "--qwen",
    ]
    if args.list_devices:
        live_argv.append("--list-devices")

    print("SocialVLA 真人对话 | q=退出 r=重置 | 站在摄像头前说话")
    live_main(live_argv)


if __name__ == "__main__":
    main()
