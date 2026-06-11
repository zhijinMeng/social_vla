#!/usr/bin/env python3
"""Quick smoke test: connect to Qwen-Omni Realtime API and get one text reply."""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse live_demo env loader
from social_vla.pipeline.live_demo import _load_env_local  # noqa: E402

_load_env_local()


def main() -> None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit(
            "DASHSCOPE_API_KEY 未设置。\n"
            "  在 Research/.env.local 写入: DASHSCOPE_API_KEY=sk-...\n"
            "  或: export DASHSCOPE_API_KEY=sk-..."
        )

    from social_vla.dialogue.qwen_omni_adapter import QwenOmniAdapter, QwenOmniConfig

    done = threading.Event()
    reply_holder: list[str] = []

    def on_reply(text: str) -> None:
        reply_holder.append(text)
        done.set()

    cfg = QwenOmniConfig(mock=False, on_reply=on_reply)
    adapter = QwenOmniAdapter(cfg)
    print("Connecting to Qwen-Omni Realtime...")
    adapter.start()

    try:
        adapter._send_text_prompt("请用一句话说：你好，API 连接成功。")
        print("Waiting for reply (timeout 30s)...")
        if not done.wait(timeout=30.0):
            raise SystemExit("超时：30s 内未收到回复，请检查网络或 API Key 区域。")
        print(f"\n✓ API OK\n  Reply: {reply_holder[0]}\n")
    finally:
        adapter.stop()


if __name__ == "__main__":
    main()
