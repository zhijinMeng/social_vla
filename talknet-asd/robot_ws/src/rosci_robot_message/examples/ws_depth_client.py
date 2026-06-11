#!/usr/bin/env python3
"""
ws_depth_client.py

PC side:
- Connect WebSocket server
- Receive depth frames
- Save to local buffer directory in Downloads

Input frame format:
  [4-byte big-endian meta_len][meta_json_utf8][raw_image_bytes]
"""

import argparse
import asyncio
import collections
import json
import os
import struct
import time
from pathlib import Path

try:
    import websockets
except Exception as e:
    raise RuntimeError(
        "Missing dependency 'websockets'. Install with: pip3 install websockets"
    ) from e


def parse_frame(payload: bytes):
    if len(payload) < 4:
        raise ValueError("payload too short")
    (meta_len,) = struct.unpack("!I", payload[:4])
    if len(payload) < 4 + meta_len:
        raise ValueError("invalid meta length")
    meta_bytes = payload[4:4 + meta_len]
    data = payload[4 + meta_len:]
    meta = json.loads(meta_bytes.decode("utf-8"))
    return meta, data


def cleanup_old(out_dir: Path, keep: int):
    files = sorted(out_dir.glob("frame_*.bin"))
    extra = len(files) - keep
    if extra <= 0:
        return
    for f in files[:extra]:
        j = f.with_suffix(".json")
        try:
            f.unlink(missing_ok=True)
            j.unlink(missing_ok=True)
        except Exception:
            pass


def calc_latency_ms(meta: dict):
    stamp_sec = int(meta.get("stamp_sec", 0))
    stamp_nanosec = int(meta.get("stamp_nanosec", 0))
    if stamp_sec <= 0:
        return None
    sent_ns = stamp_sec * 1_000_000_000 + stamp_nanosec
    now_ns = time.time_ns()
    return (now_ns - sent_ns) / 1_000_000.0


async def run_client(args):
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[client] save dir: {out_dir}")
    print(f"[client] connect: {args.url}")
    print("[client] latency based on sender stamp; host clocks should be synchronized")

    rx = 0
    t0 = time.time()
    latency_hist = collections.deque(maxlen=args.latency_window)

    async with websockets.connect(args.url, max_size=None, ping_interval=20, ping_timeout=20) as ws:
        async for msg in ws:
            if not isinstance(msg, (bytes, bytearray)):
                continue
            try:
                meta, data = parse_frame(bytes(msg))
            except Exception as e:
                print(f"[client] parse failed: {e}")
                continue

            rx += 1
            stamp_sec = int(meta.get("stamp_sec", 0))
            stamp_nanosec = int(meta.get("stamp_nanosec", 0))
            name = f"frame_{stamp_sec}_{stamp_nanosec}_{rx:06d}"
            bin_path = out_dir / f"{name}.bin"
            json_path = out_dir / f"{name}.json"

            try:
                bin_path.write_bytes(data)
                json_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
            except Exception as e:
                print(f"[client] save failed: {e}")
                continue

            cleanup_old(out_dir, args.keep)
            latency_ms = calc_latency_ms(meta)
            if latency_ms is not None:
                latency_hist.append(latency_ms)

            if rx % args.print_every == 0:
                dt = max(time.time() - t0, 1e-6)
                fps = rx / dt
                base = (
                    f"[client] rx={rx} fps={fps:.2f} "
                    f"size={meta.get('width')}x{meta.get('height')} "
                    f"encoding={meta.get('encoding')} bytes={len(data)}"
                )
                if latency_hist:
                    cur = latency_hist[-1]
                    avg = sum(latency_hist) / len(latency_hist)
                    lo = min(latency_hist)
                    hi = max(latency_hist)
                    print(
                        f"{base} latency_ms(cur/avg/min/max)="
                        f"{cur:.1f}/{avg:.1f}/{lo:.1f}/{hi:.1f}"
                    )
                else:
                    print(f"{base} latency_ms=NA")


def main():
    parser = argparse.ArgumentParser(description="WebSocket depth receiver and local buffer saver")
    parser.add_argument("--url", default="ws://172.16.3.49:8765", help="WebSocket server url")
    parser.add_argument("--out-dir", default="~/Downloads/depth_buffer", help="Local buffer directory")
    parser.add_argument("--keep", type=int, default=200, help="Keep latest N frames")
    parser.add_argument("--print-every", type=int, default=10, help="Print stats every N frames")
    parser.add_argument("--latency-window", type=int, default=50, help="Moving latency window size")
    args = parser.parse_args()

    try:
        asyncio.run(run_client(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
