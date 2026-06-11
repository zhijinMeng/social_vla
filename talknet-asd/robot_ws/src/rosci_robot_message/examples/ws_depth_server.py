#!/usr/bin/env python3
"""
ws_depth_server.py

Robot side:
- Subscribe ROS2 depth image topic
- Push frames to connected WebSocket clients over TCP

Binary frame format:
  [4-byte big-endian meta_len][meta_json_utf8][raw_image_bytes]
"""

import argparse
import asyncio
import json
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, Set

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

try:
    import websockets
except Exception as e:
    raise RuntimeError(
        "Missing dependency 'websockets'. Install with: pip3 install websockets"
    ) from e


@dataclass
class FramePacket:
    meta: dict
    data: bytes


class DepthRelayNode(Node):
    def __init__(self, topic: str, qos_mode: str, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        super().__init__("ws_depth_server")
        self._topic = topic
        self._loop = loop
        self._queue = queue
        self._rx_count = 0
        self._drop_count = 0
        self._last_warn = 0.0

        qos = QoSProfile(
            reliability=(
                ReliabilityPolicy.RELIABLE if qos_mode == "reliable" else ReliabilityPolicy.BEST_EFFORT
            ),
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Image, topic, self._on_image, qos)
        self.create_timer(2.0, self._print_status)
        self.get_logger().info(f"Subscribe topic: {topic}, qos={qos_mode}")

    def _on_image(self, msg: Image):
        self._rx_count += 1
        meta = {
            "topic": self._topic,
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nanosec": int(msg.header.stamp.nanosec),
            "width": int(msg.width),
            "height": int(msg.height),
            "encoding": str(msg.encoding),
            "is_bigendian": int(msg.is_bigendian),
            "step": int(msg.step),
            "bytes": len(msg.data),
            "rx_seq": self._rx_count,
        }
        packet = FramePacket(meta=meta, data=bytes(msg.data))

        def _enqueue():
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
                self._drop_count += 1
            try:
                self._queue.put_nowait(packet)
            except Exception:
                self._drop_count += 1

        self._loop.call_soon_threadsafe(_enqueue)

    def _print_status(self):
        now = time.time()
        if self._drop_count > 0 and now - self._last_warn > 2.0:
            self.get_logger().warn(
                f"Status: rx={self._rx_count}, queue_drop={self._drop_count}"
            )
            self._last_warn = now
        else:
            self.get_logger().info(f"Status: rx={self._rx_count}, queue_drop={self._drop_count}")


class WsBroadcaster:
    def __init__(self):
        self.clients: Set["websockets.WebSocketServerProtocol"] = set()
        self._lock = asyncio.Lock()
        self.tx_count = 0

    async def register(self, ws):
        async with self._lock:
            self.clients.add(ws)

    async def unregister(self, ws):
        async with self._lock:
            self.clients.discard(ws)

    async def broadcast(self, payload: bytes):
        async with self._lock:
            clients = list(self.clients)
        if not clients:
            return
        dead = []
        for ws in clients:
            try:
                await ws.send(payload)
                self.tx_count += 1
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self.clients.discard(ws)


def pack_frame(packet: FramePacket) -> bytes:
    meta_bytes = json.dumps(packet.meta, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return struct.pack("!I", len(meta_bytes)) + meta_bytes + packet.data


async def run_server(args):
    queue: asyncio.Queue = asyncio.Queue(maxsize=args.queue_size)
    loop = asyncio.get_running_loop()
    relay_node = DepthRelayNode(args.topic, args.qos, loop, queue)

    broadcaster = WsBroadcaster()

    async def ws_handler(ws):
        await broadcaster.register(ws)
        try:
            async for _ in ws:
                # No command protocol now; server push only.
                pass
        finally:
            await broadcaster.unregister(ws)

    ws_server = await websockets.serve(
        ws_handler,
        args.host,
        args.port,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    )
    relay_node.get_logger().info(f"WebSocket listen on ws://{args.host}:{args.port}")

    spinner_stop = threading.Event()

    def _spin():
        while rclpy.ok() and not spinner_stop.is_set():
            rclpy.spin_once(relay_node, timeout_sec=0.1)

    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    try:
        while True:
            packet: FramePacket = await queue.get()
            payload = pack_frame(packet)
            await broadcaster.broadcast(payload)
    finally:
        spinner_stop.set()
        ws_server.close()
        await ws_server.wait_closed()
        relay_node.destroy_node()


def main():
    parser = argparse.ArgumentParser(description="ROS depth -> WebSocket stream server")
    parser.add_argument("--topic", default="/camera/head/depth/image_raw", help="ROS image topic")
    parser.add_argument("--qos", choices=["reliable", "best_effort"], default="reliable", help="ROS subscribe qos")
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket bind host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket bind port")
    parser.add_argument("--queue-size", type=int, default=3, help="Internal async queue size")
    args = parser.parse_args()

    rclpy.init()
    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

