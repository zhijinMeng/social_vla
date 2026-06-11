#!/usr/bin/env python3
"""
dds_latency_measure.py

专门测量 ROS2 DDS 中间件的通信延迟，隔离掉机器人内部处理时间。

两种模式:
  1) loopback  — 同一进程内 pub→sub 回环，测 DDS 本地序列化/反序列化 + 传输开销
  2) ping-pong — 双节点往返，A发ping → B收到后立即发pong → A收到，RTT/2 = 单程DDS延迟
                 （不需要时钟同步）

支持所有带 time_stamp 字段的消息类型，通过 --msg 参数选择。

用法:
  # 模式1: 本机回环测试（默认使用 ArmCommand）
  python3 examples/dds_latency_measure.py

  # 使用指定消息类型
  python3 examples/dds_latency_measure.py --msg MotionPlatformState

  # 模式2: ping端（电脑上运行）
  python3 examples/dds_latency_measure.py --mode ping --msg GripperCommand

  # 模式2: pong端（机器人上运行）
  python3 examples/dds_latency_measure.py --mode pong --msg GripperCommand

  # 测试所有消息类型
  python3 examples/dds_latency_measure.py --msg all
"""

import argparse
import importlib
import time
import statistics
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


# 所有带 uint64 time_stamp 字段的消息
MSG_NAMES = [
    "ArmCommand",
    "ArmState",
    "CameraCommand",
    "CameraRemoteState",
    "CameraState",
    "GripperCommand",
    "GripperState",
    "InferenceResult",
    "MotionPlatformCommand",
    "MotionPlatformState",
    "TelescopicGripperCommand",
    "TelescopicGripperState",
]


def _load_msg_class(name: str):
    """动态导入 rosci_robot_message.msg 中的消息类。"""
    mod = importlib.import_module("rosci_robot_message.msg")
    return getattr(mod, name)


def _now_ms() -> float:
    return time.time() * 1000.0


class LoopbackNode(Node):
    """同一进程 pub→sub 回环，测 DDS 本地开销。"""

    def __init__(self, msg_class, msg_name: str, hz: float, window: int, print_interval: float):
        super().__init__("dds_latency_loopback")

        self._msg_class = msg_class
        self._msg_name = msg_name

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        topic = f"/dds_latency_test/{msg_name}"
        self._pub = self.create_publisher(msg_class, topic, qos)
        self.create_subscription(msg_class, topic, self._on_recv, qos)

        self._latencies = deque(maxlen=window)
        self._total = 0

        self.create_timer(1.0 / hz, self._send)
        self.create_timer(print_interval, self._print_stats)

        self.get_logger().info(
            f"[Loopback] 消息={msg_name}, 频率={hz}Hz, 窗口={window}, "
            f"话题={topic}"
        )

    def _send(self):
        msg = self._msg_class()
        msg.time_stamp = int(_now_ms())
        self._pub.publish(msg)

    def _on_recv(self, msg):
        latency = _now_ms() - float(msg.time_stamp)
        self._latencies.append(latency)
        self._total += 1

    def _print_stats(self):
        n = len(self._latencies)
        if n == 0:
            return
        data = list(self._latencies)
        avg = statistics.mean(data)
        lo, hi = min(data), max(data)
        med = statistics.median(data)
        std = statistics.stdev(data) if n >= 2 else 0.0
        self.get_logger().info(
            f"[Loopback {self._msg_name} n={self._total}] "
            f"最新={data[-1]:.3f}ms | 均值={avg:.3f}ms | 中位={med:.3f}ms | "
            f"最小={lo:.3f}ms | 最大={hi:.3f}ms | 标准差={std:.3f}ms"
        )


class PingNode(Node):
    """Ping端：发送带时间戳的消息，等待Pong回复，计算 RTT/2。"""

    MAX_VALID_RTT_MS = 1000.0

    def __init__(self, msg_class, msg_name: str, hz: float, window: int, print_interval: float, warmup: float):
        super().__init__("dds_latency_ping")

        self._msg_class = msg_class
        self._msg_name = msg_name

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        ping_topic = f"/dds_latency_ping/{msg_name}"
        pong_topic = f"/dds_latency_pong/{msg_name}"
        self._pub = self.create_publisher(msg_class, ping_topic, qos)
        self.create_subscription(msg_class, pong_topic, self._on_pong, qos)

        self._latencies = deque(maxlen=window)
        self._total = 0
        self._dropped = 0

        self._warmup_end = time.time() + warmup
        self._warmed_up = False
        self.create_timer(1.0 / hz, self._send_ping)
        self.create_timer(print_interval, self._print_stats)

        self.get_logger().info(
            f"[Ping] 消息={msg_name}, 频率={hz}Hz, "
            f"发送={ping_topic}, 接收={pong_topic}"
        )
        self.get_logger().info(f"预热中 ({warmup}s)，等待 DDS Discovery 完成...")

    def _send_ping(self):
        if not self._warmed_up:
            if time.time() < self._warmup_end:
                return
            self._warmed_up = True
            self.get_logger().info("预热完成，开始测量!")
        msg = self._msg_class()
        msg.time_stamp = int(_now_ms())
        self._pub.publish(msg)

    def _on_pong(self, msg):
        rtt = _now_ms() - float(msg.time_stamp)
        if rtt > self.MAX_VALID_RTT_MS:
            self._dropped += 1
            return
        self._latencies.append(rtt / 2.0)
        self._total += 1

    def _print_stats(self):
        if not self._warmed_up:
            return
        n = len(self._latencies)
        if n == 0:
            return
        data = list(self._latencies)
        avg = statistics.mean(data)
        lo, hi = min(data), max(data)
        med = statistics.median(data)
        std = statistics.stdev(data) if n >= 2 else 0.0
        extra = f" | 丢弃={self._dropped}" if self._dropped else ""
        self.get_logger().info(
            f"[Ping-Pong {self._msg_name} n={self._total}] "
            f"单程: 最新={data[-1]:.3f}ms | 均值={avg:.3f}ms | 中位={med:.3f}ms | "
            f"最小={lo:.3f}ms | 最大={hi:.3f}ms | 标准差={std:.3f}ms{extra}"
        )


class PongNode(Node):
    """Pong端：收到Ping后立即原样回传（保留原始time_stamp）。"""

    def __init__(self, msg_class, msg_name: str):
        super().__init__("dds_latency_pong")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        ping_topic = f"/dds_latency_ping/{msg_name}"
        pong_topic = f"/dds_latency_pong/{msg_name}"
        self._pub = self.create_publisher(msg_class, pong_topic, qos)
        self.create_subscription(msg_class, ping_topic, self._on_ping, qos)

        self._count = 0
        self.get_logger().info(
            f"[Pong] 消息={msg_name}, "
            f"接收={ping_topic}, 回复={pong_topic}"
        )
        self.get_logger().info("等待 Ping 端消息...")

    def _on_ping(self, msg):
        self._pub.publish(msg)
        self._count += 1
        if self._count % 100 == 0:
            self.get_logger().info(f"已转发 {self._count} 条")


class MultiLoopbackNode(Node):
    """同时测试所有消息类型的回环延迟。"""

    def __init__(self, hz: float, window: int, print_interval: float):
        super().__init__("dds_latency_loopback_all")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._testers = {}
        for name in MSG_NAMES:
            cls = _load_msg_class(name)
            topic = f"/dds_latency_test/{name}"
            pub = self.create_publisher(cls, topic, qos)
            self.create_subscription(
                cls, topic,
                lambda msg, n=name: self._on_recv(msg, n),
                qos,
            )
            self._testers[name] = {
                "class": cls,
                "pub": pub,
                "latencies": deque(maxlen=window),
                "total": 0,
            }

        self.create_timer(1.0 / hz, self._send_all)
        self.create_timer(print_interval, self._print_stats)

        self.get_logger().info(
            f"[Loopback ALL] 测试 {len(MSG_NAMES)} 种消息, 频率={hz}Hz"
        )

    def _send_all(self):
        ts = int(_now_ms())
        for info in self._testers.values():
            msg = info["class"]()
            msg.time_stamp = ts
            info["pub"].publish(msg)

    def _on_recv(self, msg, name: str):
        latency = _now_ms() - float(msg.time_stamp)
        info = self._testers[name]
        info["latencies"].append(latency)
        info["total"] += 1

    def _print_stats(self):
        self.get_logger().info("=" * 70)
        for name, info in self._testers.items():
            n = len(info["latencies"])
            if n == 0:
                self.get_logger().info(f"  {name:30s} | 无数据")
                continue
            data = list(info["latencies"])
            avg = statistics.mean(data)
            med = statistics.median(data)
            lo, hi = min(data), max(data)
            self.get_logger().info(
                f"  {name:30s} | 均值={avg:.3f}ms  中位={med:.3f}ms  "
                f"最小={lo:.3f}ms  最大={hi:.3f}ms  (n={info['total']})"
            )


class MultiPingNode(Node):
    """同时对所有消息类型做 Ping 测试。"""

    MAX_VALID_RTT_MS = 1000.0  # RTT 超过此值视为 discovery 堆积，丢弃

    def __init__(self, hz: float, window: int, print_interval: float, warmup: float):
        super().__init__("dds_latency_ping_all")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._testers = {}
        for name in MSG_NAMES:
            cls = _load_msg_class(name)
            ping_topic = f"/dds_latency_ping/{name}"
            pong_topic = f"/dds_latency_pong/{name}"
            pub = self.create_publisher(cls, ping_topic, qos)
            self.create_subscription(
                cls, pong_topic,
                lambda msg, n=name: self._on_pong(msg, n),
                qos,
            )
            self._testers[name] = {
                "class": cls,
                "pub": pub,
                "latencies": deque(maxlen=window),
                "total": 0,
                "dropped": 0,
            }

        self._warmup_end = time.time() + warmup
        self._warmed_up = False
        self.create_timer(1.0 / hz, self._send_all)
        self.create_timer(print_interval, self._print_stats)

        self.get_logger().info(
            f"[Ping ALL] 测试 {len(MSG_NAMES)} 种消息, 频率={hz}Hz, "
            f"预热={warmup}s"
        )
        self.get_logger().info(f"预热中 ({warmup}s)，等待 DDS Discovery 完成...")

    def _send_all(self):
        if not self._warmed_up:
            if time.time() < self._warmup_end:
                return
            self._warmed_up = True
            self.get_logger().info("预热完成，开始测量!")
        ts = int(_now_ms())
        for info in self._testers.values():
            msg = info["class"]()
            msg.time_stamp = ts
            info["pub"].publish(msg)

    def _on_pong(self, msg, name: str):
        rtt = _now_ms() - float(msg.time_stamp)
        info = self._testers[name]
        if rtt > self.MAX_VALID_RTT_MS:
            info["dropped"] += 1
            return
        info["latencies"].append(rtt / 2.0)
        info["total"] += 1

    def _print_stats(self):
        if not self._warmed_up:
            return
        self.get_logger().info("=" * 70)
        for name, info in self._testers.items():
            n = len(info["latencies"])
            if n == 0:
                dropped = info["dropped"]
                extra = f"  (已丢弃{dropped}条异常)" if dropped else ""
                self.get_logger().info(f"  {name:30s} | 无数据{extra}")
                continue
            data = list(info["latencies"])
            avg = statistics.mean(data)
            med = statistics.median(data)
            lo, hi = min(data), max(data)
            dropped = info["dropped"]
            extra = f"  丢弃={dropped}" if dropped else ""
            self.get_logger().info(
                f"  {name:30s} | 单程: 均值={avg:.3f}ms  中位={med:.3f}ms  "
                f"最小={lo:.3f}ms  最大={hi:.3f}ms  (n={info['total']}){extra}"
            )


class MultiPongNode(Node):
    """同时对所有消息类型做 Pong 回复。"""

    def __init__(self):
        super().__init__("dds_latency_pong_all")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._count = 0
        for name in MSG_NAMES:
            cls = _load_msg_class(name)
            ping_topic = f"/dds_latency_ping/{name}"
            pong_topic = f"/dds_latency_pong/{name}"
            pub = self.create_publisher(cls, pong_topic, qos)
            self.create_subscription(
                cls, ping_topic,
                lambda msg, p=pub: self._on_ping(msg, p),
                qos,
            )

        self.get_logger().info(
            f"[Pong ALL] 监听 {len(MSG_NAMES)} 种消息"
        )
        self.get_logger().info("等待 Ping 端消息...")

    def _on_ping(self, msg, pub):
        pub.publish(msg)
        self._count += 1
        if self._count % 500 == 0:
            self.get_logger().info(f"已转发 {self._count} 条")


def main():
    parser = argparse.ArgumentParser(description="ROS2 DDS 延迟测试工具")
    parser.add_argument(
        "--mode",
        choices=["loopback", "ping", "pong"],
        default="loopback",
        help="测试模式: loopback(本机回环) / ping(发起端) / pong(回复端)",
    )
    parser.add_argument(
        "--msg",
        default="ArmCommand",
        help=f"消息类型名称，或 'all' 测试所有。可选: {', '.join(MSG_NAMES)}",
    )
    parser.add_argument("--hz", type=float, default=100.0, help="发送频率 (默认100Hz)")
    parser.add_argument("--window", type=int, default=500, help="统计窗口大小 (默认500)")
    parser.add_argument("--print-interval", type=float, default=1.0, help="打印间隔秒数")
    args = parser.parse_args()

    test_all = args.msg.lower() == "all"

    if not test_all and args.msg not in MSG_NAMES:
        parser.error(f"未知消息类型 '{args.msg}'。可选: {', '.join(MSG_NAMES)}, all")

    rclpy.init()

    if test_all:
        if args.mode == "loopback":
            node = MultiLoopbackNode(args.hz, args.window, args.print_interval)
        elif args.mode == "ping":
            node = MultiPingNode(args.hz, args.window, args.print_interval)
        else:
            node = MultiPongNode()
    else:
        msg_class = _load_msg_class(args.msg)
        if args.mode == "loopback":
            node = LoopbackNode(msg_class, args.msg, args.hz, args.window, args.print_interval)
        elif args.mode == "ping":
            node = PingNode(msg_class, args.msg, args.hz, args.window, args.print_interval)
        else:
            node = PongNode(msg_class, args.msg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
