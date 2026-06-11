#!/usr/bin/env python3
"""
latency_measure.py

测量机器人手臂 (ArmState) 消息的单程通信延迟。
原理: latency = 电脑收到消息的本地时间 - 消息中 time_stamp 字段(机器人发送时间)

前提: 机器人和电脑需通过 NTP 等方式保持时钟同步。

用法:
    ros2 run rosci_robot_message latency_measure
    # 或者直接运行:
    python3 examples/latency_measure.py
"""

import time
import statistics
from collections import deque

import rclpy
from rclpy.node import Node
from rosci_robot_message.msg import ArmState


class LatencyMeasureNode(Node):
    def __init__(self):
        super().__init__("latency_measure")

        # 参数
        self.declare_parameter("state_topic", "/rosci_arm_state")
        self.declare_parameter("window_size", 100)
        self.declare_parameter("print_interval", 1.0)

        topic = self.get_parameter("state_topic").value
        window_size = self.get_parameter("window_size").value
        print_interval = self.get_parameter("print_interval").value

        self._latencies = deque(maxlen=window_size)
        self._total_count = 0
        self._last_latency = 0.0

        self.create_subscription(ArmState, topic, self._on_state, 10)
        self.create_timer(print_interval, self._print_stats)

        self.get_logger().info(f"开始监听 {topic}，窗口大小={window_size}")
        self.get_logger().info("等待 ArmState 消息...")

    def _on_state(self, msg: ArmState):
        local_time_us = time.time() * 1_000_000.0
        robot_time_us = float(msg.time_stamp)
        latency = (local_time_us - robot_time_us) / 1000.0  # 转为 ms

        self._latencies.append(latency)
        self._last_latency = latency
        self._total_count += 1

    def _print_stats(self):
        n = len(self._latencies)
        if n == 0:
            return

        data = list(self._latencies)
        avg = statistics.mean(data)
        lo = min(data)
        hi = max(data)
        med = statistics.median(data)
        std = statistics.stdev(data) if n >= 2 else 0.0

        self.get_logger().info(
            f"[n={self._total_count}] "
            f"最新={self._last_latency:.2f}ms | "
            f"均值={avg:.2f}ms | 中位={med:.2f}ms | "
            f"最小={lo:.2f}ms | 最大={hi:.2f}ms | "
            f"标准差={std:.2f}ms"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LatencyMeasureNode()
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
