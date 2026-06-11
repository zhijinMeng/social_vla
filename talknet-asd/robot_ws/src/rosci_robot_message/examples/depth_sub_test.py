#!/usr/bin/env python3
"""
depth_sub_test.py

最小订阅测试脚本：支持任意消息类型。

用法:
  python3 depth_sub_test.py
  python3 depth_sub_test.py /camera/head/depth/image_raw
  python3 depth_sub_test.py /rosci_arm_command rosci_robot_message/msg/ArmCommand
"""

import argparse
from importlib import import_module

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


def _load_msg_class(msg_type: str):
    parts = msg_type.replace("/", ".")
    module_path = ".".join(parts.split(".")[:-1])
    class_name = parts.split(".")[-1]
    mod = import_module(module_path)
    return getattr(mod, class_name)


class DepthSubTest(Node):
    def __init__(self, topic: str, msg_type: str, qos_mode: str):
        super().__init__("depth_sub_test")
        self._topic = topic
        self._msg_type = msg_type
        self._count = 0

        msg_class = _load_msg_class(msg_type)
        qos = QoSProfile(
            reliability=(
                ReliabilityPolicy.RELIABLE
                if qos_mode == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            ),
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(msg_class, topic, self._on_msg, qos)
        self.create_timer(2.0, self._on_timer)
        self.create_timer(2.0, self._debug_discovery)
        self.get_logger().info(f"开始订阅: {topic} [{msg_class.__name__}] qos={qos_mode}")

    def _on_msg(self, msg):
        self._count += 1
        # 尽量打印通用信息，避免依赖特定消息字段
        extra = ""
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            s = msg.header.stamp
            extra += f" stamp={s.sec}.{s.nanosec:09d}"
        if hasattr(msg, "width") and hasattr(msg, "height"):
            extra += f" size={msg.width}x{msg.height}"
        if hasattr(msg, "encoding"):
            extra += f" encoding={msg.encoding}"
        if hasattr(msg, "data"):
            try:
                extra += f" bytes={len(msg.data)}"
            except Exception:
                pass

        self.get_logger().info(
            f"recv#{self._count} topic={self._topic} type={self._msg_type}{extra}"
        )

    def _on_timer(self):
        if self._count == 0:
            self.get_logger().warn("还没收到任何帧")

    def _debug_discovery(self):
        try:
            infos = self.get_publishers_info_by_topic(self._topic)
        except Exception as e:
            self.get_logger().warn(f"发布者查询失败: {e}")
            return

        pub_count = len(infos)
        if pub_count == 0:
            self.get_logger().warn("DEBUG: Publisher count=0（当前未发现发布者）")
            return

        self.get_logger().info(f"DEBUG: Publisher count={pub_count}")
        for i, info in enumerate(infos[:3], start=1):
            qos = info.qos_profile
            self.get_logger().info(
                f"DEBUG pub#{i}: node={info.node_name} ns={info.node_namespace} "
                f"type={self._msg_type} rel={qos.reliability.name} dur={qos.durability.name}"
            )


def main():
    parser = argparse.ArgumentParser(description="订阅 Image 话题连通性测试")
    parser.add_argument(
        "topic",
        nargs="?",
        default="/camera/head/depth/image_raw",
        help="待订阅话题名，默认 /camera/head/depth/image_raw",
    )
    parser.add_argument(
        "msg_type",
        nargs="?",
        default="sensor_msgs/msg/Image",
        help="消息类型，默认 sensor_msgs/msg/Image",
    )
    parser.add_argument(
        "--qos",
        choices=["reliable", "best_effort"],
        default="reliable",
        help="订阅QoS，默认 reliable",
    )
    args = parser.parse_args()

    rclpy.init()
    node = DepthSubTest(args.topic, args.msg_type, args.qos)
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
