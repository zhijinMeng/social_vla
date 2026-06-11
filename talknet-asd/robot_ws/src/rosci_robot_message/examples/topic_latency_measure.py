#!/usr/bin/env python3
"""
topic_latency_measure.py

订阅指定话题，利用消息中 header.stamp / time_stamp 与本地时间的差值测量单程延迟。
要求发布端和订阅端时钟已通过 NTP 同步。

支持同时测量多个话题。

用法:
  # 测单个话题
  python3 examples/topic_latency_measure.py /camera/head/depth/image_raw,sensor_msgs/msg/Image,best_effort

  # 同时测多个话题
  python3 examples/topic_latency_measure.py \
    /camera/head/depth/forward,sensor_msgs/msg/Image,best_effort \
    /camera/head/depth/image_raw,sensor_msgs/msg/Image,best_effort \
    /robot/arm_joint_states,sensor_msgs/msg/JointState,reliable \
    /nav_to_grasp/command,rosci_robot_message/msg/NavigationCommand,reliable \
    /grasp/status,rosci_robot_message/msg/GraspStatus,reliable

  格式: topic名,消息类型,qos(reliable/best_effort)
"""

import argparse
import statistics
import time
from collections import deque
from importlib import import_module

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


def _load_msg_class(msg_type: str):
    """
    从 'package/msg/Type' 格式加载消息类。
    例如 'sensor_msgs/msg/Image' → sensor_msgs.msg.Image
    """
    parts = msg_type.replace("/", ".")
    module_path = ".".join(parts.split(".")[:-1])
    class_name = parts.split(".")[-1]
    mod = import_module(module_path)
    return getattr(mod, class_name)


def _make_qos(qos_str: str):
    if qos_str == "reliable":
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
    else:
        return QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )


def _parse_topic_specs(raw_args: list):
    """
    智能解析话题参数，支持两种格式:
      1) 逗号格式: topic,msg_type,qos  (一个参数一个话题)
      2) 空格格式: topic msg_type [qos] (兼容旧版，2-3个参数一个话题)
    """
    results = []
    args = [a.strip() for a in raw_args if a.strip()]

    # 判断是否有逗号格式
    if any("," in a for a in args):
        for spec in args:
            parts = spec.split(",")
            if len(parts) < 2:
                raise ValueError(f"格式错误: '{spec}'，应为 topic,msg_type[,qos]")
            topic = parts[0].strip()
            msg_type = parts[1].strip()
            qos_str = parts[2].strip() if len(parts) >= 3 else "best_effort"
            results.append((topic, msg_type, qos_str))
    else:
        # 空格格式: 每 2-3 个参数为一组
        i = 0
        while i < len(args):
            if i + 1 >= len(args):
                raise ValueError(f"参数不完整，至少需要 topic 和 msg_type")
            topic = args[i]
            msg_type = args[i + 1]
            qos_str = "best_effort"
            if i + 2 < len(args) and args[i + 2] in ("reliable", "best_effort"):
                qos_str = args[i + 2]
                i += 3
            else:
                i += 2
            results.append((topic, msg_type, qos_str))

    for _, _, qos_str in results:
        if qos_str not in ("reliable", "best_effort"):
            raise ValueError(f"未知 QoS: '{qos_str}'，应为 reliable 或 best_effort")

    return results


class MultiTopicLatencyNode(Node):

    def __init__(self, topic_specs: list, window: int, print_interval: float, debug_discovery: bool):
        super().__init__("topic_latency_measure")

        self._trackers = {}
        self._debug_discovery = debug_discovery

        parsed = _parse_topic_specs(topic_specs)

        for topic, msg_type, qos_str in parsed:
            msg_class = _load_msg_class(msg_type)
            qos = _make_qos(qos_str)

            self._trackers[topic] = {
                "msg_name": msg_class.__name__,
                "qos": qos_str,
                "latencies": deque(maxlen=window),
                "total": 0,
                "no_stamp": False,
            }

            self.create_subscription(
                msg_class, topic,
                lambda msg, t=topic: self._on_msg(msg, t),
                qos,
            )

            self.get_logger().info(
                f"  订阅: {topic} [{msg_class.__name__}] QoS={qos_str}"
            )

        self.create_timer(print_interval, self._print_stats)
        self.get_logger().info(f"共订阅 {len(self._trackers)} 个话题，等待消息...")

    def _on_msg(self, msg, topic: str):
        now = self.get_clock().now()
        tracker = self._trackers[topic]

        # 尝试多种时间戳字段
        stamp = None
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            s = msg.header.stamp
            stamp = s.sec + s.nanosec * 1e-9
        elif hasattr(msg, 'stamp'):
            s = msg.stamp
            stamp = s.sec + s.nanosec * 1e-9
        elif hasattr(msg, 'time_stamp'):
            # rosci_robot_message 的 uint64 time_stamp (毫秒)
            stamp = msg.time_stamp / 1000.0

        if stamp is None or stamp == 0:
            if not tracker["no_stamp"]:
                self.get_logger().warn(
                    f"{topic}: 消息中没有可用的时间戳字段，仅统计频率"
                )
                tracker["no_stamp"] = True
            tracker["total"] += 1
            return

        now_sec = now.nanoseconds * 1e-9
        latency_ms = (now_sec - stamp) * 1000.0

        # 丢弃明显异常的值（时钟漂移或首包）
        if latency_ms < -500 or latency_ms > 5000:
            return

        tracker["latencies"].append(latency_ms)
        tracker["total"] += 1

    def _print_stats(self):
        self.get_logger().info("=" * 75)
        for topic, tr in self._trackers.items():
            n = len(tr["latencies"])
            total = tr["total"]

            if total == 0:
                self.get_logger().info(f"  {topic:45s} | 无数据")
                if self._debug_discovery:
                    self._print_discovery(topic)
                continue

            if tr["no_stamp"] and n == 0:
                self.get_logger().info(
                    f"  {topic:45s} | 无时间戳，已收到 {total} 条"
                )
                continue

            if n == 0:
                self.get_logger().info(f"  {topic:45s} | 收到 {total} 条，延迟异常已丢弃")
                continue

            data = list(tr["latencies"])
            avg = statistics.mean(data)
            med = statistics.median(data)
            lo, hi = min(data), max(data)
            self.get_logger().info(
                f"  {topic:45s} | 均值={avg:.1f}ms  中位={med:.1f}ms  "
                f"最小={lo:.1f}ms  最大={hi:.1f}ms  (n={total})"
            )

    def _print_discovery(self, topic: str):
        try:
            infos = self.get_publishers_info_by_topic(topic)
        except Exception as e:
            self.get_logger().warn(f"  {topic:45s} | 发布者查询失败: {e}")
            return

        if not infos:
            self.get_logger().warn(f"  {topic:45s} | 当前发现发布者=0")
            return

        self.get_logger().info(f"  {topic:45s} | 当前发现发布者={len(infos)}")
        for i, info in enumerate(infos[:3], start=1):
            qos = info.qos_profile
            self.get_logger().info(
                f"    pub#{i}: node={info.node_name} ns={info.node_namespace} "
                f"rel={qos.reliability.name} dur={qos.durability.name}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="ROS2 多话题延迟测量（基于 header.stamp / time_stamp）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 测单个话题
  python3 examples/topic_latency_measure.py /camera/head/depth/forward,sensor_msgs/msg/Image,best_effort

  # 测多个话题
  python3 examples/topic_latency_measure.py \\
    /camera/head/depth/forward,sensor_msgs/msg/Image,best_effort \\
    /robot/arm_joint_states,sensor_msgs/msg/JointState,reliable \\
    /grasp/status,rosci_robot_message/msg/GraspStatus,reliable
""",
    )
    parser.add_argument(
        "topics", nargs="+",
        help="话题规格: topic名,消息类型,qos(reliable/best_effort)",
    )
    parser.add_argument("--window", type=int, default=200, help="统计窗口大小 (默认200)")
    parser.add_argument("--print-interval", type=float, default=2.0, help="打印间隔秒数 (默认2s)")
    parser.add_argument(
        "--debug-discovery", action="store_true",
        help="无数据时打印当前发现到的发布者信息",
    )
    args = parser.parse_args()

    rclpy.init()
    node = MultiTopicLatencyNode(args.topics, args.window, args.print_interval, args.debug_discovery)

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
