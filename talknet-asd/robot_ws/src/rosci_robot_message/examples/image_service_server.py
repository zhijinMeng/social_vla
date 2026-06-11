#!/usr/bin/env python3
"""
image_service_server.py

在机器人端运行：本地订阅指定的相机话题，缓存每个话题的最新一帧，
通过 ROS2 Service (/get_image) 供远端按需获取。

用法:
  # 订阅单个话题
  python3 examples/image_service_server.py \
    /camera/head/depth/forward,sensor_msgs/msg/Image,best_effort

  # 订阅多个话题
  python3 examples/image_service_server.py \
    /camera/head/depth/forward,sensor_msgs/msg/Image,best_effort \
    /camera/head/depth/image_raw,sensor_msgs/msg/Image,best_effort \
    /camera/head/color/image_raw,sensor_msgs/msg/Image,best_effort

  格式: topic名,消息类型,qos(reliable/best_effort)

远端获取:
  python3 examples/image_service_client.py /camera/head/depth/forward
"""

import argparse
import threading
from importlib import import_module

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image as SensorImage
from rosci_robot_message.srv import GetImage


def _load_msg_class(msg_type: str):
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
    results = []
    args = [a.strip() for a in raw_args if a.strip()]
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
        i = 0
        while i < len(args):
            if i + 1 >= len(args):
                raise ValueError("参数不完整，至少需要 topic 和 msg_type")
            topic = args[i]
            msg_type = args[i + 1]
            qos_str = "best_effort"
            if i + 2 < len(args) and args[i + 2] in ("reliable", "best_effort"):
                qos_str = args[i + 2]
                i += 3
            else:
                i += 2
            results.append((topic, msg_type, qos_str))
    return results


class ImageServiceServer(Node):

    def __init__(self, topic_specs: list):
        super().__init__("image_service_server")

        self._lock = threading.Lock()
        self._latest_images = {}   # topic_name -> SensorImage
        self._subscribed_topics = set()

        parsed = _parse_topic_specs(topic_specs)

        for topic, msg_type, qos_str in parsed:
            msg_class = _load_msg_class(msg_type)
            qos = _make_qos(qos_str)

            self._subscribed_topics.add(topic)

            self.create_subscription(
                msg_class, topic,
                lambda msg, t=topic: self._on_msg(msg, t),
                qos,
            )
            self.get_logger().info(f"  订阅: {topic} [{msg_class.__name__}] QoS={qos_str}")

        self._srv = self.create_service(GetImage, "get_image", self._handle_get_image)
        self.get_logger().info(f"服务 /get_image 已启动，共订阅 {len(parsed)} 个话题")

    def _on_msg(self, msg, topic: str):
        """收到消息时，转换为 sensor_msgs/Image 并缓存"""
        img = SensorImage()

        if isinstance(msg, SensorImage):
            img = msg
        else:
            # 兼容其他 Image 类型（如 rosci_robot_message/msg/Image）
            img.header.stamp = self.get_clock().now().to_msg()
            if hasattr(msg, 'height'):
                img.height = msg.height
            if hasattr(msg, 'width'):
                img.width = msg.width
            if hasattr(msg, 'encoding'):
                encoding_map = {0: "unknown", 1: "mono8", 2: "bgr8", 3: "mono16", 4: "jpeg", 5: "png"}
                if isinstance(msg.encoding, int):
                    img.encoding = encoding_map.get(msg.encoding, f"type_{msg.encoding}")
                else:
                    img.encoding = str(msg.encoding)
            if hasattr(msg, 'is_bigendian'):
                img.is_bigendian = msg.is_bigendian
            if hasattr(msg, 'step'):
                img.step = msg.step
            if hasattr(msg, 'data'):
                img.data = bytes(msg.data)

        with self._lock:
            self._latest_images[topic] = img

    def _handle_get_image(self, request, response):
        topic = request.topic_name

        if topic not in self._subscribed_topics:
            response.success = False
            response.message = f"话题 '{topic}' 未被订阅。已订阅: {list(self._subscribed_topics)}"
            return response

        with self._lock:
            img = self._latest_images.get(topic)

        if img is None:
            response.success = False
            response.message = f"话题 '{topic}' 尚未收到数据"
            return response

        response.success = True
        response.message = f"OK: {img.width}x{img.height} {img.encoding} ({len(img.data)} bytes)"
        response.image = img
        self.get_logger().info(f"响应请求: {topic} -> {img.width}x{img.height} ({len(img.data)} bytes)")
        return response


def main():
    parser = argparse.ArgumentParser(
        description="ROS2 图像服务端 - 缓存相机话题最新帧，供远端按需获取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python3 examples/image_service_server.py \\
    /camera/head/depth/forward,sensor_msgs/msg/Image,best_effort \\
    /camera/head/depth/image_raw,sensor_msgs/msg/Image,best_effort
""",
    )
    parser.add_argument(
        "topics", nargs="+",
        help="话题规格: topic名,消息类型,qos(reliable/best_effort)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = ImageServiceServer(args.topics)

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
