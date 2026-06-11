#!/usr/bin/env python3
"""
depth_image_fetch_server.py

机器人侧按需取图服务:
- 订阅深度图话题并缓存最新一帧
- 收到服务请求后，仅发布一帧到响应话题

用法:
  python3 examples/depth_image_fetch_server.py \
    --source-topic /camera/head/depth/image_raw \
    --service-name /depth_fetch/fetch \
    --output-topic /depth_fetch/image
"""

import argparse
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


class DepthImageFetchServer(Node):
    def __init__(self, source_topic: str, service_name: str, output_topic: str, source_qos: str):
        super().__init__("depth_image_fetch_server")

        sub_reliability = (
            ReliabilityPolicy.RELIABLE
            if source_qos == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        sub_qos = QoSProfile(
            reliability=sub_reliability,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._latest_image: Optional[Image] = None
        self._latest_size = 0
        self._rx_count = 0
        self._tx_count = 0

        self.create_subscription(Image, source_topic, self._on_image, sub_qos)
        self._pub = self.create_publisher(Image, output_topic, pub_qos)
        self.create_service(Trigger, service_name, self._on_fetch_request)
        self.create_timer(5.0, self._log_status)

        self.get_logger().info(f"订阅源图像: {source_topic}")
        self.get_logger().info(f"源图像 QoS: {source_qos}")
        self.get_logger().info(f"服务接口: {service_name}")
        self.get_logger().info(f"输出话题: {output_topic}")
        self.get_logger().info("等待请求...")

    def _on_image(self, msg: Image):
        self._latest_image = msg
        self._latest_size = len(msg.data)
        self._rx_count += 1

    def _on_fetch_request(self, _req: Trigger.Request, resp: Trigger.Response):
        if self._latest_image is None:
            resp.success = False
            resp.message = "尚未收到源图像，无法返回。"
            return resp

        self._pub.publish(self._latest_image)
        self._tx_count += 1

        stamp = self._latest_image.header.stamp
        resp.success = True
        resp.message = (
            f"ok size={self._latest_size}B "
            f"{self._latest_image.width}x{self._latest_image.height} "
            f"encoding={self._latest_image.encoding} "
            f"stamp={stamp.sec}.{stamp.nanosec:09d}"
        )
        return resp

    def _log_status(self):
        cache = "有缓存" if self._latest_image is not None else "无缓存"
        self.get_logger().info(
            f"状态: {cache}, 收到源帧={self._rx_count}, 按需返回={self._tx_count}"
        )


def main():
    parser = argparse.ArgumentParser(description="机器人侧按需深度图服务")
    parser.add_argument("--source-topic", default="/camera/head/depth/image_raw", help="源深度图话题")
    parser.add_argument(
        "--source-qos",
        default="reliable",
        choices=["reliable", "best_effort"],
        help="源话题订阅QoS (默认 reliable)",
    )
    parser.add_argument("--service-name", default="/depth_fetch/fetch", help="请求服务名")
    parser.add_argument("--output-topic", default="/depth_fetch/image", help="返回图像话题")
    args = parser.parse_args()

    rclpy.init()
    node = DepthImageFetchServer(
        source_topic=args.source_topic,
        service_name=args.service_name,
        output_topic=args.output_topic,
        source_qos=args.source_qos,
    )
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
