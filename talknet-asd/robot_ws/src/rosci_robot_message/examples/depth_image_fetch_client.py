#!/usr/bin/env python3
"""
depth_image_fetch_client.py

电脑侧按需取图客户端:
- 调用 Trigger 服务发起“取一帧”请求
- 订阅响应话题等待这一帧
- 打印请求结果与等待耗时

用法:
  # 单次请求
  python3 examples/depth_image_fetch_client.py

  # 连续请求（例如 2Hz）
  python3 examples/depth_image_fetch_client.py --rate 2.0
"""

import argparse
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


class DepthImageFetchClient(Node):
    def __init__(self, service_name: str, output_topic: str, timeout_s: float):
        super().__init__("depth_image_fetch_client")

        self._timeout_s = timeout_s
        self._last_image: Optional[Image] = None
        self._new_image = False

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image, output_topic, self._on_image, sub_qos)
        self._cli = self.create_client(Trigger, service_name)

        self.get_logger().info(f"服务: {service_name}")
        self.get_logger().info(f"接收话题: {output_topic}")

    def _on_image(self, msg: Image):
        self._last_image = msg
        self._new_image = True

    def fetch_once(self) -> bool:
        if not self._cli.wait_for_service(timeout_sec=self._timeout_s):
            self.get_logger().error("服务不可用，超时。")
            return False

        self._new_image = False
        req = Trigger.Request()

        t0 = time.time()
        future = self._cli.call_async(req)

        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.02)
            if time.time() - t0 > self._timeout_s:
                self.get_logger().error("服务调用超时。")
                return False

        result = future.result()
        if result is None:
            self.get_logger().error("服务调用失败。")
            return False
        if not result.success:
            self.get_logger().warn(f"服务返回失败: {result.message}")
            return False

        t_srv = (time.time() - t0) * 1000.0

        t1 = time.time()
        while rclpy.ok() and not self._new_image:
            rclpy.spin_once(self, timeout_sec=0.02)
            if time.time() - t1 > self._timeout_s:
                self.get_logger().error("等待图像回包超时。")
                return False

        img = self._last_image
        if img is None:
            self.get_logger().error("未拿到图像。")
            return False

        now = self.get_clock().now().nanoseconds * 1e-9
        stamp = img.header.stamp.sec + img.header.stamp.nanosec * 1e-9
        age_ms = (now - stamp) * 1000.0 if stamp > 0 else -1.0

        self.get_logger().info(
            f"成功: srv={t_srv:.1f}ms, "
            f"img={img.width}x{img.height}, bytes={len(img.data)}, "
            f"encoding={img.encoding}, age={age_ms:.1f}ms"
        )
        return True


def main():
    parser = argparse.ArgumentParser(description="电脑侧按需深度图客户端")
    parser.add_argument("--service-name", default="/depth_fetch/fetch", help="请求服务名")
    parser.add_argument("--output-topic", default="/depth_fetch/image", help="返回图像话题")
    parser.add_argument("--timeout", type=float, default=3.0, help="每次请求超时(秒)")
    parser.add_argument("--rate", type=float, default=0.0, help="连续请求频率Hz，<=0 表示只请求一次")
    args = parser.parse_args()

    rclpy.init()
    node = DepthImageFetchClient(args.service_name, args.output_topic, args.timeout)

    try:
        if args.rate and args.rate > 0.0:
            interval = 1.0 / args.rate
            node.get_logger().info(f"连续模式: {args.rate:.2f}Hz")
            while rclpy.ok():
                started = time.time()
                node.fetch_once()
                elapsed = time.time() - started
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        else:
            node.fetch_once()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()

