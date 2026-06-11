#!/usr/bin/env python3
"""
image_service_client.py

在远端电脑运行：通过 ROS2 Service 按需从机器人获取相机图像。

用法:
  # 单次获取
  python3 examples/image_service_client.py /camera/head/depth/forward

  # 循环获取（每2秒一次）
  python3 examples/image_service_client.py /camera/head/depth/forward --loop 2.0

  # 获取并保存为文件
  python3 examples/image_service_client.py /camera/head/depth/forward --save

  # 列出服务端支持的话题（发送空请求）
  python3 examples/image_service_client.py --list
"""

import argparse
import time
import sys
import os

import rclpy
from rclpy.node import Node
from rosci_robot_message.srv import GetImage


class ImageServiceClient(Node):

    def __init__(self):
        super().__init__("image_service_client")
        self._cli = self.create_client(GetImage, "get_image")

    def wait_for_service(self, timeout_sec=10.0):
        self.get_logger().info("等待 /get_image 服务...")
        if not self._cli.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f"服务 /get_image 在 {timeout_sec}s 内未就绪")
            return False
        self.get_logger().info("服务已连接")
        return True

    def get_image(self, topic_name: str):
        req = GetImage.Request()
        req.topic_name = topic_name

        future = self._cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        if future.result() is None:
            self.get_logger().error("服务调用超时或失败")
            return None
        return future.result()


def main():
    parser = argparse.ArgumentParser(
        description="ROS2 图像服务客户端 - 按需从机器人获取图像",
    )
    parser.add_argument(
        "topic_name", nargs="?", default="",
        help="要获取的话题名，如 /camera/head/depth/forward",
    )
    parser.add_argument(
        "--loop", type=float, default=0,
        help="循环获取间隔（秒），0 表示单次获取",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="将获取的图像保存为原始数据文件",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./captured_images",
        help="保存目录 (默认 ./captured_images)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="发送空话题名查询服务端支持的话题列表",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0,
        help="等待服务超时秒数 (默认 10s)",
    )
    args = parser.parse_args()

    if not args.list and not args.topic_name:
        parser.error("请指定 topic_name 或使用 --list")

    rclpy.init()
    node = ImageServiceClient()

    if not node.wait_for_service(timeout_sec=args.timeout):
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    topic_name = args.topic_name if not args.list else "__list__"

    try:
        count = 0
        while True:
            t0 = time.time()
            resp = node.get_image(topic_name)
            elapsed = (time.time() - t0) * 1000

            if resp is None:
                break

            count += 1

            if resp.success:
                img = resp.image
                node.get_logger().info(
                    f"[{count}] {resp.message} | 传输耗时={elapsed:.0f}ms"
                )

                if args.save:
                    os.makedirs(args.save_dir, exist_ok=True)
                    safe_name = topic_name.replace("/", "_").strip("_")
                    fname = os.path.join(
                        args.save_dir,
                        f"{safe_name}_{count:04d}_{img.width}x{img.height}.raw"
                    )
                    with open(fname, "wb") as f:
                        f.write(bytes(img.data))
                    node.get_logger().info(f"  已保存: {fname}")
            else:
                node.get_logger().warn(f"[{count}] 失败: {resp.message}")

            if args.loop <= 0:
                break
            time.sleep(args.loop)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
