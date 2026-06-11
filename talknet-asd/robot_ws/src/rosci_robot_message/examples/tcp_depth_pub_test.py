#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

class P(Node):
    def __init__(self):
        super().__init__('tcp_depth_pub')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub = self.create_publisher(Image, '/tcp_depth_test', qos)
        self.payload = bytes(1280 * 720 * 2)
        self.create_timer(0.25, self.tick)

    def tick(self):
        m = Image()
        m.header.stamp = self.get_clock().now().to_msg()
        m.height, m.width = 720, 1280
        m.encoding = 'mono16'
        m.is_bigendian = 0
        m.step = 1280 * 2
        m.data = self.payload
        self.pub.publish(m)

def main():
    rclpy.init()
    n = P()
    rclpy.spin(n)

if __name__ == '__main__':
    main()
