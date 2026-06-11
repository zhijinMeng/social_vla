#!/usr/bin/env python3
"""
arm_gripper_api_demo_node.py

ArmClient / GripperClient 的演示用例节点。
默认不发送控制，仅订阅并打印 command/state 关键信息。
"""

import rclpy
from rclpy.node import Node
import os
import sys

from rosci_robot_message.msg import ArmCommand, ArmState, GripperCommand, GripperState
# 固定把仓库根目录加入导入路径，支持直接运行: python3 examples/example1.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.arm_gripper_api import ArmClient, GripperClient


class ArmGripperApiDemoNode(Node):
    def __init__(self):
        super().__init__("arm_gripper_api_demo")
        self._arm_cmd_count = 0
        self._arm_state_count = 0
        self._gripper_cmd_count = 0
        self._gripper_state_count = 0

        self.arm = ArmClient(
            self,
            on_state=self._on_arm_state,
            on_command=self._on_arm_command,
        )
        self.gripper = GripperClient(
            self,
            on_state=self._on_gripper_state,
            on_command=self._on_gripper_command,
        )

        self.get_logger().info("ArmClient / GripperClient 已初始化，可在业务代码中直接调用。")
        self.get_logger().info("日志已开启：收到 Arm/Gripper 的 command/state 会输出关键信息。")

    def _on_arm_command(self, msg: ArmCommand):
        self._arm_cmd_count += 1
        self.get_logger().info(
            f"[ArmCommand#{self._arm_cmd_count}] request_state={msg.request_state}, "
            f"work_mode={msg.work_mode}, left_joint_len={len(msg.left_joint_position)}, "
            f"right_joint_len={len(msg.right_joint_position)}"
        )

    def _on_arm_state(self, msg: ArmState):
        self._arm_state_count += 1
        self.get_logger().info(
            f"[ArmState#{self._arm_state_count}] state={msg.state}, "
            f"fault_code={msg.fault_code}, is_reach_target={msg.is_reach_target}, "
            f"joint_num={msg.joint_num}"
        )

    def _on_gripper_command(self, msg: GripperCommand):
        self._gripper_cmd_count += 1
        self.get_logger().info(
            f"[GripperCommand#{self._gripper_cmd_count}] request_state={msg.request_state}, "
            f"position_len={len(msg.position)}, torque_len={len(msg.torque)}"
        )

    def _on_gripper_state(self, msg: GripperState):
        self._gripper_state_count += 1
        self.get_logger().info(
            f"[GripperState#{self._gripper_state_count}] state={msg.state}, "
            f"fault_code={msg.fault_code}, is_reach_target={msg.is_reach_target}, "
            f"joint_num={msg.joint_num}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ArmGripperApiDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
