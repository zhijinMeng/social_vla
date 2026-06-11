#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os
import sys
# 固定把仓库根目录加入导入路径，支持直接运行: python3 examples/example2.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.arm_gripper_api import ArmClient, GripperClient
from rosci_robot_message.msg import ArmCommand, GripperCommand


class MyApp(Node):
    def __init__(self):
        super().__init__("my_app")
        self._tick_count = 0

        self.arm = ArmClient(
            self,
            command_topic="/rosci_arm_command",
            state_topic="/rosci_arm_state",
        )
        self.gripper = GripperClient(
            self,
            command_topic="/rosci_left_gripper_command",
            state_topic="/rosci_left_gripper_state",
        )

        self.create_timer(1.0, self.tick)

    def tick(self):
        self._tick_count += 1

        arm_ready = True
        gripper_ready = True

        try:
            left_pos = self.arm.GetLeftJointPosition()
        except RuntimeError as e:
            arm_ready = False
            if self._tick_count % 5 == 0:
                self.get_logger().warn(f"arm状态未就绪: {e}")

        try:
            g_pos = self.gripper.GetPosition()
        except RuntimeError as e:
            gripper_ready = False
            if self._tick_count % 5 == 0:
                self.get_logger().warn(f"gripper状态未就绪: {e}")

        if arm_ready and gripper_ready:
            self.get_logger().info(f"left_pos={left_pos}, gripper_pos={g_pos}")

            # 发控制命令（示例）
            self.arm.SetWorkMode(ArmCommand.WORK_MODE_JOINT_POSITION)
            self.arm.SetLeftJointPosition([0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0], 0.2, 0.2)

            self.gripper.SetRequestState(GripperCommand.REQUEST_STATE_OPERATION)
            self.gripper.SetPosition([0.2], [0.5])


def main():
    rclpy.init()
    node = MyApp()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
