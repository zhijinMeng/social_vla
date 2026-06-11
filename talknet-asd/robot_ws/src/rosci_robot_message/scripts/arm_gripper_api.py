#!/usr/bin/env python3
"""
arm_gripper_api.py

两个可复用类:
1) ArmClient: 对应 RDK 3.3.2 手臂接口
2) GripperClient: 对应 RDK 3.3.5 末端夹爪接口

说明:
- 基于 ROS2 topic 封装，支持发送命令和读取最新状态。
- 适合在业务代码中直接 import 调用。
"""

from copy import deepcopy
import threading
from typing import Callable, List, Optional

from rclpy.node import Node

from rosci_robot_message.msg import ArmCommand, ArmState, GripperCommand, GripperState


def _array2d_to_list(array2d_msg) -> List[List[float]]:
    return [list(row.array_1d) for row in array2d_msg.array_2d]


def _array3d_to_list(array3d_msg) -> List[List[List[float]]]:
    return [_array2d_to_list(layer) for layer in array3d_msg.array_3d]


class ArmClient:
    """对应 RDK 3.3.2 的手臂接口封装。"""

    def __init__(
        self,
        node: Node,
        command_topic: str = "/rosci_arm_command",
        state_topic: str = "/rosci_arm_state",
        on_state: Optional[Callable[[ArmState], None]] = None,
        on_command: Optional[Callable[[ArmCommand], None]] = None,
    ):
        self._node = node
        self._on_state = on_state
        self._on_command = on_command
        self._lock = threading.Lock()

        self._latest_state: Optional[ArmState] = None
        self._latest_command: Optional[ArmCommand] = None

        self._pub = node.create_publisher(ArmCommand, command_topic, 10)
        self._state_sub = node.create_subscription(ArmState, state_topic, self._on_state_msg, 10)
        self._cmd_sub = node.create_subscription(ArmCommand, command_topic, self._on_command_msg, 10)

    def _on_state_msg(self, msg: ArmState):
        with self._lock:
            self._latest_state = deepcopy(msg)
        if self._on_state is not None:
            self._on_state(msg)

    def _on_command_msg(self, msg: ArmCommand):
        with self._lock:
            self._latest_command = deepcopy(msg)
        if self._on_command is not None:
            self._on_command(msg)

    def _publish(self, msg: ArmCommand):
        self._latest_command = deepcopy(msg)
        self._pub.publish(msg)

    def _require_state(self) -> ArmState:
        with self._lock:
            if self._latest_state is None:
                raise RuntimeError("尚未收到 ArmState，请先等待状态消息")
            return deepcopy(self._latest_state)

    # ----- 3.3.2 状态接口 -----
    def GetJointNum(self) -> int:
        return int(self._require_state().joint_num)

    def GetLeftJointPosition(self) -> List[float]:
        return list(self._require_state().left_joint_position)

    def GetRightJointPosition(self) -> List[float]:
        return list(self._require_state().right_joint_position)

    def GetLeftJointVelocity(self) -> List[float]:
        return list(self._require_state().left_joint_velocity)

    def GetRightJointVelocity(self) -> List[float]:
        return list(self._require_state().right_joint_velocity)

    def GetLeftJointTorque(self) -> List[float]:
        return list(self._require_state().left_joint_torque)

    def GetRightJointTorque(self) -> List[float]:
        return list(self._require_state().right_joint_torque)

    def GetLeftToolForce(self) -> List[float]:
        return list(self._require_state().left_tool_force)

    def GetRightToolForce(self) -> List[float]:
        return list(self._require_state().right_tool_force)

    def GetLeftToolPosition(self) -> List[float]:
        return list(self._require_state().left_tool_position)

    def GetRightToolPosition(self) -> List[float]:
        return list(self._require_state().right_tool_position)

    def GetLeftJointCurrent(self) -> List[float]:
        return list(self._require_state().left_joint_current)

    def GetRightJointCurrent(self) -> List[float]:
        return list(self._require_state().right_joint_current)

    def GetLeftJointTemperature(self) -> List[float]:
        return list(self._require_state().left_joint_temperature)

    def GetRightJointTemperature(self) -> List[float]:
        return list(self._require_state().right_joint_temperature)

    def GetLeftJointPositionMin(self) -> List[float]:
        return list(self._require_state().left_joint_min_pos)

    def GetRightJointPositionMin(self) -> List[float]:
        return list(self._require_state().right_joint_min_pos)

    def GetLeftJointPositionMax(self) -> List[float]:
        return list(self._require_state().left_joint_max_pos)

    def GetRightJointPositionMax(self) -> List[float]:
        return list(self._require_state().right_joint_max_pos)

    # ----- 3.3.2 指令接口 -----
    def SetRequestState(self, request_state: int):
        msg = ArmCommand()
        msg.request_state = int(request_state)
        self._publish(msg)

    def SetWorkMode(self, work_mode: int):
        msg = ArmCommand()
        msg.request_state = ArmCommand.REQUEST_STATE_NONE
        msg.work_mode = int(work_mode)
        self._publish(msg)

    def SetLeftJointPosition(
        self,
        position: List[float],
        velocity_gain: float = 0.2,
        acceleration_gain: float = 0.2,
    ):
        msg = ArmCommand()
        msg.request_state = ArmCommand.REQUEST_STATE_NONE
        msg.left_joint_position = list(position)
        msg.velocity_gain = float(velocity_gain)
        msg.acceleration_gain = float(acceleration_gain)
        self._publish(msg)

    def SetRightJointPosition(
        self,
        position: List[float],
        velocity_gain: float = 0.2,
        acceleration_gain: float = 0.2,
    ):
        msg = ArmCommand()
        msg.request_state = ArmCommand.REQUEST_STATE_NONE
        msg.right_joint_position = list(position)
        msg.velocity_gain = float(velocity_gain)
        msg.acceleration_gain = float(acceleration_gain)
        self._publish(msg)


class GripperClient:
    """对应 RDK 3.3.5 的末端夹爪接口封装。"""

    def __init__(
        self,
        node: Node,
        command_topic: str = "/rosci_left_gripper_command",
        state_topic: str = "/rosci_left_gripper_state",
        on_state: Optional[Callable[[GripperState], None]] = None,
        on_command: Optional[Callable[[GripperCommand], None]] = None,
    ):
        self._node = node
        self._on_state = on_state
        self._on_command = on_command
        self._lock = threading.Lock()

        self._latest_state: Optional[GripperState] = None
        self._latest_command: Optional[GripperCommand] = None

        self._pub = node.create_publisher(GripperCommand, command_topic, 10)
        self._state_sub = node.create_subscription(
            GripperState, state_topic, self._on_state_msg, 10
        )
        self._cmd_sub = node.create_subscription(
            GripperCommand, command_topic, self._on_command_msg, 10
        )

    def _on_state_msg(self, msg: GripperState):
        with self._lock:
            self._latest_state = deepcopy(msg)
        if self._on_state is not None:
            self._on_state(msg)

    def _on_command_msg(self, msg: GripperCommand):
        with self._lock:
            self._latest_command = deepcopy(msg)
        if self._on_command is not None:
            self._on_command(msg)

    def _publish(self, msg: GripperCommand):
        self._latest_command = deepcopy(msg)
        self._pub.publish(msg)

    def _require_state(self) -> GripperState:
        with self._lock:
            if self._latest_state is None:
                raise RuntimeError("尚未收到 GripperState，请先等待状态消息")
            return deepcopy(self._latest_state)

    # ----- 3.3.5 状态接口 -----
    def GetJointNum(self) -> int:
        return int(self._require_state().joint_num)

    def GetPosition(self) -> List[float]:
        return list(self._require_state().position)

    def GetPositionMin(self) -> List[float]:
        return list(self._require_state().min_pos)

    def GetPositionMax(self) -> List[float]:
        return list(self._require_state().max_pos)

    def GetForceData(self) -> List[List[List[float]]]:
        return _array3d_to_list(self._require_state().force)

    def GetRawForceData(self) -> List[List[List[float]]]:
        return _array3d_to_list(self._require_state().raw_force_data)

    def GetTemperature(self) -> List[List[float]]:
        return _array2d_to_list(self._require_state().temperature)

    def GetRawTemperature(self) -> List[List[float]]:
        return _array2d_to_list(self._require_state().raw_temperature_data)

    # ----- 3.3.5 指令接口 -----
    def SetRequestState(self, request_state: int):
        msg = GripperCommand()
        msg.request_state = int(request_state)
        self._publish(msg)

    def SetPosition(self, position: List[float], torque: Optional[List[float]] = None):
        msg = GripperCommand()
        msg.request_state = GripperCommand.REQUEST_STATE_NONE
        msg.position = list(position)
        msg.torque = list(torque) if torque is not None else []
        self._publish(msg)
