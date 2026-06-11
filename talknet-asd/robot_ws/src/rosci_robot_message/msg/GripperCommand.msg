# GripperCommand.msg - 夹爪控制指令消息定义
#
# 定义夹爪的运行状态和位置目标。
# 用于手臂控制系统中协调夹爪运动。
# Author: The Robotic Science Team
# Date: 2025-10-25
# Version: 1.0

# State request constants for arm control state machine
uint8 REQUEST_STATE_NONE=0      # No used
uint8 REQUEST_STATE_INIT=1      # Initialize arm system
uint8 REQUEST_STATE_SETUP=2     # Setup/configuration state
uint8 REQUEST_STATE_STOP=3      # Stop arm operation
uint8 REQUEST_STATE_OPERATION=4 # Normal operation state
uint8 REQUEST_STATE_BOOT=5       # Boot/startup state

uint64 time_stamp
uint32 permission

# Request state request telescopic gripper
uint8 request_state            # [0-5] Target operational mode from state definitions

# Desired position
float32[] position               # Target extension position in meters (m)
# Max torque coefficient when run as postion mode
float32[] torque                 # Troqeu coefficient, 0~1
