# ArmCommand.msg - ROS 2 Message Definition for Robotic Arm Control Commands
#
# This IDL file defines the message structure used for sending control commands
# to robotic arms in the robot system. It includes state requests and
# joint-level commands for both right and left arms.
#
# Author: The Robotic Science Team
# Date: 2025-10-10
# Version: 1.0

# State request constants for arm control state machine
uint8 REQUEST_STATE_NONE=0      # No used
uint8 REQUEST_STATE_INIT=1      # Initialize arm system
uint8 REQUEST_STATE_SETUP=2     # Setup/configuration state
uint8 REQUEST_STATE_STOP=3      # Stop arm operation
uint8 REQUEST_STATE_OPERATION=4 # Normal operation state
uint8 REQUEST_STATE_BOOT=5       # Boot/startup state

uint8 WORK_MODE_STOP=0
uint8 WORK_MODE_JOINT_POSITION=1
uint8 WORK_MODE_JOINT_POSITION_PD=2
# uint8 WORK_MODE_TOOL_POSITION=3
uint8 WORK_MODE_FREE_DRAG=4

uint64 time_stamp
uint32 permission

# Request state request for the arm system
uint8   request_state

uint8   work_mode

# Current joint position in radians
float32[] left_joint_position
float32[] right_joint_position

float32[] left_joint_torque
float32[] right_joint_torque

# Velocity gain factor (0.0 to 1.0) for scaling maximum joint velocity
float32 velocity_gain

# Acceleration gain factor (0.0 to 1.0) for scaling maximum joint acceleration
float32 acceleration_gain

float32[6] left_tool_position
float32[6] right_tool_position
