# ArmState.msg - ROS 2 Message Definition for Robotic Arm State Information
#
# This IDL file defines the message structure used for reporting the current state
# of robotic arms in the ROSci robot system. It includes system state, fault codes,
# joint states, and tool force/torque measurements for both arms.
#
# Author: The Robotic Science Team
# Date: 2025-10-10
# Version: 1.0
# Version: 1.1 Add reach target flag

# State constants for arm system state machine
uint8 STATE_INIT=0       # Initialization state
uint8 STATE_SETUP=1      # Setup/configuration state
uint8 STATE_STOP=2       # Stopped state
uint8 STATE_OPERATION=3 # Normal operation state
uint8 STATE_BOOT=4       # Boot/startup state
uint8 STATE_ERROR=5      # Error/fault state

uint64 time_stamp
uint32 permission

# Current system state of the arm
uint8           state

# Fault code indicating specific error conditions (0 = no fault)
uint32          fault_code

# Check the Arm is reach target (True if reach target)
bool is_reach_target

uint8 joint_num

float32[] left_joint_position
float32[] left_joint_min_pos
float32[] left_joint_max_pos

float32[] right_joint_position
float32[] right_joint_min_pos
float32[] right_joint_max_pos

float32[] left_joint_velocity
float32[] right_joint_velocity

float32[] left_joint_torque
float32[] right_joint_torque

float32[] left_joint_temperature
float32[] right_joint_temperature

float32[] left_joint_current
float32[] right_joint_current

float32[6] left_tool_position
float32[6] right_tool_position

# Tool force/torque measurements for right arm (6-element array: Fx, Fy, Fz, Tx, Ty, Tz)
float32[6] right_tool_force

# Tool force/torque measurements for left arm (6-element array: Fx, Fy, Fz, Tx, Ty, Tz)
float32[6] left_tool_force
