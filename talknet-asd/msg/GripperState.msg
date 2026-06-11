# GripperState.msg - ROS 2 Message Definition for Telescopic Gripper Status
#
# This IDL file defines the message structure used for reporting the current state
# of telescopic grippers in the ROSci robot system. It includes operational state,
# position feedback, and target achievement status.
#
# Author: The Robotic Science Team
# Date: 2025-10-10
# Version: 1.0


uint8 STATE_INIT=0       # Initialization state: Establishing communication
uint8 STATE_SETUP=1      # Configuration state: Setting motion parameters
uint8 STATE_STOP=2       # Stopped state: Motion halted, brakes engaged
uint8 STATE_OPERATION=3  # Operational state: Actively controlling position
uint8 STATE_BOOT=4       # Boot state: Performing hardware self-tests
uint8 STATE_ERROR=5      # Error state: Critical fault detected

uint64 time_stamp
uint32 permission

# Current operational state (use STATE_* constants)
uint8 state

# Number of joints in the gripper
uint8 joint_num

# Fault code indicating specific error conditions (0 = no fault)
uint32 fault_code

# Current extension position in meters (m)
float32[] position
float32[] min_pos
float32[] max_pos

# 各关节力矩反馈
float32[] torque

# Target achievement status true = position reached
bool is_reach_target

Array3D force
Array2D temperature

Array3D raw_force_data
Array2D raw_temperature_data
