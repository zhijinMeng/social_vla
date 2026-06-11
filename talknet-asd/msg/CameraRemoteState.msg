# CameraState.msg - ROS 2 Message Definition for Camera Gripper Status
#
# This IDL file defines the message structure used for reporting the current state
# of Camera in the ROSci robot system. It includes operational state,
#
# Author: The Robotic Science Team
# Date: 2025-11-11
# Version: 1.0
# Version: 1.1
#   1. Add motion information
#

##########################################################################
# Macro
##########################################################################
uint8 STATE_INIT=0       # Initialization state: Establishing communication
uint8 STATE_SETUP=1      # Configuration state: Setting motion parameters
uint8 STATE_STOP=2       # Stopped state: Motion halted, brakes engaged
uint8 STATE_OPERATION=3  # Operational state: Actively controlling position
uint8 STATE_BOOT=4       # Boot state: Performing hardware self-tests
uint8 STATE_ERROR=5      # Error state: Critical fault detected

##########################################################################
# Base information
##########################################################################
uint64 time_stamp       # Current operational state (use STATE_* constants)
uint32 permission
uint32 state            
uint32 fault_code       # Fault code indicating specific error conditions (0 = no fault)

##########################################################################
# Motion
##########################################################################
uint64 motion_timestamp 
float32[3] acce         # motion data, acc_x, acc_y, acc_z
float32[3] gyro         # motion data, gyro, gyro_x, gyro_y, gyro_z

##########################################################################
# Image
##########################################################################
RemoteImage rgb_frame         # RGB image frame
RemoteImage depth_frame       # Depth image frame
RemoteImage infrared_frame_1  # Infrared frame 1
RemoteImage infrared_frame_2  # Infrared frame 2

