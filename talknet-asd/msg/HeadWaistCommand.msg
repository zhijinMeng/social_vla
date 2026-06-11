# HeadWaistControl.msg - ROS 2 Message Definition for Head and Waist Control
#
# This message defines the control interface for head and waist joints
# in the ROSci robot system. It includes state requests and joint command arrays
# for coordinated control of upper body movements.
#
# Author: The Robotic Science Team
# Date: 2025-11-13
# Version: 1.0

# State request constants for arm control state machine
uint8 REQUEST_STATE_NONE=0      # No used
uint8 REQUEST_STATE_INIT=1      # Initialize arm system
uint8 REQUEST_STATE_SETUP=2     # Setup/configuration state
uint8 REQUEST_STATE_STOP=3      # Stop arm operation
uint8 REQUEST_STATE_OPERATION=4 # Normal operation state
uint8 REQUEST_STATE_BOOT=5       # Boot/startup state

std_msgs/Header header

# head enable flags and waist enable flags
uint8[2]            enable_head       #0: disable, 1: enable
uint8[2]            enable_waist

# Request state request for the head and waist control
uint8[2]           head_request_state        # commonly used：2、3  
uint8[2]           waist_request_state

# Joint commands for 2-DOF head (array of 2 JointCommand messages)
JointCommand[2] head_joint  

# Joint commands for 2-DOF waist arm (array of 2 JointCommand messages)
JointCommand[2] waist_joint
