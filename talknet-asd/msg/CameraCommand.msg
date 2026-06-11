# GripperCommand.msg - Control commands Camera mechanism
#
# Author: The Robotic Science Team
# Date: 2025-10-25
# Version: 1.0

# State machine definitions (0-5 reserved for system control)
uint8 REQUEST_STATE_INIT=0      # Initialize communication and reset parameters
uint8 REQUEST_STATE_SETUP=1     # Configure motion profiles and safety limits
uint8 REQUEST_STATE_STOP=2      # Emergency stop (immediate motion halt)
uint8 REQUEST_STATE_OPERATION=3 # Normal closed-loop position control mode
uint8 REQUEST_STATE_BOOT=4      # Boot/startup state


uint64 time_stamp
uint32 permission

# Request state request telescopic gripper
uint8 request_state            # [0-5] Target operational mode from state definitions

uint8 reserved_1
uint8 reserved_2
uint8 reserved_3

bool enable_rgb_frame True
bool enable_depth_frame True
bool enable_infrared_frame true



