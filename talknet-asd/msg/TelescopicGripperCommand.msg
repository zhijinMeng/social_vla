# TelescopicGripperCommand.msg - Control commands for telescopic gripper mechanism
#
# This message defines operational states and position targets for robotic telescopic grippers.
# Used in arm control systems to coordinate extension/retraction movements.
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
float32 position               # Target extension position in meters (m) 
