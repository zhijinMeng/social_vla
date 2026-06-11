# MotionPlatformCommand.msg - ROS 2 Message Definition for Motion Platform Control
#
# This IDL file defines the command interface for robotic motion platforms,
# including state transitions and control parameters for linear and rotary motion.
# It serves as a communication protocol between the motion planning system and
# the hardware control layer of the robotic platform.
#
# Author: The Robotic Science Team
# Date: 2025-10-25
# Version: 1.0

# Platform State Constants
# These constants define the valid operational states that the motion platform
# State request constants for arm control state machine
uint8 REQUEST_STATE_NONE=0      # No used
uint8 REQUEST_STATE_INIT=1      # Initialize arm system
uint8 REQUEST_STATE_SETUP=2     # Setup/configuration state
uint8 REQUEST_STATE_STOP=3      # Stop arm operation
uint8 REQUEST_STATE_OPERATION=4 # Normal operation state
uint8 REQUEST_STATE_BOOT=5       # Boot/startup state

uint64 time_stamp
uint32 permission

# Target operational state
# Specifies which operational state the platform should transition into
uint8 request_state

# Emergency stop command
# When set to true, this overrides all other motion commands and immediately stops the platform
# Priority: Highest - overrides all other commands when asserted
# uint8 emerge_stop

# Direction control commands
# Binary flags to enable movement in specific directions
# Note: Conflicting directions (e.g., move_forward and move_backward simultaneously)
#       will be ignored or handled according to the platform's conflict resolution policy

uint32 position_cmd_idx
float32[6] position
uint8 world_coordinate

uint32 velocity_cmd_idx
float32[6] velocity
uint32 duration

uint32 navigation_cmd_idx
string station_name
uint8 cancel_navigation

# Velocity limiting parameters
# Maximum allowable linear velocity coefficient, 0~1
float32 max_velocity_coeff
