# LiftPlatformState.msg - ROS 2 Message Definition for Motion Platform Status
#
# This IDL file defines the feedback interface for robotic motion platforms,
# providing real-time status monitoring and diagnostic information.
#
# Author: The Robotic Science Team
# Date: 2025-10-25
# Version: 1.0

uint32 STATE_INIT=0       # Initialization state: Communication established
uint32 STATE_SETUP=1      # Configuration state: Parameters loaded
uint32 STATE_STOP=2       # Stopped state: Motion halted, brakes engaged
uint32 STATE_OPERATION=3 # Operational state: Active position control
uint32 STATE_BOOT=4       # Boot state: Performing self-diagnostics
uint32 STATE_ERROR=5      # Error state: Critical fault detected

uint64 time_stamp
uint32 permission

# Current operational state
uint32 state

# Fault code indicating specific error conditions (0 = no fault)
uint32 fault_code

float32[6] position
uint32 position_cmd_idx

float32[6] velocity
uint32 velocity_cmd_idx

uint32 is_reach_target
uint32 navigation_cmd_idx

uint32 is_blocked
uint32 block_reason
float32[3] block_position

uint32 is_slow
uint32 slow_reason
float32[3] slow_position

float32 battery_level
float32 battery_temperature
float32 battery_voltage
float32 battery_current
bool battery_charging
