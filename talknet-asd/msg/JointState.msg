# JointState.msg - ROS 2 Message Definition for Individual Joint State Information
#
# This IDL file defines the message structure used for reporting the current state
# of individual joints in the robotic arm system. It includes joint state, position,
# torque measurements, and fault codes for monitoring joint health and performance.
#
# Author: The Robotic Science Team
# Date: 2025-10-10
# Version: 1.0

# Joint state indicator (0 = normal, 1 = warning, 2 = error, etc.)
uint8   state

# Current joint position in radians
float32 position

# Measured joint torque in Newton-meters (Nm)
float32 torque

# Fault code indicating specific joint error conditions (0 = no fault)
uint32  fault_code
