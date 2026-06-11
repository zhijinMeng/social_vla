# JointCommand.msg - ROS 2 Message Definition for Individual Joint Control Commands
#
# This IDL file defines the message structure used for sending control commands
# to individual joints in the robotic arm system. Each joint command specifies
# desired position and motion profile parameters.
#
# Author: The Robotic Science Team
# Date: 2025-10-10
# Version: 1.0

# Desired position for the joint in radians
float32 position_desired

# Velocity gain factor (0.0 to 1.0) for scaling maximum joint velocity
float32 velocity_gain

# Acceleration gain factor (0.0 to 1.0) for scaling maximum joint acceleration
float32 acceleration_gain
