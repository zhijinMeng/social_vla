# Image.msg - ROS 2 Message Definition for Image
#
# This IDL file defines the message structure used for reporting the current image data
#
# Author: The Robotic Science Team
# Date: 2025-12-15
# Version: 1.0
#

##########################################################################
# Macro
##########################################################################


##########################################################################
# Intrinsics
##########################################################################
float32 fx
float32 fy
float32 cx
float32 cy
uint32 distortion_model
float32[5] distortion_coeff

##########################################################################
# Image data
##########################################################################
uint32 time_stamp_h     # Image time stamp low bytes
uint32 time_stamp_l     # Image time stamp hight bytes
uint32 height           # Image height, that is, number of rows
uint32 width            # Image width, that is, number of columns
uint32 is_bigendian     # Is this data bigendian?
uint32 step             # Full row length in bytes
# Encoding of pixels -- channel meaning, ordering, 
# 1, CV_8UC1 , "mono8"
# 2, CV_8UC3 , "bgr8"
# 3, CV_16UC1, "mono16"
# 4, CV_32FC1, "32FC1"
# *, CV_8UC3, "bgr8"
uint32 encoding 

# 1920 x 1280 x 3 = 7,372,800, Will segmentation fault in AGX platform
# 1280 X 720 X 3 = 2,764,800
# 640 X 480 X 3 = 921,600
uint8[3000000] data          # actual matrix data, size is (step * rows)
