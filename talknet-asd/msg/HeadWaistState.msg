

# State constants for arm system state machine
uint8 STATE_INIT=0       # Initialization state
uint8 STATE_SETUP=1      # Setup/configuration state
uint8 STATE_STOP=2       # Stopped state
uint8 STATE_OPERATION=3 # Normal operation state
uint8 STATE_BOOT=4       # Boot/startup state
uint8 STATE_ERROR=5      # Error/fault state

# timestamp of the state message
std_msgs/Header header

#0:pitch motor 1:yaw motor
# Current system state of the head waist system
#uint8[2]           head_state      
#uint8[2]           waist_state     

# Fault code indicating specific error conditions (0 = no fault)
#uint32[2]          head_fault_code     
#uint32[2]          waist_fault_code    

# Check the head or waist is reach target (True if reach target)
uint8[2] is_head_reach_target          
uint8[2] is_waist_reach_target         

# Joint state information for 2-DOF head (array of 2 JointState messages)
JointState[2]   head_joint            

# Joint state information for 2-DOF waist (array of 2 JointState messages)
JointState[2]   waist_joint