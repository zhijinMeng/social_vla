"""
robot_bringup.launch.py - 一键启动所有 ROSci 机器人 ROS2 Wrapper 节点。

启动手臂、夹爪、腰颈和推理 RPC Wrapper 节点，
支持通过 launch 参数进行配置。

用法:
    ros2 launch rosci_robot_message robot_bringup.launch.py
    ros2 launch rosci_robot_message robot_bringup.launch.py \
        config_file:=/path/to/config.json \
        server_address:=192.168.1.100:50051
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ----- Launch 参数 -----
    config_file_arg = DeclareLaunchArgument(
        'config_file', default_value='',
        description='RDK 配置 JSON 文件路径')

    arm_name_arg = DeclareLaunchArgument(
        'arm_name', default_value='arm',
        description='RDK 手臂实例名称')

    gripper_name_arg = DeclareLaunchArgument(
        'gripper_name', default_value='gripper',
        description='RDK 夹爪实例名称')

    head_waist_name_arg = DeclareLaunchArgument(
        'head_waist_name', default_value='head_waist',
        description='RDK 腰颈实例名称')

    server_address_arg = DeclareLaunchArgument(
        'server_address', default_value='192.168.1.100:50051',
        description='gRPC 推理服务器地址 (ip:port)')

    # ----- 节点 -----
    arm_node = Node(
        package='rosci_robot_message',
        executable='arm_wrapper_node',
        name='arm_wrapper_node',
        output='screen',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'arm_name': LaunchConfiguration('arm_name'),
            'state_rate': 50.0,
        }],
    )

    gripper_node = Node(
        package='rosci_robot_message',
        executable='gripper_wrapper_node',
        name='gripper_wrapper_node',
        output='screen',
        parameters=[{
            'gripper_name': LaunchConfiguration('gripper_name'),
            'state_rate': 20.0,
        }],
    )

    head_waist_node = Node(
        package='rosci_robot_message',
        executable='head_waist_wrapper_node',
        name='head_waist_wrapper_node',
        output='screen',
        parameters=[{
            'head_waist_name': LaunchConfiguration('head_waist_name'),
            'state_rate': 20.0,
        }],
    )

    inference_node = Node(
        package='rosci_robot_message',
        executable='inference_rpc_node',
        name='inference_rpc_node',
        output='screen',
        parameters=[{
            'server_address': LaunchConfiguration('server_address'),
            'model_name': 'default',
            'jpeg_quality': 80,
        }],
    )

    return LaunchDescription([
        # 参数
        config_file_arg,
        arm_name_arg,
        gripper_name_arg,
        head_waist_name_arg,
        server_address_arg,
        # 节点
        arm_node,
        gripper_node,
        head_waist_node,
        inference_node,
    ])
