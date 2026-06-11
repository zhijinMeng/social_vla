#!/bin/bash
# 新增：任一命令报错即退出，避免后续步骤在异常状态下继续执行
set -e

# 加载 ROS2 环境：
# 1) 优先使用当前环境变量 ROS_DISTRO
# 2) 否则回退到 /opt/ros 下检测到的第一个发行版
# 新增：避免写死单一发行版（如 rolling），使 humble/其他版本也可构建
if [ -n "$ROS_DISTRO" ] && [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
  source "/opt/ros/$ROS_DISTRO/setup.bash"
else
  first_distro=$(ls /opt/ros 2>/dev/null | head -n 1 || true)
  if [ -z "$first_distro" ] || [ ! -f "/opt/ros/$first_distro/setup.bash" ]; then
    echo "No ROS2 setup found under /opt/ros"
    exit 1
  fi
  source "/opt/ros/$first_distro/setup.bash"
fi

colcon build
