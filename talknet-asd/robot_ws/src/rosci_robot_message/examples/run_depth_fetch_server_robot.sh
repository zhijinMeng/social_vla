#!/usr/bin/env bash
set -eo pipefail

# ROS setup.bash may read optional vars that are unset; keep nounset off here.
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
export CYCLONEDDS_URI=file://$HOME/Downloads/delay_test/examples/cyclonedds_unicast.xml

echo "[robot] CYCLONEDDS_URI=$CYCLONEDDS_URI"
echo "[robot] checking source topic..."
ros2 topic hz /camera/head/depth/image_raw &
HZ_PID=$!
sleep 3
kill "$HZ_PID" >/dev/null 2>&1 || true

exec python3 "$HOME/Downloads/delay_test/examples/depth_image_fetch_server.py" \
  --source-topic /camera/head/depth/image_raw \
  --service-name /depth_fetch/fetch \
  --output-topic /depth_fetch/image
