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

echo "[pc] CYCLONEDDS_URI=$CYCLONEDDS_URI"
echo "[pc] waiting for /depth_fetch/fetch service..."

for i in $(seq 1 10); do
  if ros2 service list | grep -q "/depth_fetch/fetch"; then
    echo "[pc] service discovered."
    break
  fi
  sleep 1
done

exec python3 "$HOME/Downloads/delay_test/examples/depth_image_fetch_client.py" \
  --service-name /depth_fetch/fetch \
  --output-topic /depth_fetch/image
