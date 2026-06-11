#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

set +u
source /opt/ros/humble/setup.bash
if [ -f "${REPO_DIR}/robot_ws/install/setup.bash" ]; then
  source "${REPO_DIR}/robot_ws/install/setup.bash"
else
  echo "[WARN] ${REPO_DIR}/robot_ws/install/setup.bash not found."
  echo "[WARN] Run: bash scripts/setup_local_robot_ws.sh /real/path/to/rosci_robot_message"
fi
set -u

source .venv-talknet/bin/activate

exec python online_demoTalkNet.py \
  --camera --cameraId 0 --showWindow \
  --windowSec 0.5 --inferStrideSec 0.1 --scoreThres -0.2\
  --enableRobotHeadTrack --robotActiveHoldSec 2.0 \
  --robotCmdTopic /rosci_head_waist_command \
  --robotMsgModule rosci_robot_message.msg \
  --robotCmdMsg HeadWaistCommand \
  "$@"
