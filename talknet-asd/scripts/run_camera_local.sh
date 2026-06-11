#!/usr/bin/env bash
# Camera + mic realtime ASD — no ROS, no robot head tracking.
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [[ -f "${REPO_DIR}/.venv-talknet/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.venv-talknet/bin/activate"
elif [[ -f "${HOME}/Research/MERCI_Plus/analysis/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/Research/MERCI_Plus/analysis/.venv/bin/activate"
else
  echo "[error] 未找到 Python 环境。请先运行: bash scripts/setup_venv.sh"
  exit 1
fi

bash "${REPO_DIR}/scripts/setup_weights.sh"

CAMERA_ID="${CAMERA_ID:-0}"
MIC_DEVICE="${MIC_DEVICE:-4}"

exec python online_demoTalkNet.py \
  --camera --cameraId "${CAMERA_ID}" --showWindow --flip \
  --micDevice "${MIC_DEVICE}" \
  --windowSec 0.5 --inferStrideSec 0.1 --scoreThres -0.2 \
  --onlyTopSpeaker \
  --facePositionLog --facePositionMode top \
  "$@"
