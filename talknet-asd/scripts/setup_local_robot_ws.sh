#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS_DIR="${REPO_DIR}/robot_ws"
SRC_DIR="${WS_DIR}/src"
PKG_NAME="rosci_robot_message"

mkdir -p "${SRC_DIR}"

echo "[INFO] repo: ${REPO_DIR}"
echo "[INFO] ws:   ${WS_DIR}"

if [ -d "${SRC_DIR}/${PKG_NAME}" ]; then
  echo "[INFO] package already exists: ${SRC_DIR}/${PKG_NAME}"
else
  if [ $# -lt 1 ]; then
    echo "[ERROR] Missing package source path."
    echo "Usage: bash scripts/setup_local_robot_ws.sh /real/path/to/${PKG_NAME}"
    exit 1
  fi
  PKG_SRC="$1"
  if [[ "${PKG_SRC}" == "/abs/path/to/rosci_robot_message" ]]; then
    echo "[ERROR] You used the placeholder path. Please replace it with a real path."
    exit 1
  fi
  if [ ! -d "${PKG_SRC}" ]; then
    echo "[ERROR] Package path not found: ${PKG_SRC}"
    echo "[HINT] Try: find ~ -type d -name roscI_robot_message 2>/dev/null | head"
    exit 1
  fi
  echo "[INFO] copying package: ${PKG_SRC} -> ${SRC_DIR}/${PKG_NAME}"
  rm -rf "${SRC_DIR:?}/${PKG_NAME}"
  cp -a "${PKG_SRC}" "${SRC_DIR}/${PKG_NAME}"
fi

set +u
source /opt/ros/humble/setup.bash
set -u

cd "${WS_DIR}"
colcon build --packages-select "${PKG_NAME}"

echo "[OK] build done. run:"
echo "source /opt/ros/humble/setup.bash"
echo "source ${WS_DIR}/install/setup.bash"
