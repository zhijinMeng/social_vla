#!/usr/bin/env bash
# 首次解压后：创建独立 Python 环境并安装依赖（摄像头模式，无需 ROS）
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

VENV="${REPO_DIR}/.venv-talknet"

if [[ ! -d "${VENV}" ]]; then
  echo "[1/3] 创建虚拟环境: ${VENV}"
  python3 -m venv "${VENV}"
else
  echo "[1/3] 虚拟环境已存在: ${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip wheel

echo "[2/3] 安装 PyTorch（CUDA 12.x，无 GPU 可改用 CPU 版见 使用说明.md）"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "[3/3] 安装其余依赖"
pip install -r "${REPO_DIR}/requirement.txt"

echo ""
echo "完成。激活环境: source ${VENV}/bin/activate"
echo "运行演示:   bash scripts/run_camera_local.sh"
