#!/usr/bin/env bash
# 打开发给同事的完整包：源码 + 权重 + 说明，不含 venv / ROS / TalkNCE
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(cd "${REPO_DIR}/.." && pwd)"
DATE_TAG="$(date +%Y%m%d)"
RELEASE_NAME="talknet-asd-camera-release-${DATE_TAG}"
STAGING="${PARENT_DIR}/_pack_staging/${RELEASE_NAME}"
ARCHIVE="${PARENT_DIR}/${RELEASE_NAME}.tar.gz"

PRETRAIN_SRC="${PARENT_DIR}/weights/pretrain_TalkSet.model"
if [[ ! -f "${PRETRAIN_SRC}" ]]; then
  PRETRAIN_SRC="${REPO_DIR}/pretrain_TalkSet.model"
fi
if [[ ! -f "${PRETRAIN_SRC}" ]]; then
  echo "[error] pretrain_TalkSet.model not found"
  exit 1
fi

S3FD_SRC="${REPO_DIR}/model/faceDetector/s3fd/sfd_face.pth"
if [[ ! -f "${S3FD_SRC}" ]]; then
  echo "[error] sfd_face.pth not found — run scripts/setup_weights.sh first"
  exit 1
fi

echo "==> staging: ${STAGING}"
rm -rf "${STAGING}"
mkdir -p "${STAGING}"

RSYNC_EXCLUDES=(
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.venv-talknet/'
  --exclude '.git/'
  --exclude 'TalkNCE-main/'
  --exclude 'robot_ws/'
  --exclude 'demo/'
  --exclude '*.avi'
  --exclude '*.mp4'
  --exclude 'engines/'
  --exclude '_pack_staging/'
  --exclude 'pretrain_TalkSet.model'
)

rsync -a "${RSYNC_EXCLUDES[@]}" "${REPO_DIR}/" "${STAGING}/"

# 真实权重文件（不用软链，同事解压即可用）
cp -fL "${PRETRAIN_SRC}" "${STAGING}/pretrain_TalkSet.model"
cp -f "${S3FD_SRC}" "${STAGING}/model/faceDetector/s3fd/sfd_face.pth"

chmod +x "${STAGING}/scripts/"*.sh 2>/dev/null || true

# 清单
{
  echo "# ${RELEASE_NAME}"
  echo "created: $(date -Iseconds)"
  echo "pretrain_TalkSet.model: $(du -h "${STAGING}/pretrain_TalkSet.model" | cut -f1)"
  echo "sfd_face.pth: $(du -h "${STAGING}/model/faceDetector/s3fd/sfd_face.pth" | cut -f1)"
  echo ""
  find "${STAGING}" -type f | sort | head -500
  echo "..."
  echo "total files: $(find "${STAGING}" -type f | wc -l)"
} > "${STAGING}/MANIFEST.txt"

echo "==> archive: ${ARCHIVE}"
tar -czf "${ARCHIVE}" -C "$(dirname "${STAGING}")" "${RELEASE_NAME}"

rm -rf "$(dirname "${STAGING}")"

ls -lh "${ARCHIVE}"
echo ""
echo "完成。发给同事："
echo "  ${ARCHIVE}"
echo ""
echo "同事解压后："
echo "  tar -xzf $(basename "${ARCHIVE}")"
echo "  cd ${RELEASE_NAME}"
echo "  bash scripts/setup_venv.sh"
echo "  bash scripts/run_camera_local.sh"
