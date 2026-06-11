#!/usr/bin/env bash
# Link TalkNet weights and fetch S3FD face detector if missing.
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

S3FD="${REPO_DIR}/model/faceDetector/s3fd/sfd_face.pth"
LOCAL_PRETRAIN="${REPO_DIR}/pretrain_TalkSet.model"
WEIGHTS_DIR="${REPO_DIR}/../weights"
EXT_PRETRAIN="${WEIGHTS_DIR}/pretrain_TalkSet.model"

if [[ -f "${LOCAL_PRETRAIN}" && ! -L "${LOCAL_PRETRAIN}" ]]; then
  echo "[ok] pretrain_TalkSet.model ($(du -h "${LOCAL_PRETRAIN}" | cut -f1))"
elif [[ -f "${EXT_PRETRAIN}" ]]; then
  ln -sf "${EXT_PRETRAIN}" "${LOCAL_PRETRAIN}"
  echo "[ok] pretrain_TalkSet.model -> ${EXT_PRETRAIN}"
else
  echo "[warn] pretrain_TalkSet.model not found (expected in repo root or ${EXT_PRETRAIN})"
fi

if [[ -f "${S3FD}" ]]; then
  echo "[ok] sfd_face.pth already present ($(du -h "${S3FD}" | cut -f1))"
else
  echo "[fetch] downloading sfd_face.pth (~90MB)..."
  if command -v gdown >/dev/null 2>&1; then
    gdown 1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt -O "${S3FD}"
  else
    echo "[error] gdown not found. pip install gdown && retry"
    exit 1
  fi
fi
