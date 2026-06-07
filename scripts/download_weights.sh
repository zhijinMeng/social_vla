#!/usr/bin/env bash
# Download TalkNet pretrained weights (public). LAM weights require Ego4D license.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W="$ROOT/weights"
mkdir -p "$W"

TALKNET="$W/pretrain_TalkSet.model"
if [[ ! -f "$TALKNET" ]]; then
  echo "Downloading TalkNet pretrained model..."
  if ! command -v gdown &>/dev/null; then
    python -m pip install -q gdown
  fi
  gdown "https://drive.google.com/uc?id=1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea" -O "$TALKNET"
fi

echo "TalkNet weights: $TALKNET"
echo "LAM GazeLSTM: place checkpoint at $W/lam_gazelstm.pth (from Ego4D social benchmark)"
