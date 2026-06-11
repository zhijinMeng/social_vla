#!/bin/bash
# Simplified startup script for QwenOmni dialogue system

cd "$(dirname "$0")"

# Activate virtual environment
source /home/zhijinmeng/Research/HF_Data/percy_data/.venv/bin/activate

# Load environment variables
export $(grep -v '^#' /home/zhijinmeng/Research/.env.local | grep -v '^$' | xargs)

# Run with sensible defaults
python -W ignore social_vla/pipeline/live_demo.py \
  --camera 0 \
  --device cuda \
  --mic 4 \
  --vad-backend silero \
  --audio-gain 2.0 \
  --score-threshold 0.45 \
  --qwen
