#!/bin/bash
# Debug version with full logging

cd "$(dirname "$0")"

# Activate virtual environment
source /home/zhijinmeng/Research/HF_Data/percy_data/.venv/bin/activate

# Load environment variables
export $(grep -v '^#' /home/zhijinmeng/Research/.env.local | grep -v '^$' | xargs)

# Enable debug logging
export PYTHONUNBUFFERED=1

# Run with logging to file
python -W ignore social_vla/pipeline/live_demo.py \
  --camera 0 \
  --device cuda \
  --mic 4 \
  --vad-backend silero \
  --audio-gain 2.0 \
  --score-threshold 0.45 \
  --qwen 2>&1 | tee qwen_debug.log
