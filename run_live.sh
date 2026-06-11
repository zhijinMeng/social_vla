#!/usr/bin/env bash
# 一键启动真人对话（自动选用 MERCI venv 若存在）
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "${SOCIAL_VLA_PYTHON:-}" ]]; then
  PY="$SOCIAL_VLA_PYTHON"
elif [[ -x "../MERCI_Plus/analysis/.venv/bin/python" ]]; then
  PY="../MERCI_Plus/analysis/.venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

exec "$PY" run_live.py "$@"
