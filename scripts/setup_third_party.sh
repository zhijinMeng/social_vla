#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

clone_if_missing() {
  local dir="$1"
  local url="$2"
  local branch="${3:-}"
  if [[ -d "$dir/.git" ]]; then
    echo "OK: $dir"
    return
  fi
  if [[ -n "$branch" ]]; then
    git clone --depth 1 -b "$branch" "$url" "$dir"
  else
    git clone --depth 1 "$url" "$dir"
  fi
}

clone_if_missing "$TP/TalkNet-ASD" "https://github.com/TaoRuijie/TalkNet-ASD.git"
clone_if_missing "$TP/ego4d-lam" "https://github.com/EGO4D/social-interactions.git" "lam"
echo "Third-party repos ready under $TP"
