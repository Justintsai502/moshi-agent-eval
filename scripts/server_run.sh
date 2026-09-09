#!/usr/bin/env bash
# Real run on the GPU box.
#   usage: scripts/server_run.sh [config] [run_name]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
export MOSHI_EVAL_ALLOW_HEAVY=1     # unlocks whisper + vllm

CFG="${1:-config/server_moshi.yaml}"
RUN="${2:-}"
ARGS=(-c "$CFG")
[[ -n "$RUN" ]] && ARGS+=(-s "run_name=$RUN")

# Staged so a judge crash does not throw away the ASR pass.
python3 -m moshi_eval "${ARGS[@]}" asr
python3 -m moshi_eval "${ARGS[@]}" pair
python3 -m moshi_eval "${ARGS[@]}" judge
python3 -m moshi_eval "${ARGS[@]}" report
