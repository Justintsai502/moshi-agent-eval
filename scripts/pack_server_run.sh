#!/usr/bin/env bash
# Real clip-pack run on the GPU box.
#   usage: scripts/pack_server_run.sh [config] [run_name]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
export MOSHI_EVAL_ALLOW_HEAVY=1

CFG="${1:-config/pack_server.yaml}"
RUN="${2:-}"
ARGS=(-c "$CFG")
[[ -n "$RUN" ]] && ARGS+=(-s "run_name=$RUN")

python3 scripts/vad_preflight.py          # cheap sanity check first
python3 -m moshi_eval "${ARGS[@]}" pack
