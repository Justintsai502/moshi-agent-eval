#!/usr/bin/env bash
# Real IF4 run on the GPU box.
#   usage: scripts/if4_server_run.sh [config] [run_name]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
export MOSHI_EVAL_ALLOW_HEAVY=1
export HF_HOME="${HF_HOME:-/work/b12901015/hf_cache}"

CFG="${1:-config/if4_server.yaml}"
RUN="${2:-}"
ARGS=(-c "$CFG")
[[ -n "$RUN" ]] && ARGS+=(-s "run_name=$RUN")

python3 -m moshi_eval "${ARGS[@]}" timed
python3 scripts/inspect_run.py "runs/${RUN:-if4_p0_step1350}" | tail -20
