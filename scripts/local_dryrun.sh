#!/usr/bin/env bash
# Laptop smoke test. Loads no models. Safe to run anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
unset MOSHI_EVAL_ALLOW_HEAVY || true

python3 -m moshi_eval -c config/sample_offline.yaml run
echo
echo "--- first judge prompt (this is what Qwen will see) ---"
python3 -m moshi_eval -c config/sample_offline.yaml pair --print-prompt 1
