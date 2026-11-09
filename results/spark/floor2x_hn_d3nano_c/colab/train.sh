#!/bin/bash
# $1 = smoke (5 steps) or full. Output: /content/out/<mode>; log: /content/train-<mode>.log
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH S1_ASSETS=/content/s1-assets
cd /content/repo
LIMIT=""; [ "$1" = smoke ] && LIMIT="--limit-steps 5"
timeout 6h uv run --frozen --with huggingface_hub python scripts/train-floor.py /content/floor2x_hn_d3nano_c.yaml \
    --out /content/out/$1 $LIMIT
echo TRAIN_DONE $1
