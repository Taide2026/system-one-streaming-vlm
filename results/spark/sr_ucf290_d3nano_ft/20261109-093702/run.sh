#!/bin/bash
# fovfull2x_d3nano_c scored in its own training format (Eric, 2026-11-09): frames 1, 3, 5, 7 bicubic 2x.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
R=$(dirname "$0")
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
A=results/pro6000/fovfull2x_d3nano_c/20261106-013724
timeout 4h uv run --frozen python scripts/sr-score.py --frames-dir $F --index $I --upscale odd-bicubic2 --adapter $A \
    --out $R/fovfull2x_d3nano_c/odd/scores.jsonl
echo "odd done $(date -u +%FT%TZ)"
