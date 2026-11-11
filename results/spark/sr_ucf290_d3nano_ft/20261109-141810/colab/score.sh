#!/bin/bash
# $1 = none (draft) or bicubic2 (B); $2 = optional --limit-windows N. Output: /content/out/<upscale>/scores.jsonl,
# copied to Drive s1-colab-wise-score/out-<upscale>/ at the end.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH S1_ASSETS=/content/s1-assets
cd /content/repo
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
timeout 6h uv run --frozen python scripts/sr-score.py --adapter results/pro6000/wise_a5 --frames-dir /content/frames \
    --index $I --upscale $1 --out /content/out/$1/scores.jsonl ${2:-}
mkdir -p /content/drive/MyDrive/s1-colab-wise-score/out-$1
cp /content/out/$1/* /content/drive/MyDrive/s1-colab-wise-score/out-$1/
echo SCORE_DONE $1
