#!/bin/bash
# One job: $1 = mixkl or fov2, $2 = --upscale value, $3 = optional window limit (smoke test).
# Output: /content/out/<model>/<upscale>/scores.jsonl; copied to Drive s1-colab-scale-score/out/ at the end.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH S1_ASSETS=/content/s1-assets
cd /content/repo
declare -A ADAPTER=([mixkl]=results/pro6000/floor_d3nano_c/20261031-045123 [fov2]=results/pro6000/fov2_d3nano_c/20261101-183539)
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
O=/content/out/$1/$2
LIMIT=()
[ -n "${3:-}" ] && { LIMIT=(--limit-windows $3); O=/content/smoke/$1/$2; }
mkdir -p $O
timeout 12h uv run --frozen python scripts/sr-score.py --adapter ${ADAPTER[$1]} --frames-dir /content/frames \
    --index $I --upscale $2 --out $O/scores.jsonl "${LIMIT[@]}" > $O.log 2>&1 || { echo "FAILED $1 $2"; tail -5 $O.log; exit 1; }
mkdir -p /content/drive/MyDrive/s1-colab-scale-score/out/$1/$2
cp $O/* $O.log /content/drive/MyDrive/s1-colab-scale-score/out/$1/$2/
echo "DONE $1 $2 $(tail -1 $O.log)"
