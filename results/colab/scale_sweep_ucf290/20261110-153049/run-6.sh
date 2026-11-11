#!/bin/bash
# 6 jobs on Colab (Eric, 2026-11-11: 3x, 1.75x, 1.5x of both models); 1x and 1.25x run on the GB10. $1 at a time (default 3).
# Every 10 minutes, partial scores are copied to Drive out-partial/ (sr-score.py resumes from them).
set -uo pipefail
P=${1:-3}
mkdir -p /content/drive/MyDrive/s1-colab-scale-score/out-partial
( while true; do sleep 600; cp -r /content/out/. /content/drive/MyDrive/s1-colab-scale-score/out-partial/ 2>/dev/null; done ) &
SYNC=$!
printf '%s\n' "fov2 bicubic3" "mixkl bicubic3" "fov2 bicubic1.75" "mixkl bicubic1.75" "fov2 bicubic1.5" "mixkl bicubic1.5" |
  xargs -P "$P" -L 1 bash /content/drive/MyDrive/s1-colab-scale-score/score.sh
kill $SYNC
echo ALL_DONE
