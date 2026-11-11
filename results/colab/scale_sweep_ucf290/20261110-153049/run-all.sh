#!/bin/bash
# All 9 jobs, $1 at a time (default 3). Larger factors first, so the longest job does not start last.
# Every 10 minutes, partial scores are copied to Drive out-partial/ (sr-score.py resumes from them).
set -uo pipefail
P=${1:-3}
mkdir -p /content/drive/MyDrive/s1-colab-scale-score/out-partial
( while true; do sleep 600; cp -r /content/out/. /content/drive/MyDrive/s1-colab-scale-score/out-partial/ 2>/dev/null; done ) &
SYNC=$!
printf '%s\n' "fov2 bicubic3" "mixkl bicubic3" "fov2 bicubic1.75" "mixkl bicubic1.75" "fov2 bicubic1.5" \
    "mixkl bicubic1.5" "fov2 bicubic1.25" "mixkl bicubic1.25" "fov2 none" |
  xargs -P "$P" -L 1 bash /content/drive/MyDrive/s1-colab-scale-score/score.sh
kill $SYNC
echo ALL_DONE
