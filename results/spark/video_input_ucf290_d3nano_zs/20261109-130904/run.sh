#!/usr/bin/env bash
# d3-nano zero-shot, test 290 videos, each window's 8 frames sent as one video (sr-score.py --send-as video), 1x.
set -euo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
R=results/spark/video_input_ucf290_d3nano_zs/20261109-130904
echo "start $(date -u +%FT%TZ)"
timeout 4h uv run --frozen python scripts/sr-score.py --frames-dir $F --index $I --send-as video --upscale none --out $R/video-1x/scores.jsonl
echo "video-1x done $(date -u +%FT%TZ)"
