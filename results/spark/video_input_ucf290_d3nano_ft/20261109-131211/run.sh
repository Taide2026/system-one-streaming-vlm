#!/usr/bin/env bash
# floor_d3nano_c (mix + KL), test 290 videos, each window's 8 frames sent as one video (sr-score.py --send-as video), 1x.
set -euo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
A=results/pro6000/floor_d3nano_c/20261031-045123
R=results/spark/video_input_ucf290_d3nano_ft/20261109-131211
echo "start $(date -u +%FT%TZ)"
timeout 4h uv run --frozen python scripts/sr-score.py --adapter $A --frames-dir $F --index $I --send-as video --upscale none --out $R/floor_d3nano_c/video-1x/scores.jsonl
echo "video-1x done $(date -u +%FT%TZ)"
