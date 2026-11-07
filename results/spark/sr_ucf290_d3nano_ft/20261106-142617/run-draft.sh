#!/usr/bin/env bash
# floor_d3nano_c on the original frames (Eric, 2026-11-07), for the TGM gate and paired intervals.
# Resumes the partial draft (3,080 windows) from the stopped first run; same script, same machine.
set -euo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
R=results/spark/sr_ucf290_d3nano_ft/20261106-142617
echo "floor_d3nano_c draft resume start $(date -u +%FT%TZ)"
timeout 4h uv run --frozen python scripts/sr-score.py --frames-dir $F --index $I --upscale none --adapter results/pro6000/floor_d3nano_c/20261031-045123 --out $R/floor_d3nano_c/draft/scores.jsonl
echo "floor_d3nano_c draft done $(date -u +%FT%TZ)"
