#!/usr/bin/env bash
# WiSE-FT alpha 0.5 (results/pro6000/wise_a5: ft_ucf_v1_d3nano_hard/20261023-200623, lora_B x 0.5, readout and T
# interpolated), test 290 videos: original frames (draft), then bicubic 2x (B). Runs next to the floor2x_hn training.
set -euo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
A=results/pro6000/wise_a5
R=results/spark/sr_ucf290_d3nano_ft/20261109-141810
echo "start $(date -u +%FT%TZ)"
timeout 8h uv run --frozen python scripts/sr-score.py --adapter $A --frames-dir $F --index $I --upscale none --out $R/wise_a5/draft/scores.jsonl
echo "draft done $(date -u +%FT%TZ)"
timeout 16h uv run --frozen python scripts/sr-score.py --adapter $A --frames-dir $F --index $I --upscale bicubic2 --out $R/wise_a5/B/scores.jsonl
echo "B done $(date -u +%FT%TZ)"
