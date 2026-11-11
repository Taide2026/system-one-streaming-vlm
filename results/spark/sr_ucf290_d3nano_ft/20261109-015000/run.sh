#!/usr/bin/env bash
# fovfull2x_d3nano_c on test 290 (Eric, 2026-11-09): does training on 2x frames lower the 2x scores of normal windows?
# Same recipe as floor_d3nano_c (mix + KL 0.5); training input had frames 1, 3, 5, 7 at full-frame 2x.
# draft = original frames (1x), B = all 8 frames bicubic 2x. Smoke test: results/smoke/fovfull2x_ft/20261109-014834/.
set -euo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
R=results/spark/sr_ucf290_d3nano_ft/20261109-015000
A=results/pro6000/fovfull2x_d3nano_c/20261106-013724
echo "fovfull2x draft start $(date -u +%FT%TZ)"
timeout 3h uv run --frozen python scripts/sr-score.py --frames-dir $F --index $I --upscale none --adapter $A --out $R/fovfull2x_d3nano_c/draft/scores.jsonl
echo "fovfull2x B start $(date -u +%FT%TZ)"
timeout 6h uv run --frozen python scripts/sr-score.py --frames-dir $F --index $I --upscale bicubic2 --adapter $A --out $R/fovfull2x_d3nano_c/B/scores.jsonl
echo "fovfull2x done $(date -u +%FT%TZ)"
