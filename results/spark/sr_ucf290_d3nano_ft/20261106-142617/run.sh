#!/usr/bin/env bash
# full-frame-sr with fine-tuned adapters (Eric, 2026-11-06). Per model: B (bicubic 2x), then C (SR 2x).
set -euo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
SR=$S1_FEATURES/frames/frames_ucf290_f8_sr2/20261106-040502
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
R=results/spark/sr_ucf290_d3nano_ft/20261106-142617
for m in floor_d3nano_c/20261031-045123 fov2_d3nano_c/20261101-183539; do
  n=${m%%/*}; A=results/pro6000/$m
  echo "$n B start $(date -u +%FT%TZ)"
  timeout 7h uv run --frozen python scripts/sr-score.py --frames-dir $F --index $I --upscale bicubic2 --adapter $A --out $R/$n/B/scores.jsonl
  echo "$n C start $(date -u +%FT%TZ)"
  timeout 7h uv run --frozen python scripts/sr-score.py --frames-dir $SR --index $I --upscale none --adapter $A --out $R/$n/C/scores.jsonl
  echo "$n done $(date -u +%FT%TZ)"
done
