#!/usr/bin/env bash
# full-frame-sr full run (proposal .specs/changes/2026-11-06-full-frame-sr, approved 2026-11-06). SR, then B, then C.
set -euo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
SR=$S1_FEATURES/frames/frames_ucf290_f8_sr2/20261106-040502
R=results/spark/sr_ucf290_d3nano_zs/20261106-040502
echo "start $(date -u +%FT%TZ)"
timeout 6h uv run --frozen python scripts/sr-frames.py --frames-dir $F --index $I --out-dir $SR
mkdir -p $R/sr && cp $SR/config.yaml $SR/sr-log.jsonl $R/sr/
echo "sr done $(date -u +%FT%TZ)"
timeout 7h uv run --frozen python scripts/sr-score.py --frames-dir $F --index $I --upscale bicubic2 --out $R/B/scores.jsonl
echo "B done $(date -u +%FT%TZ)"
timeout 7h uv run --frozen python scripts/sr-score.py --frames-dir $SR --index $I --upscale none --out $R/C/scores.jsonl
echo "C done $(date -u +%FT%TZ)"
