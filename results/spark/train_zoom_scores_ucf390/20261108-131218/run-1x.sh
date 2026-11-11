#!/usr/bin/env bash
# 1x scores (no up-scale) of the 390 train videos on the GB10 (spark), 2026-11-08. Run from the worktree root.
# Smoke before: 5 windows each, 0.28 s/window at 1x, 0.70 s/window at 2x, peak GPU 4.6 GB; free -g available 106 GB.
set -euo pipefail
R=results/spark/train_zoom_scores_ucf390/20261108-131218
F=$S1_FEATURES/frames/frames_ucf390train_f8/20261108-131218
A=results/pro6000/floor_d3nano_c/20261031-045123
echo "zeroshot 1x start $(date -u +%FT%TZ)"
timeout 3h uv run --frozen python scripts/sr-score.py --frames-dir $F --index $R/windows.jsonl --upscale none --out $R/zeroshot/1x/scores.jsonl
echo "mixkl 1x start $(date -u +%FT%TZ)"
timeout 3h uv run --frozen python scripts/sr-score.py --frames-dir $F --index $R/windows.jsonl --upscale none --adapter $A --out $R/mixkl/1x/scores.jsonl
echo "done $(date -u +%FT%TZ)"
