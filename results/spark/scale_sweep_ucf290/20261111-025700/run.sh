#!/usr/bin/env bash
# Scale sweep, GB10 part (Eric, 2026-11-11): fov2 1x, mix + KL 1.25x, fov2 1.25x (bicubic, all 8 full frames).
# The other 6 jobs (1.5x, 1.75x, 3x of both models) run on Colab: results/colab/scale_sweep_ucf290/20261110-153049.
# $1 = PID to wait for (the interpolation queue), so the GB10 runs one scoring job at a time.
set -uo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
R=results/spark/scale_sweep_ucf290/20261111-025700
F=$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529
I=results/spark/frames_ucf290_f8/20261105-180529/windows.jsonl
declare -A ADAPTER=([mixkl]=results/pro6000/floor_d3nano_c/20261031-045123 [fov2]=results/pro6000/fov2_d3nano_c/20261101-183539)

for i in $(seq 1 1440); do kill -0 "$1" 2>/dev/null || break; sleep 60; done   # cap 24 h
kill -0 "$1" 2>/dev/null && { echo "queue still running after 24 h, stop"; exit 1; }
echo "queue ended $(date -u +%FT%TZ)"

set -e
for job in "fov2 none" "mixkl bicubic1.25" "fov2 bicubic1.25"; do
  read -r model up <<< "$job"
  avail=$(free -g | awk '/^Mem:/ {print $7}'); echo "available ${avail} GB"
  [ "$avail" -ge 15 ] || { echo "not enough memory, stop"; exit 1; }
  SCORE=(uv run --frozen python scripts/sr-score.py --adapter ${ADAPTER[$model]} --frames-dir $F --index $I --upscale $up)
  timeout 15m "${SCORE[@]}" --limit-windows 5 --out results/smoke/scale_sweep/$model-$up-$(date -u +%H%M%S)/scores.jsonl
  echo "$model $up smoke ok, start $(date -u +%FT%TZ)"
  timeout 6h "${SCORE[@]}" --out $R/$model/$up/scores.jsonl
  echo "$model $up done $(date -u +%FT%TZ)"
done
echo "all done $(date -u +%FT%TZ)"
