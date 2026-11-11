#!/usr/bin/env bash
# GB10 queue (Eric, 2026-11-10): wait for the WiSE run (PID $1), then floor2x_d3nano_c 1x / 2x, then interpolation.
set -uo pipefail
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov
for i in $(seq 1 1080); do kill -0 "$1" 2>/dev/null || break; sleep 60; done   # cap 18 h
kill -0 "$1" 2>/dev/null && { echo "WiSE still running after 18 h, stop"; exit 1; }
echo "WiSE ended $(date -u +%FT%TZ)"
A=results/spark/sr_ucf290_d3nano_ft/20261110-153049
B=results/spark/interp_ucf290_mixkl/20261110-153049
bash $A/run.sh > $A/run.log 2>&1 || { echo "floor2x run failed, see $A/run.log"; exit 1; }
echo "floor2x done $(date -u +%FT%TZ)"
bash $B/run.sh > $B/run.log 2>&1 || { echo "interp run failed, see $B/run.log"; exit 1; }
echo "queue done $(date -u +%FT%TZ)"
