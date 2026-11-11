#!/usr/bin/env bash
# Local watcher (Eric, 2026-11-11): every 10 min ask the VM; when all 6 jobs are on Drive, stop the VM.
# Stops waiting (VM left running) when a job failed or after 20 h.
cd /home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov/results/colab/scale_sweep_ucf290/20261110-153049
for i in $(seq 1 120); do
  out=$(timeout 120 colab exec -s scale-sweep -f check.py 2>&1 | grep -v -i -E "token|bearer|authorization" | tail -2)
  echo "$(date -u +%FT%TZ) $out"
  echo "$out" | grep -q DONE_OK && { colab stop -s scale-sweep 2>&1 | tail -1; echo "VM stopped $(date -u +%FT%TZ)"; exit 0; }
  echo "$out" | grep -q "True {" && echo "$out" | grep -q "^[A-Za-z]* True" && { echo "a job failed, VM left running"; exit 1; }
  sleep 600
done
echo "20 h passed, VM left running"; exit 1
