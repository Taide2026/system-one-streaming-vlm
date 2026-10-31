#!/usr/bin/env bash
# run-1 of encoder_probe_ucf_short_v1_f8: extract every encoder, check decision 9, fit every feature folder, compare.
# This folder was named run-1 until 2026-10-31; the feature folders in $S1_FEATURES keep the run name run-1.
# Usage (worktree root): bash results/spark/encoder_probe_ucf_short_v1_f8/20261028-five-encoders/scripts/run.sh
# One GPU job at a time. Each step has a timeout. The script stops at the first failure (set -e).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
C=configs/encoder_probe_ucf_short_v1_f8.yaml
R=results/spark/encoder_probe_ucf_short_v1_f8/20261028-five-encoders
F="$S1_FEATURES/encoder_probe_ucf_short_v1_f8"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/logs/run.log"; }

for e in qwen35-vit internvit-300m siglip2-so400m dinov3-vitl vjepa2-vitl; do
  log "extract $e: start, free $(free -g | awk '/Mem:/ {print $7}') GiB"
  timeout 3h uv run s1stream encoder-extract "$C" --encoder "$e" --run run-1 > "$R/logs/extract-$e.log" 2>&1
  log "extract $e: done"
done

log "decision 9 check"
timeout 30m uv run python results/spark/encoder_probe_ucf_short_v1_f8/smoke-1/scripts/check_equal.py run-1 \
  > "$R/logs/check-equal.log" 2>&1
log "decision 9 check: passed"

for f in qwen35-vit internvit-300m internvit-300m-head siglip2-so400m siglip2-so400m-head dinov3-vitl vjepa2-vitl; do
  log "fit $f: start"
  timeout 3h uv run s1stream probe-fit "$F/$f/run-1" --out "$R/fit-$f" --device cuda > "$R/logs/fit-$f.log" 2>&1
  log "fit $f: done"
done

log "compare: start"
timeout 1h uv run s1stream encoder-compare "$C" "$R" > "$R/logs/compare.log" 2>&1
log "compare: done"
