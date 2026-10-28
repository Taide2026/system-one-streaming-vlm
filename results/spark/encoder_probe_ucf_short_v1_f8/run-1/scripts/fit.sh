#!/usr/bin/env bash
# run-1, part 2: fit every feature folder, then compare. run.sh stopped at the decision 9 check (logs/check-equal.log);
# logs/verify-pipeline.log shows the cause (the d3 runtime's linear patch embedding), and Eric chose to go on (2026-10-28).
# Usage (worktree root): bash results/spark/encoder_probe_ucf_short_v1_f8/run-1/scripts/fit.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
C=configs/encoder_probe_ucf_short_v1_f8.yaml
R=results/spark/encoder_probe_ucf_short_v1_f8/run-1
F="$S1_FEATURES/encoder_probe_ucf_short_v1_f8"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/logs/run.log"; }

log "decision 9 check: FAIL, accepted by Eric (see logs/verify-pipeline.log)"
for f in qwen35-vit internvit-300m internvit-300m-head siglip2-so400m siglip2-so400m-head dinov3-vitl vjepa2-vitl; do
  log "fit $f: start, free $(free -g | awk '/Mem:/ {print $7}') GiB"
  timeout 3h uv run s1stream probe-fit "$F/$f/run-1" --out "$R/fit-$f" --device cuda > "$R/logs/fit-$f.log" 2>&1
  log "fit $f: done"
done

log "compare: start"
timeout 1h uv run s1stream encoder-compare "$C" "$R" > "$R/logs/compare.log" 2>&1
log "compare: done"
