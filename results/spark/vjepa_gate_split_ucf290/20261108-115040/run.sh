#!/usr/bin/env bash
# Split check of the V-JEPA 2 gate, 2026-11-08, GB10 (spark). CPU only. Run from the worktree root.
set -euo pipefail
D=results/spark/vjepa_gate_split_ucf290/20261108-115040
timeout 1800 uv run --frozen python $D/split.py --config $D/config.yaml 2>&1 | grep --line-buffered -v "nstalled" | tee $D/run.log
