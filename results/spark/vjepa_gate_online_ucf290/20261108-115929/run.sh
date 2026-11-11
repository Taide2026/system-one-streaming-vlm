#!/usr/bin/env bash
# V-JEPA 2 gate with an online threshold, 2026-11-08, GB10 (spark). CPU only. Run from the worktree root.
set -euo pipefail
D=results/spark/vjepa_gate_online_ucf290/20261108-115929
timeout 1800 uv run --frozen python $D/online.py --config $D/config.yaml 2>&1 | grep --line-buffered -v "nstalled" | tee $D/run.log
