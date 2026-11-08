#!/usr/bin/env bash
# V-JEPA 2 gate calibration, 2026-11-08, GB10 (spark). CPU only. Run from the worktree root.
set -euo pipefail
D=results/spark/vjepa_gate_calib_ucf117/20261108-063111
timeout 3600 uv run --frozen python $D/calibrate.py --config $D/config.yaml --out-dir $D 2>&1 | grep -v "nstalled" | tee $D/run.log
