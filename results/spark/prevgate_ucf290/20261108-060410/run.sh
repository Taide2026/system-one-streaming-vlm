#!/usr/bin/env bash
# Previous-window gate, 2026-11-08, GB10 (spark). CPU only. Run from the worktree root.
set -euo pipefail
D=results/spark/prevgate_ucf290/20261108-060410
timeout 3600 uv run --frozen python $D/gate.py --config $D/config.yaml --out-dir $D 2>&1 | grep -v "nstalled" | tee $D/run.log
