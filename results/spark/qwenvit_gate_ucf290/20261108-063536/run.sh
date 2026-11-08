#!/usr/bin/env bash
# Qwen ViT gate on 290 videos, 2026-11-08, GB10 (spark). CPU only. Run from the worktree root.
set -euo pipefail
D=results/spark/qwenvit_gate_ucf290/20261108-063536
timeout 3600 uv run --frozen python $D/gate290.py --config $D/config.yaml --out-dir $D 2>&1 | grep -v "nstalled" | tee $D/run.log
