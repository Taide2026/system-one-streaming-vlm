#!/usr/bin/env bash
# Frame reading check, 2026-11-07, GB10 (spark). CPU only. Run from the worktree root.
set -euo pipefail
D=results/spark/frame_read_check/20261107-084258
timeout 1800 uv run --frozen python $D/check.py --config $D/config.yaml \
    --frames-dir "$S1_FEATURES/frames/frames_ucf290_f8/20261105-180529" --out-dir $D 2>&1 | tee $D/run.log
