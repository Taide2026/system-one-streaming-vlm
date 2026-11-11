#!/bin/bash
# Colab setup for the WiSE-FT alpha 0.5 scoring on test 290 (2026-11-09). Drive folder s1-colab-wise-score holds
# code.tar.gz (small-event-fov worktree: tracked files + scripts/sr-score.py with --send-as), wise_a5.tar
# (results/pro6000/wise_a5) and frames-group00..09.tar.zst (frames_ucf290_f8/20261105-180529, one uint8 npy per video).
set -euo pipefail
D=/content/drive/MyDrive/s1-colab-wise-score
mkdir -p /content/repo /content/frames /content/s1-assets
tar -xzf $D/code.tar.gz -C /content/repo
tar -xf $D/wise_a5.tar -C /content/repo
for f in $D/frames-group*.tar.zst; do zstd -q -d -T16 -c "$f" | tar -xf - -C /content/frames; done
echo "frames: $(ls /content/frames | wc -l)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH S1_ASSETS=/content/s1-assets
cd /content/repo && uv sync --frozen
uv run --frozen s1stream fetch d3-nano
echo SETUP_DONE
