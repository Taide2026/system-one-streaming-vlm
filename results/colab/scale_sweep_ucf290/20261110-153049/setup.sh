#!/bin/bash
# Colab setup for the scale sweep (Eric, 2026-11-10). Drive s1-colab-scale-score holds code.tar.gz (small-event-fov
# worktree with scripts/sr-score.py --upscale <method><factor>) and adapters.tar (mix + KL, fov2).
# Frames come from s1-colab-wise-score (frames_ucf290_f8/20261105-180529, one uint8 npy per video).
set -euo pipefail
D=/content/drive/MyDrive/s1-colab-scale-score
W=/content/drive/MyDrive/s1-colab-wise-score
command -v zstd >/dev/null || apt-get -qq install -y zstd
mkdir -p /content/repo /content/frames /content/s1-assets
tar -xzf $D/code.tar.gz -C /content/repo
tar -xf $D/adapters.tar -C /content/repo
for f in $W/frames-group*.tar.zst; do zstd -q -d -T16 -c "$f" | tar -xf - -C /content/frames; done
echo "frames: $(ls /content/frames | wc -l)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH S1_ASSETS=/content/s1-assets
cd /content/repo && uv sync --frozen
uv run --frozen s1stream fetch d3-nano
uv run --frozen pytest -q tests/test_sr_score_scale.py
echo SETUP_DONE
