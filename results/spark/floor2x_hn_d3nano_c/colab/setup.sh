#!/bin/bash
# Colab G4 setup for floor2x_hn_d3nano_c (2026-11-09). Drive folder s1-colab-floor2x_hn_d3nano_c holds
# code.tar.gz (the small-event-fov worktree, tracked files + this change), extract.tar (records.jsonl, val.jsonl)
# and frames-*.tar.zst (the frame cache, uint8 .npy per window).
set -euo pipefail
D=/content/drive/MyDrive/s1-colab-floor2x_hn_d3nano_c
mkdir -p /content/repo /content/frames /content/extract /content/s1-assets
tar -xzf $D/code.tar.gz -C /content/repo
tar -xf $D/extract.tar -C /content/extract
for f in $D/frames-*.tar.zst; do zstd -q -d -T16 -c "$f" | tar -xf - -C /content/frames; done
echo "frames: $(ls /content/frames | wc -l)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH S1_ASSETS=/content/s1-assets
cd /content/repo && uv sync --frozen
uv run --frozen s1stream fetch d3-nano
sed -e 's|  # records_dir: .*|  records_dir: /content/extract|' -e 's|  # frames_cache: .*|  frames_cache: /content/frames|' \
    configs/floor2x_hn_d3nano_c.yaml > /content/floor2x_hn_d3nano_c.yaml
grep -n "records_dir\|frames_cache" /content/floor2x_hn_d3nano_c.yaml
echo SETUP_DONE
