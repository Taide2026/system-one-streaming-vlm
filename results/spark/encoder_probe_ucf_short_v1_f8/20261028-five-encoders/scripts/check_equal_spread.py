"""Spread of the decision 9 difference on run-1: how many (window, state) pairs miss the cosine threshold.

Usage (worktree root): uv run python results/spark/encoder_probe_ucf_short_v1_f8/20261028-five-encoders/scripts/check_equal_spread.py
"""
import os
from pathlib import Path

import numpy as np

features = Path(os.environ["S1_FEATURES"])
new = features / "encoder_probe_ucf_short_v1_f8" / "qwen35-vit" / "run-1"
old = features / "probe_ucf_short_v1_f8" / "V" / "vit-1"
for name in ("train_anomaly", "train_normal", "test"):
    for new_kind, old_kind in (("patch_mean", "visual"), ("patch_max", "last")):
        a = np.load(new / name / f"{new_kind}.npy", mmap_mode="r")
        b = np.load(old / name / f"{old_kind}.npy", mmap_mode="r")
        cos = np.stack([np.sum(a[:, s] * b[:, s], -1) / np.linalg.norm(a[:, s], axis=-1)
                        / np.linalg.norm(b[:, s], axis=-1) for s in range(a.shape[1])], 1)
        bad = cos < 0.9999
        windows = bad.any(1)
        print(f"{name} {new_kind}: pairs below 0.9999 {bad.sum()} / {bad.size} ({bad.mean():.3%}), "
              f"windows {windows.sum()} / {len(windows)}, below 0.999 {(cos < 0.999).sum()}, "
              f"median cosine {np.median(cos):.7f}, states with misses {np.flatnonzero(bad.any(0)).tolist()}")
