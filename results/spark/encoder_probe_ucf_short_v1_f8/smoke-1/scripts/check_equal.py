"""Proposal decision 9: qwen35-vit through encoder-extract must equal the vit-1 ViT features.

vit-1 (results/spark/probe_ucf_short_v1_f8/vit-1) wrote visual.npy = token mean and last.npy = token max of every ViT
state. Here patch_mean must equal visual and patch_max must equal last, for the windows of the encoder-extract run:
cosine >= MIN_COSINE for every window and state, and max abs difference <= MAX_RATIO x max abs value per kind and set.
The first threshold (max abs difference <= 1e-3 x max abs value) is below the bfloat16 step of 3.9e-3; Eric changed it
on 2026-10-28 (see README.md).

Usage (worktree root): uv run python results/spark/encoder_probe_ucf_short_v1_f8/smoke-1/scripts/check_equal.py <run>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

features = Path(os.environ["S1_FEATURES"])
new = features / "encoder_probe_ucf_short_v1_f8" / "qwen35-vit" / sys.argv[1]
old = features / "probe_ucf_short_v1_f8" / "V" / "vit-1"
MIN_COSINE, MAX_RATIO = 0.9999, 2e-2
failed = False
for name in ("train_anomaly", "train_normal", "test"):
    new_meta = [json.loads(line) for line in (new / name / "meta.jsonl").read_text().splitlines()]
    old_meta = [json.loads(line) for line in (old / name / "meta.jsonl").read_text().splitlines()][:len(new_meta)]
    keys = ("video", "window", "start_s", "end_s", "label", "holmes")
    if [[row[k] for k in keys] for row in new_meta] != [[row[k] for k in keys] for row in old_meta]:
        sys.exit(f"{name}: window plan differs from vit-1")
    for new_kind, old_kind in (("patch_mean", "visual"), ("patch_max", "last")):
        a = np.load(new / name / f"{new_kind}.npy")
        b = np.load(old / name / f"{old_kind}.npy", mmap_mode="r")[:len(a)].astype(np.float32)
        diff = float(np.abs(a - b).max())
        scale = float(np.abs(b).max())
        cosine = float((np.sum(a * b, -1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)).min())
        ok = a.shape == b.shape and diff <= MAX_RATIO * scale and cosine >= MIN_COSINE
        failed |= not ok
        per_state = np.abs(a - b).max(axis=(0, 2)) / np.maximum(np.abs(b).max(axis=(0, 2)), 1e-12)
        print(f"{name} {new_kind} vs {old_kind}: shape {a.shape} vs {b.shape}, max abs diff {diff:.4g}, "
              f"max abs {scale:.4g}, ratio {diff / scale:.2e}, worst state ratio {per_state.max():.2e} "
              f"(state {int(per_state.argmax())}), min cosine {cosine:.6f} -> {'ok' if ok else 'FAIL'}")
print("decision 9 check:", "FAIL" if failed else "passed")
sys.exit(1 if failed else 0)
