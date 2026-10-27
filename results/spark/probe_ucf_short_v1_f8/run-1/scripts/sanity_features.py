"""NaN/inf check and value range per condition on states 0, 15, 31, 32 (mmap, one state at a time)."""
import os
import json
import sys

import numpy as np

root = os.environ["S1_FEATURES"] + "/probe_ucf_short_v1_f8"
for c in "CAB":
    for s in ("train_anomaly", "train_normal", "test"):
        bad, top = 0, {}
        for kind in ("last", "visual"):
            a = np.load(f"{root}/{c}/run-1/{s}/{kind}.npy", mmap_mode="r")
            for st in (0, 15, 31, 32):
                x = np.asarray(a[:, st], dtype=np.float32)
                bad += int((~np.isfinite(x)).sum())
                top[f"{kind}{st}"] = round(float(np.abs(x).max()), 1)
        p = np.array([json.loads(l)["p_yes"] for l in open(f"{root}/{c}/run-1/{s}/meta.jsonl")])
        print(c, s, "nonfinite", bad, "absmax", top, "p_yes mean", round(p.mean(), 3), "range", round(p.min(), 3), round(p.max(), 3))
        sys.stdout.flush()
