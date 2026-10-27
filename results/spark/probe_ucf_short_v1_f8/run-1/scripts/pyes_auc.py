"""Test window AUC of each P(Yes) reading (ties count half)."""
import os
import json
from pathlib import Path

import numpy as np

S = Path(__file__).parent
d = np.load(S / "pyes_recompute.npy", allow_pickle=True).item()
labels = np.array([json.loads(l)["label"] for l in open(os.environ["S1_FEATURES"] + "/probe_ucf_short_v1_f8/C/run-1/test/meta.jsonl")])


def auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


print("test windows", len(labels), "positive", int(labels.sum()))
for c in "ABC":
    print(c, {k: round(auc(d[f"{c}/test/{k}"], labels), 3) for k in ("head_AB", "head_YesNo", "readout_AB")},
          "share > 0.5 (A/B code):", round(float((d[f"{c}/test/head_AB"] > 0.5).mean()), 3))
