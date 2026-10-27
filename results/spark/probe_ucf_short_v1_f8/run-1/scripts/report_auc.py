"""Peak frame AUC per condition/probe/feature from the preview compare.json, and P(Yes) frame AUC with code tokens."""
import os
import json
import sys
from pathlib import Path

import numpy as np

from s1stream.probe.compare import frame_metric
from s1stream.probe.extract import load_probe_config
from s1stream.registry import DATASETS
import s1stream.plugins  # noqa: F401

S = Path(sys.argv[1])
r = json.loads((S / "compare-preview/compare.json").read_text())
for m in ("mil", "holmes", "knn"):
    for k in ("last", "visual"):
        cells = []
        for c in "ABC":
            e = r["metrics"][f"{c}/{m}/{k}/frame_auc"]
            i = int(np.nanargmax(e["value"]))
            cells.append(f"{c} {e['value'][i]:.3f} [{e['low'][i]:.3f}, {e['high'][i]:.3f}] @{i}")
        print(m, k, " | ".join(cells))
items = {i.name: i for i in DATASETS.build(load_probe_config(Path("configs/probe_ucf_short_v1_f8.yaml")).test).items()}
meta = [json.loads(l) for l in open(os.environ["S1_FEATURES"] + "/probe_ucf_short_v1_f8/C/run-1/test/meta.jsonl")]
d = np.load(S / "pyes_recompute.npy", allow_pickle=True).item()
for c in "ABC":
    print("P(Yes)", c, {k: round(frame_metric(d[f"{c}/test/{k}"], meta, items, "frame_auc"), 3) for k in ("head_AB", "head_YesNo", "readout_AB")})
