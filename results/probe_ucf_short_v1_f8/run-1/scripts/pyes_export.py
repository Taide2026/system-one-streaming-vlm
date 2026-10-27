"""Write pyes/pyes.json: test P(Yes) per condition and reading (per window) with frame AUC and window AUC."""
import os
import json
import sys
from pathlib import Path

import numpy as np

from s1stream.probe.compare import frame_metric
from s1stream.probe.extract import load_probe_config
from s1stream.registry import DATASETS
import s1stream.plugins  # noqa: F401

source, out = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(source, allow_pickle=True).item()
items = {i.name: i for i in DATASETS.build(load_probe_config(Path("configs/probe_ucf_short_v1_f8.yaml")).test).items()}
meta = [json.loads(l) for l in open(os.environ["S1_FEATURES"] + "/probe_ucf_short_v1_f8/C/run-1/test/meta.jsonl")]
result = {"note": "P(Yes) recomputed from state 32 of last.npy (float16). head_* = base lm_head rows, readout_AB = d3-flash "
                  "readout rows 0-1. stored = meta.jsonl p_yes. Test set only.", "windows": len(meta), "conditions": {}}
for c in "ABC":
    result["conditions"][c] = {k: {"frame_auc": frame_metric(d[f"{c}/test/{k}"], meta, items, "frame_auc"),
                                   "window_auc": frame_metric(d[f"{c}/test/{k}"], meta, items, "window_auc"),
                                   "p_yes": [round(float(x), 6) for x in d[f"{c}/test/{k}"]]}
                               for k in ("stored", "head_AB", "head_YesNo", "readout_AB")}
out.write_text(json.dumps(result))
print({c: {k: round(v["frame_auc"], 3) for k, v in r.items()} for c, r in result["conditions"].items()})
