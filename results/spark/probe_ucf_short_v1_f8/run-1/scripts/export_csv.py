"""Flatten preview-compare/compare.json into curves.csv (one row per condition, probe, feature, state)."""
import csv
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
r = json.loads((run / "preview-compare/compare.json").read_text())
with open(run / "curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "probe", "feature", "state", "frame_auc", "frame_auc_low", "frame_auc_high",
                "frame_ap", "window_auc", "cka_vs_B"])
    for c in r["conditions"]:
        for m in r["methods"]:
            for k in r["kinds"]:
                e = {x: r["metrics"][f"{c}/{m}/{k}/{x}"] for x in ("frame_auc", "frame_ap", "window_auc")}
                cka = r["cka"].get(f"{c}-B/{k}")
                for s in range(r["n_states"]):
                    w.writerow([c, m, k, s, *(round(v, 4) for v in (e["frame_auc"]["value"][s], e["frame_auc"]["low"][s],
                                                                   e["frame_auc"]["high"][s], e["frame_ap"]["value"][s],
                                                                   e["window_auc"]["value"][s])),
                                "" if cka is None else round(cka[s], 4)])
print(sum(1 for _ in open(run / "curves.csv")) - 1, "rows")
