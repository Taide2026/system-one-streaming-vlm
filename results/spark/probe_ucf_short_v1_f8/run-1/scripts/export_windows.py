"""Write windows.csv: one row per test window with label, P(Yes) of each condition (code-token reading) and the score
of each probe at its peak state (peak chosen on the test set, preview only). No threshold is applied."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
compare = json.loads((run / "preview-compare/compare.json").read_text())
pyes = json.loads((run / "pyes/pyes.json").read_text())["conditions"]
meta = [json.loads(line) for line in (run / "fit-C/test_meta.jsonl").read_text().splitlines()]
methods, kinds = compare["methods"], compare["kinds"]
columns, values = [], []
for c in "ABC":
    columns.append(f"pyes_{c}")
    values.append(pyes[c]["head_AB"]["p_yes"])
for c in "ABC":
    scores = np.load(run / f"fit-{c}/test_scores.npy", mmap_mode="r")
    for mi, m in enumerate(methods):
        for ki, k in enumerate(kinds):
            state = int(np.nanargmax(compare["metrics"][f"{c}/{m}/{k}/frame_auc"]["value"]))
            columns.append(f"{c}_{m}_{k}_s{state}")
            values.append(np.asarray(scores[mi, ki, state], dtype=float))
with open(run / "windows.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["i", "video", "category", "window", "start_s", "end_s", "label", *columns])
    for i, row in enumerate(meta):
        w.writerow([row["i"], row["video"], row["category"], row["window"], row["start_s"], row["end_s"], row["label"],
                    *(round(float(v[i]), 5) for v in values)])
print(len(meta), "rows,", 7 + len(columns), "columns")
