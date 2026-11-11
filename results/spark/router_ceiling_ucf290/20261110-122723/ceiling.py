"""AP all of a simulated 1x/2x router on mix + KL (test 290, window level).

The router up-scales each event window with probability TPR and each negative window with probability FPR;
the window then takes its 2x score, else its 1x score. AP is the mean over 20 random draws.
Usage: ceiling.py <1x scores.jsonl> <2x scores.jsonl> <out.json>
"""
import json
import sys

import numpy as np

from s1stream import metrics

GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DRAWS = 20


def load(path):
    """{(video, window): (label, score)}"""
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["video"], r["window"])] = (r["label"], r["score"])
    return out


def mean_ap(s1, s2, y, tpr, fpr, rng):
    """Mean AP over DRAWS random routers with the given TPR and FPR."""
    p = np.where(y == 1, tpr, fpr)
    aps = []
    for _ in range(DRAWS):
        zoom = rng.random(len(y)) < p
        aps.append(metrics.ap(np.where(zoom, s2, s1), y))
    return float(np.mean(aps))


def main():
    one, two = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(one)
    y = np.array([one[k][0] for k in keys])
    s1 = np.array([one[k][1] for k in keys])
    s2 = np.array([two[k][1] for k in keys])
    rng = np.random.default_rng(0)
    print(f"windows {len(y)}, events {int(y.sum())}")
    print(f"check: 1x {metrics.ap(s1, y):.4f}  all 2x {metrics.ap(s2, y):.4f}  oracle {metrics.ap(np.where(y == 1, s2, s1), y):.4f}")
    table = {f"{tpr:.1f}": {f"{fpr:.1f}": mean_ap(s1, s2, y, tpr, fpr, rng) for fpr in GRID} for tpr in GRID}
    print("rows = TPR, columns = FPR, AP all %")
    print("TPR\\FPR " + " ".join(f"{f:5.1f}" for f in GRID))
    for tpr, row in table.items():
        print(f"{tpr:7s} " + " ".join(f"{100 * v:5.1f}" for v in row.values()))
    json.dump({"windows": len(y), "events": int(y.sum()), "ap": table}, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
