"""Macro AUC per tier on UCF-Crime test 290 (Eric, 2026-11-10: "把macro auc也作為指標").

Macro AUC of a tier: for each anomaly video with an event of that tier, the window AUC of that tier's event
windows against the video's label-0 windows; then the mean over those videos. "all" uses every event window.
Tiers and windows follow scripts/layout-eval.py (tier_masks). Interval: paired bootstrap of the difference to the
first run, 1,000 draws of the tier's videos with replacement, seed 20261022.
Usage: macro_auc.py <manifest.json> <events.jsonl> <out.json> name=<scores.jsonl> ... (first = baseline)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
le = __import__("layout-eval")
from s1stream import metrics  # noqa: E402

DRAWS, SEED = 1000, 20261022
TIERS = ("small", "medium", "large", "all")


def per_video_auc(rows, scores, masks):
    """{tier: {video: AUC}} for videos that have both classes inside the tier."""
    videos = np.array([r["video"] for r in rows])
    out = {}
    for tier in TIERS:
        include, y = masks[tier]
        out[tier] = {}
        for v in np.unique(videos[include & (y == 1)]):
            m = include & (videos == v)
            if 0 < y[m].sum() < m.sum():
                out[tier][v] = metrics.auc(list(scores[m]), list(y[m]))
    return out


def load_scores(path, keys):
    by_key = {(r["video"], r["window"]): r["score"] for r in map(json.loads, Path(path).read_text().splitlines())}
    return np.array([by_key[k] for k in keys])


def main():
    manifest = {m["name"]: m for m in json.loads(Path(sys.argv[1]).read_text())}
    events = [json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines()]
    runs = dict(arg.split("=", 1) for arg in sys.argv[4:])
    names = list(runs)
    rows = sorted((json.loads(l) for l in Path(runs[names[0]]).read_text().splitlines()),
                  key=lambda r: (r["video"], r["window"]))
    keys = [(r["video"], r["window"]) for r in rows]
    masks = le.tier_masks(rows, manifest, events)
    aucs = {n: per_video_auc(rows, load_scores(p, keys), masks) for n, p in runs.items()}
    rng = np.random.default_rng(SEED)
    result = {}
    for tier in TIERS:
        vids = sorted(aucs[names[0]][tier])
        draws = [rng.integers(0, len(vids), len(vids)) for _ in range(DRAWS)]
        base = np.array([aucs[names[0]][tier][v] for v in vids])
        for n in names:
            vals = np.array([aucs[n][tier][v] for v in vids])
            diff = [vals[d].mean() - base[d].mean() for d in draws]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            result.setdefault(n, {})[tier] = {"macro_auc": float(vals.mean()), "videos": len(vids),
                                              "d": [float(lo), float(hi)]}
    for n in names:
        cells = "  ".join(f"{t}: {100 * result[n][t]['macro_auc']:.1f} d[{100 * result[n][t]['d'][0]:+.1f},"
                          f"{100 * result[n][t]['d'][1]:+.1f}] (n={result[n][t]['videos']})" for t in TIERS)
        print(f"{n:14s} {cells}", flush=True)
    Path(sys.argv[3]).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
