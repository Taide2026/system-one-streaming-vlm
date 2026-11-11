"""Why does 2x help small events most? (Eric, 2026-11-10). CPU only, existing scores.

For each model: the 1x -> 2x score change d = s2 - s1 of event windows per tier, raw and within bins of the 1x score.
"Adjusted" difference A - B = sum over 1x bins of w_b * (mean d of A in bin b - mean d of B in bin b), w_b = share of
A+B windows in bins where both groups have windows. If small minus large stays > 0 after the adjustment, small events
gain more than large ones at the same starting score (not only headroom). 95% interval: 1,000 draws of anomaly
videos with replacement, seed 20261022 (negatives of normal videos are not used).
Usage: why_small.py <manifest.json> <events.jsonl> name=<1x scores>,<2x scores> ...
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
le = __import__("layout-eval")

BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
DRAWS, SEED = 1000, 20261022


def load(path):
    return {(r["video"], r["window"]): r for r in map(json.loads, Path(path).read_text().splitlines())}


def groups(rows, manifest, events):
    """Per window: 'small' / 'medium' / 'large' (event tier), 'anomaly negative', or '' (normal video)."""
    out = np.array([""] * len(rows), dtype=object)
    for tier, (include, y) in le.tier_masks(rows, manifest, events).items():
        if tier != "all":
            out[include & (y == 1)] = tier
    for i, r in enumerate(rows):
        if r["label"] == 0 and manifest[r["video"]]["events"]:
            out[i] = "anomaly negative"
    return out


def adjusted(d, s1, g, a, b, idx):
    """Bin-adjusted mean difference of d between groups a and b over the window indices idx."""
    total, weight = 0.0, 0
    bins = np.digitize(s1[idx], BINS) - 1
    for k in range(len(BINS) - 1):
        in_bin = idx[bins == k]
        da, db = d[in_bin][g[in_bin] == a], d[in_bin][g[in_bin] == b]
        if len(da) and len(db):
            n = len(da) + len(db)
            total += n * (da.mean() - db.mean())
            weight += n
    return total / weight if weight else np.nan


def main():
    manifest = {m["name"]: m for m in json.loads(Path(sys.argv[1]).read_text())}
    events = [json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines()]
    pairs = [("small", "large"), ("small", "medium"), ("medium", "large"), ("small", "anomaly negative"),
             ("large", "anomaly negative")]
    result = {}
    for spec in sys.argv[3:]:
        name, paths = spec.split("=")
        one, two = (load(p) for p in paths.split(","))
        keys = sorted(one)
        rows = [one[k] for k in keys]
        s1 = np.array([one[k]["score"] for k in keys])
        d = np.array([two[k]["score"] for k in keys]) - s1
        g = groups(rows, manifest, events)
        print(f"== {name}")
        for tier in ("small", "medium", "large", "anomaly negative"):
            m = g == tier
            print(f"  {tier:16s} n={int(m.sum()):5d}  1x mean {s1[m].mean():.3f}  d mean {d[m].mean():+.3f}")
        videos = sorted({r["video"] for r in rows if manifest[r["video"]]["events"]})
        by_video = {v: [] for v in videos}
        for i, r in enumerate(rows):
            if r["video"] in by_video:
                by_video[r["video"]].append(i)
        rng = np.random.default_rng(SEED)
        draws = [np.concatenate([by_video[v] for v in rng.choice(videos, len(videos))]) for _ in range(DRAWS)]
        all_idx = np.concatenate([by_video[v] for v in videos])
        result[name] = {}
        for a, b in pairs:
            raw = d[g == a].mean() - d[g == b].mean()
            adj = adjusted(d, s1, g, a, b, all_idx)
            boot = np.array([adjusted(d, s1, g, a, b, idx) for idx in draws])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            result[name][f"{a} - {b}"] = {"raw": raw, "adjusted": adj, "ci": [lo, hi]}
            print(f"  {a:6s} - {b:16s} raw {raw:+.3f}  adjusted {adj:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    json.dump(result, open(Path(__file__).parent / "why_small.json", "w"), indent=1)


if __name__ == "__main__":
    main()
