#!/usr/bin/env python3
"""Small-tier AP per UCF class for several runs (2026-11-06): positives = small-tier windows of that class, negatives =
every negative window. python perclass-small.py <events.jsonl> <run name> [<run name> ...]  (run dirs under results/pro6000)"""
import json, sys
import numpy as np
from pathlib import Path

import os
TIER = os.environ.get("TIER", "small")
R = Path("results/pro6000")
ev = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
a = np.array([e["area_frac"] for e in ev]); t1, t2 = np.quantile(a, [1 / 3, 2 / 3])
tier = {}
for e in ev:
    t = "small" if e["area_frac"] < t1 else ("medium" if e["area_frac"] < t2 else "large")
    tier.setdefault(e["video"], []).append((e["event"][0], e["event"][1], t))


def latest(n):
    ds = sorted(d for d in (R / n).iterdir() if (d / "scores.jsonl").exists() or (d / "merged" / "scores.jsonl").exists())
    d = ds[-1]
    return d if (d / "scores.jsonl").exists() else d / "merged"


def ap(y, s):
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


names = sys.argv[2:]
runs = {n: latest(n) for n in names}
man = {m["name"]: m for m in json.loads((runs[names[0]] / "manifest.json").read_text())}
sc = {n: {(r["video"], r["window"]): r for r in map(json.loads, (d / "scores.jsonl").read_text().splitlines()) if r}
      for n, d in runs.items()}
keys = sorted(set.intersection(*[set(v) for v in sc.values()]))
neg = [k for k in keys if not sc[names[0]][k]["label"]]
cls = {}
for k in keys:
    r = sc[names[0]][k]; m = man[r["video"]]
    if not r["label"] or not m.get("events"):
        continue
    fps = m["fps"]; w0, w1 = r["start_s"] * fps, r["end_s"] * fps
    if any(w0 < e1 + 1 and w1 > e0 and t == TIER for e0, e1, t in tier.get(r["video"], [])):
        cls.setdefault(m["category"], []).append(k)
print("class            n+  " + "  ".join(f"{n[:12]:>12s}" for n in names))
for c, ks in sorted(cls.items(), key=lambda x: -len(x[1])):
    line = f"{c:14s} {len(ks):5d}"
    for n in names:
        y = np.array([1] * len(ks) + [0] * len(neg), float)
        s = np.array([sc[n][k]["score"] for k in ks] + [sc[n][k]["score"] for k in neg])
        line += f"  {100 * ap(y, s):12.1f}"
    print(line)
