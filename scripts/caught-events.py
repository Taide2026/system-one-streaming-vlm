#!/usr/bin/env python3
"""Event-level view per tier (2026-11-06): for each run, how many events of each tier have at least one window with
score >= tau, and the mean of each event's max window score.  python caught-events.py <events.jsonl> <tau> <run> [...]"""
import json, sys
import numpy as np
from pathlib import Path

R = Path("results/pro6000")
ev = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
tau = float(sys.argv[2])
a = np.array([e["area_frac"] for e in ev]); t1, t2 = np.quantile(a, [1 / 3, 2 / 3])
for e in ev:
    e["tier"] = "small" if e["area_frac"] < t1 else ("medium" if e["area_frac"] < t2 else "large")


def latest(n):
    ds = sorted(d for d in (R / n).iterdir() if (d / "scores.jsonl").exists() or (d / "merged" / "scores.jsonl").exists())
    d = ds[-1]
    return d if (d / "scores.jsonl").exists() else d / "merged"


for n in sys.argv[3:]:
    d = latest(n)
    man = {m["name"]: m for m in json.loads((d / "manifest.json").read_text())}
    rows = {}
    for r in map(json.loads, (d / "scores.jsonl").read_text().splitlines()):
        if r:
            rows.setdefault(r["video"], []).append(r)
    out = {}
    for e in ev:
        m = man.get(e["video"])
        if m is None:
            continue
        fps = m["fps"]; e0, e1 = e["event"]
        ws = [r for r in rows.get(e["video"], []) if r["start_s"] * fps < e1 + 1 and r["end_s"] * fps > e0]
        if not ws:
            continue
        mx = max(r["score"] for r in ws)
        out.setdefault(e["tier"], []).append(mx)
    line = f"{n[:28]:28s}"
    for t in ("small", "medium", "large"):
        v = np.array(out.get(t, []))
        line += f"  {t}: {int((v >= tau).sum()):3d}/{len(v):3d} caught, max-score mean {v.mean():.2f}"
    print(line)
