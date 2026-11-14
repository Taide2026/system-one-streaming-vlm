#!/usr/bin/env python3
"""Alarm-level view (2026-11-06): event recall per tier at the threshold that gives a target false-alarm rate on the
normal test videos (an alarm = a window with score >= tau; FA/h counted on normal videos only).
  python alarm-metric.py <events.jsonl> <fa_per_hour,...> <run> [<run> ...]"""
import json, sys
import numpy as np
from pathlib import Path

R = Path("results/pro6000")
ev = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
targets = [float(x) for x in sys.argv[2].split(",")]
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
    normal_scores, normal_hours = [], 0.0
    for v, rs in rows.items():
        if not man[v].get("events"):
            normal_scores += [r["score"] for r in rs]; normal_hours += sum(r["end_s"] - r["start_s"] for r in rs) / 3600
    normal_scores = np.sort(np.array(normal_scores))[::-1]
    ev_max = []
    for e in ev:
        m = man.get(e["video"])
        if m is None:
            continue
        fps = m["fps"]; e0, e1 = e["event"]
        ws = [r["score"] for r in rows.get(e["video"], []) if r["start_s"] * fps < e1 + 1 and r["end_s"] * fps > e0]
        if ws:
            ev_max.append((e["tier"], max(ws)))
    line = f"{n[:26]:26s} normal {normal_hours:.1f} h"
    for fa in targets:
        k = int(round(fa * normal_hours))
        tau = normal_scores[k - 1] if k >= 1 else normal_scores[0] + 1e-9  # k alarms allowed on normal video
        line += f" | FA {fa:g}/h tau {tau:.2f}:"
        for t in ("small", "medium", "large"):
            v = [mx for tt, mx in ev_max if tt == t]
            line += f" {t[0]} {sum(1 for x in v if x >= tau)}/{len(v)}"
    print(line)
