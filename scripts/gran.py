#!/usr/bin/env python3
"""Temporal granularity x input resolution (2026-11-13). CPU only, from existing score files.

A 2 s window gets one score from 8 frames sampled at (k + 0.5) / 4 s. A positive window may contain the event in all
8 sampled frames or in one; an event may also be small in the frame. Both reduce the number of visual tokens that
fall on the event. For every positive window of UCF-Crime: n = how many of its 8 sampled frames lie inside an
annotated event (0..8), and the event's size group (< 4% / >= 4% of the frame).
Per group: AUC of its windows against ALL negative windows, per input, mean over the runs given.

  python scripts/gran.py --events E --inputs 1x=a,b,c 2x=d,e,f ...     (names under results/nano4/, latest finished run)
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

p = argparse.ArgumentParser(); p.add_argument("--events", required=True); p.add_argument("--inputs", nargs="+", required=True)
p.add_argument("--cut", type=float, default=0.04); p.add_argument("--fps", type=float, default=30.0); p.add_argument("--out")
a = p.parse_args()
key = lambda n: n[:-4] if n.endswith(".mp4") else n
ev = defaultdict(list)
for l in open(a.events):
    e = json.loads(l); ev[key(e["video"])].append((e["event"][0], e["event"][1], e["area_frac"]))


def U(pos, neg):
    neg = np.sort(neg)
    return (np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right")).sum() / 2 / (len(pos) * len(neg))


def load(name):
    ds = sorted(x for x in glob.glob(f"results/nano4/{name}/*/") if glob.glob(x + "run.json"))
    return {(key(r["video"]), r["window"]): r for r in map(json.loads, open(ds[-1] + "scores.jsonl"))}


def describe(r):
    """(sampled frames inside an event, frames 0 or 7 inside, small?) of a positive window."""
    v = key(r["video"]); t0 = r["start_s"]; L = r["end_s"] - r["start_s"]
    fr = [round((t0 + (k + 0.5) * L / 8) * a.fps) for k in range(8)]
    lo, hi = t0 * a.fps, r["end_s"] * a.fps - 1
    hit = [(b, e, ar) for b, e, ar in ev.get(v, []) if lo <= e and b <= hi]
    if not hit:
        return None
    inside = [any(b <= f <= e for b, e, _ in hit) for f in fr]
    return sum(inside), inside[0] or inside[7], min(ar for _, _, ar in hit) < a.cut


runs = {}
for spec in a.inputs:
    tag, names = spec.split("="); runs[tag] = [load(n) for n in names.split(",")]
ref = next(iter(runs.values()))[0]
info = {k: describe(r) for k, r in ref.items() if r["label"] == 1}
info = {k: v for k, v in info.items() if v is not None}
groups = {"0 frames (event between samples)": lambda n, kf, sm: n == 0, "1-2 frames": lambda n, kf, sm: 1 <= n <= 2, "3-5 frames": lambda n, kf, sm: 3 <= n <= 5,
          "6-7 frames": lambda n, kf, sm: 6 <= n <= 7, "all 8 frames": lambda n, kf, sm: n == 8,
          "partial (1-7), small event": lambda n, kf, sm: 1 <= n <= 7 and sm, "partial (1-7), large event": lambda n, kf, sm: 1 <= n <= 7 and not sm,
          "full (8), small event": lambda n, kf, sm: n == 8 and sm, "full (8), large event": lambda n, kf, sm: n == 8 and not sm,
          "partial, event only in frames 1-6": lambda n, kf, sm: 1 <= n <= 7 and not kf, "partial, event reaches frame 0 or 7": lambda n, kf, sm: 1 <= n <= 7 and kf}
print(f"{len(info)} positive windows; share by sampled frames inside the event: " + "  ".join(f"{n}:{100 * np.mean([v[0] == n for v in info.values()]):.1f}%" for n in range(9)))
print(f"{'group':38s} {'n':>5s}  " + "  ".join(f"{t:>9s}" for t in runs))
rows = {}
for g, f in groups.items():
    ks = [k for k, v in info.items() if f(*v)]
    vals = []
    for tag, rs in runs.items():
        aucs = []
        for R in rs:
            neg = np.array([r["score"] for r in R.values() if r["label"] == 0]); pos = np.array([R[k]["score"] for k in ks if k in R])
            aucs.append(100 * U(pos, neg) if len(pos) else np.nan)
        vals.append((float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0))
    rows[g] = {"n": len(ks), **{t: v for t, v in zip(runs, vals)}}
    print(f"{g:38s} {len(ks):5d}  " + "  ".join(f"{m:5.1f}±{s:3.1f}" for m, s in vals))
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
print("GRAN_DONE")
