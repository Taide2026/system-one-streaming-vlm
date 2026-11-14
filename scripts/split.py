#!/usr/bin/env python3
"""Granularity case 3 (deck p. 20): a short event split over two windows, each window seeing part of it (2026-11-14).
CPU only, from existing score files. Events are grouped by how the 2 s window grid cuts them:
  split   shorter than one window (<= 2 s) and lying across a window boundary: two windows, neither full
  inside  shorter than one window and inside a single window
  nofull  any length, but no window lies fully inside the event (includes the two above)
  full    at least one window lies fully inside the event
Reported per input: share of events caught at A false alarms per hour of normal video (threshold per input, as
stream.py), mean over seeds.   python scripts/split.py --events E
"""
import argparse, glob, json
from collections import defaultdict
import numpy as np
p = argparse.ArgumentParser(); p.add_argument("--events", required=True); a = p.parse_args()
key = lambda n: n[:-4] if n.endswith(".mp4") else n
E = [json.loads(l) for l in open(a.events)]
def load(name):
    ds = sorted(x for x in glob.glob(f"results/nano4/{name}/*/") if glob.glob(x + "run.json"))
    return [json.loads(l) for l in open(ds[-1] + "scores.jsonl")]
NAME = {"0.5x": "dn05", "1x": "frames", "1.5x": "1p5", "2x": "all2x"}
runs = {t: [load(f"up290_floor_s{s}_{n}") for s in range(3)] for t, n in NAME.items()}
R0 = runs["1x"][0]; wins = defaultdict(list)
for r in R0: wins[key(r["video"])].append((r["start_s"], r["end_s"]))
grp = {}
for i, e in enumerate(E):
    b, en = e["event"]; d = (en - b + 1) / 30; ws = [(s, t) for s, t in wins[key(e["video"])] if s * 30 <= en and b <= t * 30 - 1]
    full = [1 for s, t in ws if b <= s * 30 and t * 30 - 1 <= en]
    g = []
    if d <= 2.0 and len(ws) == 2: g.append("split")
    if d <= 2.0 and len(ws) == 1: g.append("inside")
    g.append("full" if full else "nofull")
    grp[i] = (g, d, len(ws), e["area_frac"])
names = ["split", "inside", "nofull", "full"]
idx = {n: [i for i, (g, *_r) in grp.items() if n in g] for n in names}
print("events:", len(E), {n: len(v) for n, v in idx.items()}, "| small (<4%) among them:", {n: sum(grp[i][3] < 0.04 for i in v) for n, v in idx.items()})
print("durations of split events (s):", sorted(round(grp[i][1], 1) for i in idx["split"]))
def caught(R, A):
    pos = {key(r["video"]) for r in R if r["label"] == 1}
    normal = np.sort([r["score"] for r in R if key(r["video"]) not in pos])[::-1]
    tau = normal[min(int(round(A * len(normal) * 2 / 3600)), len(normal) - 1)]
    best = defaultdict(float)
    for r in R:
        if r["label"] != 1: continue
        lo, hi = r["start_s"] * 30, r["end_s"] * 30 - 1
        for i, e in enumerate(E):
            if key(e["video"]) == key(r["video"]) and lo <= e["event"][1] and e["event"][0] <= hi:
                best[i] = max(best[i], r["score"])
    return {i: best[i] > tau for i in range(len(E))}, best
for A in (2, 6):
    print(f"\ncaught at {A} false alarms / hour (%), mean of three seeds")
    print(f"{'group':8s} {'n':>4s} " + " ".join(f"{t:>7s}" for t in NAME))
    res = {t: [caught(R, A)[0] for R in runs[t]] for t in NAME}
    for n in names:
        if not idx[n]: continue
        print(f"{n:8s} {len(idx[n]):4d} " + " ".join(f"{100 * np.mean([np.mean([c[i] for i in idx[n]]) for c in res[t]]):7.1f}" for t in NAME))
    for n in ("nofull", "full"):
        sm = [i for i in idx[n] if grp[i][3] < 0.04]
        if sm: print(f"{n+' <4%':10s} {len(sm):2d} " + " ".join(f"{100 * np.mean([np.mean([c[i] for i in sm]) for c in res[t]]):7.1f}" for t in NAME))
    # paired bootstrap over events, 2x - 1x, nofull group
    for n in ("split", "nofull", "full"):
        if len(idx[n]) < 5: continue
        d = np.array([np.mean([res["2x"][s][i] for s in range(3)]) - np.mean([res["1x"][s][i] for s in range(3)]) for i in idx[n]])
        rng = np.random.default_rng(0); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        print(f"  2x - 1x, {n}: {100*d.mean():+.1f} +- {100*(np.percentile(bs,97.5)-np.percentile(bs,2.5))/2:.1f} (95% half-width, events)")
print("SPLIT_DONE")
