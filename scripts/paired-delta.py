#!/usr/bin/env python3
"""Paired effect of a test input (2026-11-11): the same adapter scored on two inputs, window by window.

  python scripts/paired-delta.py --events results/pro6000/jev-hard-events.jsonl <run A (base)> <run B (changed)>

(1) mean change of the logit, B - A, per window group, with a 95% bootstrap interval over videos; and the
    difference-in-differences against normal-video windows (a uniform shift of all scores cancels).
(2) paired bootstrap over videos of the change in tier AP (tier positives against all negatives), all-AP and Macro-AUC.
UCF-Crime is 30 fps; events are in frames. Numpy only.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ap_ = argparse.ArgumentParser(); ap_.add_argument("--events", required=True); ap_.add_argument("--n", type=int, default=1000)
ap_.add_argument("a"); ap_.add_argument("b"); args = ap_.parse_args()
key = lambda n: n[:-4] if n.endswith(".mp4") else n
events = [json.loads(l) for l in open(args.events)]
areas = sorted(e["area_frac"] for e in events); t1, t2 = areas[len(areas) // 3], areas[2 * len(areas) // 3]
ev = defaultdict(list)
for e in events:
    ev[key(e["video"])].append((e["event"][0], e["event"][1], "small" if e["area_frac"] < t1 else "medium" if e["area_frac"] < t2 else "large"))
load = lambda p: {(key(r["video"]), r["window"]): r for r in map(json.loads, open(Path(p) / "scores.jsonl"))}
A, B = load(args.a), load(args.b)
assert A.keys() == B.keys(), "runs cover different windows"
per = defaultdict(list)
for k in A: per[k[0]].append(k)
G = ["small+", "medium+", "large+", "near-", "far-", "normal-"]
vids, rows = sorted(per), {}
for v in vids:
    ks = sorted(per[v], key=lambda k: k[1]); pos = [k[1] for k in ks if A[k]["label"] == 1]; out = []
    for k in ks:
        r = A[k]
        if r["label"] == 1:
            a, b = r["start_s"] * 30, r["end_s"] * 30 - 1
            g = next((t + "+" for s, e, t in ev.get(v, []) if a <= e and s <= b), None)
        elif not pos: g = "normal-"
        else: g = "near-" if min(abs(k[1] - p) for p in pos) <= 5 else "far-"
        out.append((G.index(g) if g else -1, r["label"], A[k]["score"], B[k]["score"]))
    rows[v] = np.array(out, float)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def ap(y, s):
    o = np.argsort(-s, kind="stable"); y = y[o]; return (np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / max(y.sum(), 1)
def auc(p, n):
    n = np.sort(n); return (np.searchsorted(n, p, "left") + np.searchsorted(n, p, "right")).sum() / 2 / (len(p) * len(n))
def stats(sel):
    M = np.concatenate([rows[v] for v in sel]); g, y, a, b = M.T; d = lg(b) - lg(a); out = {}
    for i, name in enumerate(G): out["dlogit " + name] = d[g == i].mean() if (g == i).any() else np.nan
    for name in G[:5]: out["DiD " + name] = out["dlogit " + name] - out["dlogit normal-"]
    neg = y == 0
    for i, name in enumerate(G[:3]):
        m = neg | (g == i); out["dAP " + name] = 100 * (ap(y[m], b[m]) - ap(y[m], a[m]))
    out["dAP all"] = 100 * (ap(y, b) - ap(y, a))
    mac = [(auc(r[r[:, 1] == 1, 3], r[r[:, 1] == 0, 3]) - auc(r[r[:, 1] == 1, 2], r[r[:, 1] == 0, 2])) for r in (rows[v] for v in sel) if 0 < r[:, 1].sum() < len(r)]
    out["dMacroAUC"] = 100 * float(np.mean(mac)); return out
point = stats(vids); rng = np.random.default_rng(0)
boot = [stats(list(rng.choice(vids, len(vids)))) for _ in range(args.n)]
print(f"{Path(args.a).parent.name} -> {Path(args.b).parent.name}")
for k, v in point.items():
    xs = np.array([b[k] for b in boot]); xs = xs[~np.isnan(xs)]; lo, hi = np.percentile(xs, [2.5, 97.5])
    print(f"  {k:16s} {v:+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {'*' if lo > 0 or hi < 0 else ''}")
