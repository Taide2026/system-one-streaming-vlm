#!/usr/bin/env python3
"""Event-weighted two-group table (2026-11-12): per-event AUC (an event's windows against all negative windows),
averaged over the events below / at-or-above an area cut; every event counts once. Numpy only.
  python scripts/twogroup.py --events results/pro6000/jev-hard-events.jsonl [--cut 0.04]"""
import argparse, glob, json
from collections import defaultdict
import numpy as np
p = argparse.ArgumentParser(); p.add_argument("--events", required=True); p.add_argument("--cut", type=float, default=0.04); a_ = p.parse_args()
key = lambda n: n[:-4] if n.endswith(".mp4") else n
E = [json.loads(l) for l in open(a_.events)]; area = np.array([e["area_frac"] for e in E])
ev = defaultdict(list)
for i, e in enumerate(E): ev[key(e["video"])].append((e["event"][0], e["event"][1], i))
def per_event(name):
    d = sorted(x for x in glob.glob(f"results/nano4/{name}/*/") if glob.glob(x + "run.json"))[-1]
    R = [json.loads(l) for l in open(d + "scores.jsonl")]
    neg = np.sort([r["score"] for r in R if r["label"] == 0]); by = defaultdict(list)
    for r in R:
        if r["label"] == 1:
            lo, hi = r["start_s"] * 30, r["end_s"] * 30 - 1
            j = next((j for s, e, j in ev.get(key(r["video"]), []) if lo <= e and s <= hi), None)
            if j is not None: by[j].append(r["score"])
    out = np.full(len(E), np.nan)
    for j, s in by.items():
        s = np.array(s); out[j] = 100 * (np.searchsorted(neg, s, "left") + np.searchsorted(neg, s, "right")).sum() / 2 / (len(s) * len(neg))
    return out
sm, lg = area < a_.cut, area >= a_.cut
print(f"cut {100 * a_.cut:.1f}%: events below {int(sm.sum())}, at or above {int(lg.sum())}")
rows = [("orig", "frames", "floor_s{}_frames"), ("orig", "f16", "floor_s{}_f16"), ("orig", "mix", "floor_s{}_full2x"), ("orig", "all2x", "floor_s{}_all2x"),
        ("mix", "frames", "fovfull2x_s{}_frames"), ("mix", "mix", "fovfull2x_s{}_full2x"),
        ("all2x", "frames", "pure2x_s{}_frames"), ("all2x", "all2x", "pure2x_s{}_all2x"), ("crop3.6", "crop", "fov2g036_s{}")]
P = {}
for tr, te, pat in rows:
    pe = np.array([per_event("up290_" + pat.format(k)) for k in range(3)]); P[(tr, te)] = pe
    s_, l_, al = np.nanmean(pe[:, sm], 1), np.nanmean(pe[:, lg], 1), np.nanmean(pe, 1)
    print(f"{tr:8s} -> {te:7s} small {s_.mean():.1f}+-{s_.std(ddof=1):.1f}  large {l_.mean():.1f}+-{l_.std(ddof=1):.1f}  all events {al.mean():.1f}+-{al.std(ddof=1):.1f}   per seed small {np.round(s_, 1)} large {np.round(l_, 1)}")
rng = np.random.default_rng(0)
def paired(a, b, label):
    d = P[b] - P[a]; out = []
    for m, n in ((sm, "small"), (lg, "large")):
        idx = np.where(m)[0]; pt = np.nanmean(d[:, idx], 1)
        dm = np.nanmean(d, 0); bs = [np.nanmean(dm[rng.choice(idx, len(idx))]) for _ in range(4000)]
        out.append(f"{n} {np.nanmean(dm[idx]):+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] per seed {np.round(pt, 2)} improved {100 * np.mean(dm[idx] > 0):.0f}%")
    print(f"paired {label}: " + " | ".join(out))
paired(("orig", "frames"), ("orig", "mix"), "orig model, frames -> mix")
paired(("orig", "frames"), ("orig", "all2x"), "orig model, frames -> all2x")
paired(("orig", "frames"), ("orig", "f16"), "orig model, frames -> 16 frames")
paired(("mix", "frames"), ("mix", "mix"), "mix model, frames -> mix")
paired(("all2x", "frames"), ("all2x", "all2x"), "2x model, frames -> all2x")
paired(("orig", "all2x"), ("all2x", "all2x"), "training effect at all2x (orig -> 2x model)")
paired(("orig", "frames"), ("all2x", "frames"), "training effect at frames (orig -> 2x model)")
