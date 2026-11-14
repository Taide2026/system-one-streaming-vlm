#!/usr/bin/env python3
"""Event-weighted and within-video metrics of named runs (2026-11-12): python scripts/evalrun.py --events E name ...
name = results/nano4/<name>/<latest finished>/. Prints per-event AUC (<4% / >=4%), Macro-AUC, Within-AUC, all AP."""
import argparse, glob, json
from collections import defaultdict
import numpy as np
p = argparse.ArgumentParser(); p.add_argument("--events", required=True); p.add_argument("--cut", type=float, default=0.04); p.add_argument("--videos-from"); p.add_argument("names", nargs="+"); a_ = p.parse_args()
key = lambda n: n[:-4] if n.endswith(".mp4") else n
E = [json.loads(l) for l in open(a_.events)]; area = np.array([e["area_frac"] for e in E]); ev = defaultdict(list)
for i, e in enumerate(E): ev[key(e["video"])].append((e["event"][0], e["event"][1], i))
def U(pos, neg): neg = np.sort(neg); return (np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right")).sum() / 2
def ap(y, s): o = np.argsort(-s, kind="stable"); y = y[o]; return 100 * (np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / y.sum()
keepv = None
if a_.videos_from:
    d0 = sorted(x for x in glob.glob(f"results/nano4/{a_.videos_from}/*/") if glob.glob(x + "run.json"))[-1]
    keepv = {key(json.loads(l)["video"]) for l in open(d0 + "scores.jsonl")}; print("restricted to", len(keepv), "videos of", a_.videos_from)
for name in a_.names:
    ds = sorted(x for x in glob.glob(f"results/nano4/{name}/*/") if glob.glob(x + "run.json"))
    if not ds: print(f"{name:34s} (not finished)"); continue
    R = [json.loads(l) for l in open(ds[-1] + "scores.jsonl")]
    if keepv is not None: R = [r for r in R if key(r["video"]) in keepv]
    tk = [r["tokens"] for r in R if "tokens" in r]
    y = np.array([r["label"] for r in R]); s = np.array([r["score"] for r in R]); vid = np.array([key(r["video"]) for r in R])
    neg = s[y == 0]; pe = np.full(len(E), np.nan); by = defaultdict(list)
    for r, sc in zip(R, s):
        if r["label"] == 1:
            lo, hi = r["start_s"] * 30, r["end_s"] * 30 - 1
            j = next((j for b, e, j in ev.get(key(r["video"]), []) if lo <= e and b <= hi), None)
            if j is not None: by[j].append(sc)
    for j, v in by.items(): pe[j] = 100 * U(np.array(v), neg) / (len(v) * len(neg))
    mac, uin, nin = [], 0.0, 0.0
    for v in np.unique(vid[y == 1]):
        m = vid == v; pp, nn = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(pp) and len(nn): u = U(pp, nn); mac.append(u / (len(pp) * len(nn))); uin += u; nin += len(pp) * len(nn)
    print((f"tokens/window {np.mean(tk):6.0f}  " if tk else "") + f"{name:34s} small {np.nanmean(pe[area < a_.cut]):.2f}  large {np.nanmean(pe[area >= a_.cut]):.2f}  macro {100 * np.mean(mac):.2f}  within {100 * uin / nin:.2f}  allAP {ap(y, s):.2f}  mean logit normal-video {np.mean(np.log(np.clip(s[~np.isin(vid, np.unique(vid[y == 1]))], 1e-6, 1 - 1e-6) / (1 - np.clip(s[~np.isin(vid, np.unique(vid[y == 1]))], 1e-6, 1 - 1e-6)))):+.2f}")
