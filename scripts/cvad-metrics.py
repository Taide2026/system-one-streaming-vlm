#!/usr/bin/env python3
"""Metrics of ComplexVAD runs (2026-11-13): python scripts/cvad-metrics.py <name> ...   (results/nano4/<name>/<latest finished>/)
Every video has an anomaly, so: Macro-AUC (mean per-video AUC over videos with both labels), pooled AUC, pooled AP,
and a paired bootstrap over videos of the Macro-AUC difference to the first run named."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np


def auc(pos, neg):
    neg = np.sort(neg)
    return (np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right")).sum() / 2 / (len(pos) * len(neg))


def ap(y, s):
    o = np.argsort(-s, kind="stable"); y = y[o]
    return (np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / y.sum()


keep = None
args_ = sys.argv[1:]
if args_ and args_[0] == "--videos":          # restrict to the ids listed in a file (e.g. the held-out list)
    keep = set(open(args_[1]).read().split()); args_ = args_[2:]
base = None
for name in args_:
    ds = sorted(x for x in glob.glob(f"results/nano4/{name}/*/") if glob.glob(x + "run.json"))
    if not ds:
        print(f"{name:28s} (not finished)"); continue
    R = [json.loads(l) for l in open(ds[-1] + "scores.jsonl")]
    if keep is not None:
        R = [r for r in R if r["video"] in keep]
    y = np.array([r["label"] for r in R]); s = np.array([r["score"] for r in R]); m = np.array([r.get("margin", 0.0) for r in R])
    by = defaultdict(lambda: ([], []))
    for r in R:
        by[r["video"]][r["label"]].append(r["score"])
    per = {v: auc(np.array(p), np.array(n)) for v, (n, p) in by.items() if p and n}
    tk = [r["tokens"] for r in R if "tokens" in r]
    line = (f"{name:28s} windows {len(R):5d} pos {100 * y.mean():4.1f}%  videos {len(per):3d}  macro {100 * np.mean(list(per.values())):5.2f}  "
            f"pooled AUC {100 * auc(s[y == 1], s[y == 0]):5.2f}  AP {100 * ap(y, s):5.2f}" + (f"  tokens {np.mean(tk):5.0f}" if tk else ""))
    if base is None:
        base = per
    else:
        vs = sorted(set(per) & set(base)); d = np.array([per[v] - base[v] for v in vs]); rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        line += f"  vs first {100 * d.mean():+5.2f} [{100 * np.percentile(bs, 2.5):+.2f}, {100 * np.percentile(bs, 97.5):+.2f}] better in {int((d > 0).sum())}/{len(d)}"
    print(line)
