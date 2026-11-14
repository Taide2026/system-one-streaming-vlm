#!/usr/bin/env python3
"""Paired bootstrap over test videos for jev-hard tier AP / AUC differences between runs (2026-11-05).

  python tier-bootstrap.py --events results/pro6000/jev-hard-events.jsonl --base <run dir> <run dir> [...] [--n 2000]

Same tier construction as scripts/jev-hard.py (positives = windows of the tier's events, negatives = every negative
window of the test set). Resamples the 290 videos with replacement, recomputes each tier's AUC / AP for the base and
for every other run on the same resample, and reports the mean difference with a 95% percentile interval and the
fraction of resamples where the difference is <= 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def auc_ap(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / tp[-1]; fpr = fp / fp[-1]
    auc = float(np.trapezoid(tpr, fpr))
    prec = tp / np.arange(1, len(y) + 1)
    ap = float((prec * y).sum() / y.sum())
    return auc, ap


def load(run, tier):
    man = {m["name"]: m for m in json.loads((run / "manifest.json").read_text())}
    rows = [json.loads(l) for l in (run / "scores.jsonl").read_text().splitlines() if l.strip()]
    per_video = {}  # video -> {tier: (y list, s list)}
    for r in rows:
        m = man[r["video"]]
        d = per_video.setdefault(r["video"], {k: ([], []) for k in ("small", "medium", "large", "all")})
        if not m.get("events"):
            for k in d:
                d[k][0].append(0); d[k][1].append(r["score"])
            continue
        fps = m["fps"]
        w0, w1 = r["start_s"] * fps, r["end_s"] * fps
        here = {t for e0, e1, t in tier.get(r["video"], []) if w0 < e1 + 1 and w1 > e0}
        d["all"][0].append(r["label"]); d["all"][1].append(r["score"])
        if r["label"]:
            for t in here:
                d[t][0].append(1); d[t][1].append(r["score"])
        else:
            for t in ("small", "medium", "large"):
                d[t][0].append(0); d[t][1].append(r["score"])
    return {v: {k: (np.array(y, float), np.array(s, float)) for k, (y, s) in d.items()} for v, d in per_video.items()}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--events", type=Path, required=True)
    ap_.add_argument("--base", type=Path, required=True)
    ap_.add_argument("runs", nargs="+", type=Path)
    ap_.add_argument("--n", type=int, default=2000)
    ap_.add_argument("--seed", type=int, default=0)
    a = ap_.parse_args()
    events = [json.loads(l) for l in a.events.read_text().splitlines() if l.strip()]
    area = np.array([e["area_frac"] for e in events])
    t1, t2 = np.quantile(area, [1 / 3, 2 / 3])
    tier = {}
    for e in events:
        t = "small" if e["area_frac"] < t1 else ("medium" if e["area_frac"] < t2 else "large")
        tier.setdefault(e["video"], []).append((e["event"][0], e["event"][1], t))

    base = load(a.base, tier)
    others = {str(r): load(r, tier) for r in a.runs}
    videos = sorted(base)
    for r, d in others.items():
        missing = set(videos) - set(d)
        if missing:
            raise SystemExit(f"{r}: {len(missing)} videos missing vs base")
    rng = np.random.default_rng(a.seed)
    keys = ("small", "medium", "large", "all")

    def score(d, idx):
        out = {}
        for k in keys:
            y = np.concatenate([d[videos[i]][k][0] for i in idx])
            s = np.concatenate([d[videos[i]][k][1] for i in idx])
            out[k] = auc_ap(y, s)
        return out

    full_idx = np.arange(len(videos))
    b0 = score(base, full_idx)
    print(f"base {a.base}")
    for k in keys:
        print(f"  {k:6s} AUC {100*b0[k][0]:.1f}  AP {100*b0[k][1]:.1f}")
    point = {r: score(d, full_idx) for r, d in others.items()}
    diffs = {r: {k: [] for k in keys} for r in others}
    for _ in range(a.n):
        idx = rng.integers(0, len(videos), len(videos))
        bb = score(base, idx)
        for r, d in others.items():
            oo = score(d, idx)
            for k in keys:
                diffs[r][k].append((oo[k][0] - bb[k][0], oo[k][1] - bb[k][1]))
    for r in others:
        print(f"\n{r}  (diff vs base; point estimate, bootstrap mean [95% CI], P(diff<=0))")
        for k in keys:
            arr = np.array(diffs[r][k])
            for j, name in enumerate(("AUC", "AP")):
                x = arr[:, j]; x = x[~np.isnan(x)]
                pt = 100 * (point[r][k][j] - b0[k][j])
                lo, hi = np.percentile(x, [2.5, 97.5]) * 100
                print(f"  {k:6s} {name}: {pt:+.1f}  boot {100*x.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]  P<=0 {np.mean(x <= 0):.3f}")


if __name__ == "__main__":
    main()
