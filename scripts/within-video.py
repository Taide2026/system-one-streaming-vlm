#!/usr/bin/env python3
"""Where does the score live: video level or window level? (2026-11-06)
  python within-video.py <run> [<run> ...]
Per run: (1) video-level AUC: max window score per video, anomaly videos vs normal videos; (2) within-video window AP
and AUC on anomaly videos only (positives = labelled windows, negatives = the unlabelled windows of the same videos),
mean over videos with both kinds of windows and pooled; (3) standard all-window AP for reference; (4) mean score of
normal videos' windows vs the unlabelled windows inside anomaly videos."""
import json, sys
import numpy as np
from pathlib import Path

R = Path("results/pro6000")


def latest(n):
    if (R / n / "scores.jsonl").exists():
        return R / n
    ds = sorted(d for d in (R / n).iterdir() if (d / "scores.jsonl").exists() or (d / "merged" / "scores.jsonl").exists())
    d = ds[-1]
    return d if (d / "scores.jsonl").exists() else d / "merged"


def ap_auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    o = np.argsort(-s, kind="stable"); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    auc = float(np.trapezoid(tp / tp[-1], fp / fp[-1]))
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum()), auc


print(f"{'run':28s} video-AUC | within-video AP/AUC pooled | mean per video | all-window AP | normal-win mean | anomaly-video unlabelled mean | labelled mean")
for n in sys.argv[1:]:
    d = latest(n)
    man = {m["name"]: m for m in json.loads((d / "manifest.json").read_text())}
    byv = {}
    for r in map(json.loads, (d / "scores.jsonl").read_text().splitlines()):
        if r:
            byv.setdefault(r["video"], []).append(r)
    vy, vs = [], []
    wy, ws = [], []
    per = []
    norm, unl, lab = [], [], []
    ally, alls = [], []
    for v, rs in byv.items():
        s = [r["score"] for r in rs]; y = [r["label"] for r in rs]
        ally += y; alls += s
        anom = bool(man[v].get("events"))
        vy.append(1 if anom else 0); vs.append(max(s))
        if anom:
            wy += y; ws += s
            unl += [x for x, l in zip(s, y) if not l]; lab += [x for x, l in zip(s, y) if l]
            a, _ = ap_auc(y, s)
            if not np.isnan(a):
                per.append(a)
        else:
            norm += s
    vauc = ap_auc(vy, vs)[1]
    wap, wauc = ap_auc(wy, ws)
    aap, _ = ap_auc(ally, alls)
    print(f"{n[:28]:28s} {100*vauc:5.1f}     | {100*wap:5.1f} / {100*wauc:5.1f}            | {100*np.mean(per):5.1f}          | {100*aap:5.1f}         | {np.mean(norm):.2f}            | {np.mean(unl):.2f}                          | {np.mean(lab):.2f}")
