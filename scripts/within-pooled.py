#!/usr/bin/env python3
"""Pooled versus within-video metrics (2026-11-11), from finished runs' scores.jsonl. Numpy only, no model.

  python scripts/within-pooled.py [--events <jev-hard-events.jsonl> --manifest-fps 30] <run dir> ...

Per run:
  pooled AUC / AP                  every positive window against every negative window
  within AUC (pair-weighted)       only pairs of the same video; Macro = the same videos, equal weight
  cross AUC                        only pairs from two different videos;  share of within pairs among all pairs
  video-mean baseline AUC / AP     every window score replaced by the mean score of its video
  video-level AUC                  max window score per video, anomaly videos against normal videos
The pooled AUC is the pair-weighted mean of the within and the cross AUC, so the three are consistent.
Output: one JSON line per run on stdout (and a readable line on stderr).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def ustat(pos, neg):
    """Number of (pos, neg) pairs won by pos (ties count half)."""
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    neg = np.sort(neg)
    return float((np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right")).sum() / 2)


def ap(y, s):
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1) * y).sum() / max(y.sum(), 1))


def one(run: Path) -> dict:
    rows = [json.loads(l) for l in (run / "scores.jsonl").open()]
    y = np.array([r["label"] for r in rows], float)
    s = np.array([r["score"] for r in rows], float)
    vid = np.array([r["video"] for r in rows])
    by = defaultdict(list)
    for i, v in enumerate(vid):
        by[v].append(i)
    pos, neg = s[y == 1], s[y == 0]
    u_all, n_all = ustat(pos, neg), len(pos) * len(neg)
    u_in = n_in = 0.0
    per_video = []
    mean_s = np.empty_like(s)
    vmax, vlab = [], []
    for v, idx in by.items():
        idx = np.array(idx)
        sv, yv = s[idx], y[idx]
        mean_s[idx] = sv.mean()
        vmax.append(sv.max()); vlab.append(float(yv.max()))
        p, n = sv[yv == 1], sv[yv == 0]
        if len(p) and len(n):
            u = ustat(p, n)
            u_in += u; n_in += len(p) * len(n)
            per_video.append(u / (len(p) * len(n)))
    vmax, vlab = np.array(vmax), np.array(vlab)
    out = {
        "run": str(run), "windows": len(rows), "videos": len(by), "mixed_videos": len(per_video),
        "pooled_auc": 100 * u_all / n_all, "pooled_ap": 100 * ap(y, s),
        "within_auc": 100 * u_in / n_in, "macro_auc": 100 * float(np.mean(per_video)),
        "cross_auc": 100 * (u_all - u_in) / (n_all - n_in), "within_pair_share": 100 * n_in / n_all,
        "videomean_auc": 100 * ustat(mean_s[y == 1], mean_s[y == 0]) / n_all, "videomean_ap": 100 * ap(y, mean_s),
        "video_auc": 100 * ustat(vmax[vlab == 1], vmax[vlab == 0]) / max((vlab == 1).sum() * (vlab == 0).sum(), 1),
    }
    return out


for a in sys.argv[1:]:
    r = one(Path(a))
    print(json.dumps(r))
    print(f"{Path(a).parent.name:32s} pooled {r['pooled_auc']:.1f}/{r['pooled_ap']:.1f}  within {r['within_auc']:.1f} "
          f"macro {r['macro_auc']:.1f} cross {r['cross_auc']:.1f} (within pairs {r['within_pair_share']:.2f}%)  "
          f"video-mean {r['videomean_auc']:.1f}/{r['videomean_ap']:.1f}  video-level {r['video_auc']:.1f}", file=sys.stderr)
