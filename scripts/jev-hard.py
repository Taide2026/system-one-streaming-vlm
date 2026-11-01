#!/usr/bin/env python3
"""jev-hard subset scores (2026-11-01): frame AUC / AP restricted to the hard spatial regime, from finished runs.

  python scripts/jev-hard.py --events results/pro6000/jev-hard-events.jsonl <run dir> [<run dir> ...]

The subset: windows overlapping a small-tercile event (area_frac terciles over the 156 events from tgm-size.py)
plus every window of every normal video (so false alarms still count). Windows of medium / large events and the
remaining anomaly-video windows are dropped - the metric asks: can the model separate SMALL events from normal
footage? Frame AUC / AP weight windows by their frame span as scripts/score-records.py does (uniform windows, so
plain window-level AUC / AP equals the frame metric up to the final partial window)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def auc_ap(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / max(tp[-1], 1); fpr = fp / max(fp[-1], 1)
    auc = float(np.trapezoid(tpr, fpr))
    prec = tp / np.arange(1, len(y) + 1)
    ap = float((prec * y).sum() / max(y.sum(), 1))
    return auc, ap


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--events", type=Path, required=True)
    ap_.add_argument("runs", nargs="+", type=Path)
    args = ap_.parse_args()
    events = [json.loads(l) for l in args.events.read_text().splitlines() if l.strip()]
    a = np.array([e["area_frac"] for e in events])
    t1, t2 = np.quantile(a, [1 / 3, 2 / 3])
    tier = {}
    for e in events:
        t = "small" if e["area_frac"] < t1 else ("medium" if e["area_frac"] < t2 else "large")
        tier.setdefault(e["video"], []).append((e["event"][0], e["event"][1], t))
    print(f"{len(events)} events; terciles at {t1:.4f} / {t2:.4f}")

    for run in args.runs:
        man = {m["name"]: m for m in json.loads((run / "manifest.json").read_text())}
        rows = [json.loads(l) for l in (run / "scores.jsonl").read_text().splitlines() if l.strip()]
        subsets = {k: ([], []) for k in ("small", "medium", "large", "all")}
        for r in rows:
            m = man[r["video"]]
            fps = m["fps"]
            if not m.get("events"):  # normal video: negative everywhere, in every subset
                for k in subsets:
                    subsets[k][0].append(0); subsets[k][1].append(r["score"])
                continue
            w0, w1 = r["start_s"] * fps, r["end_s"] * fps
            tiers_here = {t for e0, e1, t in tier.get(r["video"], []) if w0 < e1 + 1 and w1 > e0}
            subsets["all"][0].append(r["label"]); subsets["all"][1].append(r["score"])
            if r["label"] and tiers_here:
                for t in tiers_here:
                    subsets[t][0].append(1); subsets[t][1].append(r["score"])
            # anomaly-video windows outside any event are dropped from the tier subsets (boundary ambiguity)
        line = [str(run)]
        for k in ("small", "medium", "large", "all"):
            y, s = np.array(subsets[k][0], float), np.array(subsets[k][1], float)
            auc, ap = auc_ap(y, s)
            line.append(f"{k}: AUC {100 * auc:.1f} AP {100 * ap:.1f} (n+ {int(y.sum())})")
        print("  ".join(line))


if __name__ == "__main__":
    main()
