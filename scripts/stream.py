#!/usr/bin/env python3
"""Streaming-side readings of existing score files (2026-11-13). CPU only.

(1) Alarm protocol. A camera produces 1,800 windows an hour. For a budget of A false alarms per hour of NORMAL video
    the threshold is set on the normal test videos; reported: the share of events caught (any window of the event
    above the threshold) and the delay of the first alarm (end of the first flagged window minus the event's start),
    all events and events under 4% of the frame.
(2) Calibration of the decision probability. Expected calibration error (10 equal-width bins) and Brier score of
    P(Yes) against the window label; the share of normal-video windows above 0.5; and the share above the threshold
    that gives 5% on the FIRST input named (the threshold a deployment would have set before changing the input).

  python scripts/stream.py --events E --inputs 1x=a,b,c 2x=d,e,f ...     (first input = reference for the fixed threshold)
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

p = argparse.ArgumentParser(); p.add_argument("--events", required=True); p.add_argument("--inputs", nargs="+", required=True)
p.add_argument("--budgets", type=float, nargs="+", default=[2, 6, 20]); p.add_argument("--cut", type=float, default=0.04); p.add_argument("--fps", type=float, default=30.0)
p.add_argument("--out")
a = p.parse_args()
key = lambda n: n[:-4] if n.endswith(".mp4") else n
E = [json.loads(l) for l in open(a.events)]
ev = defaultdict(list)
for i, e in enumerate(E):
    ev[key(e["video"])].append((e["event"][0], e["event"][1], i))
small = np.array([e["area_frac"] < a.cut for e in E])


def load(name):
    ds = sorted(x for x in glob.glob(f"results/nano4/{name}/*/") if glob.glob(x + "run.json"))
    return [json.loads(l) for l in open(ds[-1] + "scores.jsonl")]


def analyse(R, tau5_ref=None):
    vids_pos = {key(r["video"]) for r in R if r["label"] == 1}
    normal = np.array([r["score"] for r in R if key(r["video"]) not in vids_pos])
    hours = len(normal) * 2 / 3600
    y = np.array([r["label"] for r in R], float); s = np.array([r["score"] for r in R])
    bins = np.minimum((s * 10).astype(int), 9)
    ece = sum(abs(s[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    out = {"ece": 100 * ece, "brier": 100 * float(((s - y) ** 2).mean()), "normal_above_half": 100 * float((normal > 0.5).mean()), "normal_hours": hours}
    tau5 = float(np.quantile(normal, 0.95)); out["tau5"] = tau5
    out["normal_above_ref_tau"] = 100 * float((normal > (tau5_ref if tau5_ref is not None else tau5)).mean())
    per_event = defaultdict(list)                                   # event -> [(end_s, score)]
    for r in R:
        if r["label"] == 1:
            lo, hi = r["start_s"] * a.fps, r["end_s"] * a.fps - 1
            for b, e, i in ev.get(key(r["video"]), []):
                if lo <= e and b <= hi:
                    per_event[i].append((r["end_s"], r["score"]))
    order = np.sort(normal)[::-1]
    for A in a.budgets:
        k = int(round(A * hours)); tau = float(order[min(k, len(order) - 1)])
        caught = np.zeros(len(E), bool); delay = np.full(len(E), np.nan)
        for i, ws in per_event.items():
            hit = [t for t, sc in sorted(ws) if sc > tau]
            if hit:
                caught[i] = True; delay[i] = max(hit[0] - E[i]["event"][0] / a.fps, 0.0)
        seen = np.array([i in per_event for i in range(len(E))])
        out[f"A{A:g}"] = {"recall": 100 * caught[seen].mean(), "recall_small": 100 * caught[seen & small].mean(), "recall_large": 100 * caught[seen & ~small].mean(),
                          "delay_median": float(np.nanmedian(delay)) if caught.any() else float("nan"), "delay_mean": float(np.nanmean(delay)) if caught.any() else float("nan"), "tau": tau}
    return out


res, ref_tau = {}, None
for spec in a.inputs:
    tag, names = spec.split("="); runs = [load(n) for n in names.split(",")]
    per = [analyse(R, None if ref_tau is None else ref_tau[min(j, len(ref_tau) - 1)]) for j, R in enumerate(runs)]
    if ref_tau is None:
        ref_tau = [x["tau5"] for x in per]
    res[tag] = per
m = lambda tag, f: (float(np.mean([f(x) for x in res[tag]])), float(np.std([f(x) for x in res[tag]], ddof=1)) if len(res[tag]) > 1 else 0.0)
print(f"normal video in the test set: {res[next(iter(res))][0]['normal_hours']:.2f} h")
print(f"{'input':14s} " + " ".join(f"| {A:g}/h: caught  small  large  delay(med,s)" for A in a.budgets))
for tag in res:
    print(f"{tag:14s} " + " ".join(f"|       {m(tag, lambda x: x[f'A{A:g}']['recall'])[0]:5.1f}  {m(tag, lambda x: x[f'A{A:g}']['recall_small'])[0]:5.1f}  {m(tag, lambda x: x[f'A{A:g}']['recall_large'])[0]:5.1f}  {m(tag, lambda x: x[f'A{A:g}']['delay_median'])[0]:5.1f}       " for A in a.budgets))
print(f"\n{'input':14s}    ECE      Brier   normal>0.5   normal>tau fixed on the first input (5% there)")
for tag in res:
    print(f"{tag:14s} {m(tag, lambda x: x['ece'])[0]:5.1f}±{m(tag, lambda x: x['ece'])[1]:3.1f}  {m(tag, lambda x: x['brier'])[0]:5.1f}   {m(tag, lambda x: x['normal_above_half'])[0]:6.2f}%      {m(tag, lambda x: x['normal_above_ref_tau'])[0]:5.1f}±{m(tag, lambda x: x['normal_above_ref_tau'])[1]:3.1f}%")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
print("STREAM_DONE")
