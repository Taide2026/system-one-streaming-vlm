#!/usr/bin/env python3
"""Tokens on the event, counted (2026-11-14). CPU only, from existing score files.

For every positive window and every input size: T = the number of visual tokens that land on the event, summed over
the window's eight frames. Then: AUC of the positive windows with a given T against all negative windows of the same
input. If the model is limited by tokens on the event, inputs of different size should give the same AUC at the same T.

  ucf    T = (share of the frame the event's motion covers) x (tokens per frame at that input) x (sampled frames
         inside the event). Tokens per frame as the processor produces them: 70 at 0.5x and 0.75x (it enlarges small
         inputs), 80, 108, 165, 234, 300 at 1x .. 2x.
  cvad   T = token cells overlapped by the annotated boxes, counted frame by frame on the grid of that input
         (6x11 at 180p, 11x20 at 360p and at 180p enlarged 2x).

  python scripts/tokevent.py ucf --events E
  python scripts/tokevent.py cvad --root /work/u9117820/datasets/complexvad
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(); p.add_argument("what", choices=["ucf", "cvad"]); p.add_argument("--events"); p.add_argument("--root", type=Path)
p.add_argument("--out"); a = p.parse_args()
EDGES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 1e9]
LABELS = ["<2", "2-4", "4-8", "8-16", "16-32", "32-64", "64-128", "128-256", ">=256"]
key = lambda n: n[:-4] if n.endswith(".mp4") else n


def U(pos, neg):
    neg = np.sort(neg)
    return (np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right")).sum() / 2 / (len(pos) * len(neg))


def load(name):
    ds = sorted(x for x in glob.glob(f"results/nano4/{name}/*/") if glob.glob(x + "run.json"))
    return {(key(r["video"]), r["window"]): r for r in map(json.loads, open(ds[-1] + "scores.jsonl"))}


def table(inputs, T_of):
    """inputs: {tag: [runs]}; T_of(tag) -> {window key: tokens on the event}. Prints AUC per token bin and input."""
    res = {}
    print(f"{'tokens on event':16s} " + " ".join(f"{t:>14s}" for t in inputs))
    for b, lab in enumerate(LABELS):
        cells = []
        for tag, runs in inputs.items():
            T = T_of(tag); ks = [k for k, v in T.items() if EDGES[b] <= v < EDGES[b + 1]]
            aucs = []
            for R in runs:
                neg = np.array([r["score"] for r in R.values() if r["label"] == 0]); pos = np.array([R[k]["score"] for k in ks if k in R])
                if len(pos) >= 20:
                    aucs.append(100 * U(pos, neg))
            cells.append((float(np.mean(aucs)) if aucs else float("nan"), len(ks)))
            res.setdefault(tag, {})[lab] = {"auc": cells[-1][0], "n": len(ks)}
        print(f"{lab:16s} " + " ".join((f"{m:6.1f} (n{n:5d})" if n >= 20 else f"   --  (n{n:5d})") for m, n in cells))
    return res


if a.what == "ucf":
    ev = defaultdict(list)
    for l in open(a.events):
        e = json.loads(l); ev[key(e["video"])].append((e["event"][0], e["event"][1], e["area_frac"]))
    TPF = {"0.5x": 70, "0.75x": 70, "1x": 80, "1.25x": 108, "1.5x": 165, "1.75x": 234, "2x": 300}
    NAME = {"0.5x": "dn05", "0.75x": "dn075", "1x": "frames", "1.25x": "1p25", "1.5x": "1p5", "1.75x": "1p75", "2x": "all2x"}
    inputs = {t: [load(f"up290_floor_s{s}_{n}") for s in range(3)] for t, n in NAME.items()}
    base = {}
    for k, r in inputs["1x"][0].items():
        if r["label"] != 1:
            continue
        t0, L = r["start_s"], r["end_s"] - r["start_s"]; lo, hi = t0 * 30, r["end_s"] * 30 - 1
        hit = [(b, e, ar) for b, e, ar in ev.get(k[0], []) if lo <= e and b <= hi]
        if not hit:
            continue
        fr = [round((t0 + (j + 0.5) * L / 8) * 30) for j in range(8)]
        n = sum(any(b <= f <= e for b, e, _ in hit) for f in fr)
        base[k] = (min(ar for _, _, ar in hit), n)
    print(f"UCF-Crime, fine-tuned d3-nano, three seeds; {len(base)} positive windows. AUC against all negatives of the same input.")
    res = table(inputs, lambda tag: {k: ar * TPF[tag] * n for k, (ar, n) in base.items()})
else:
    GRID = {"180p": (6, 11), "180p bicubic 2x": (11, 20), "real 360p": (11, 20)}
    NAME = {"180p": "v180", "180p bicubic 2x": "v180up2", "real 360p": "v360"}
    ann, meta = {}, {}
    def boxes_of(v):
        if v not in ann:
            by = defaultdict(list)
            for x in json.loads((a.root / "annotations" / f"{v}_annotation.json").read_text())["annotations"]:
                by[int(x["frame_id"])].append(x["bbox"])
            ann[v] = by; meta[v] = json.loads((a.root / "meta" / f"{v}.json").read_text())["fps"]
        return ann[v], meta[v]
    def count(v, r, grid):
        by, fps = boxes_of(v); gh, gw = grid; t0, L = r["start_s"], r["end_s"] - r["start_s"]; tot = 0
        for j in range(8):
            f = round((t0 + (j + 0.5) * L / 8) * fps); m = np.zeros((gh, gw), bool)
            for ff in range(f - 2, f + 3):
                for x, y, w, h in by.get(ff, []):
                    c0, c1 = int(x / 1920 * gw), min(int(np.ceil((x + w) / 1920 * gw)), gw); r0, r1 = int(y / 1080 * gh), min(int(np.ceil((y + h) / 1080 * gh)), gh)
                    m[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)] = True
            tot += int(m.sum())
        return tot
    for mtag, mname in (("zs", "zero-shot"), ("f0", "fine-tuned on UCF-Crime")):
        inputs = {t: [load(f"cvadall_{mtag}_{n}")] for t, n in NAME.items()}
        pos = {k: r for k, r in inputs["180p"][0].items() if r["label"] == 1}
        Ts = {t: {k: count(k[0], r, GRID[t]) for k, r in pos.items()} for t in GRID}
        print(f"\nComplexVAD 113 videos, {mname}; {len(pos)} positive windows; median tokens on the event: " + ", ".join(f"{t} {np.median(list(Ts[t].values())):.0f}" for t in GRID))
        res = table(inputs, lambda tag: Ts[tag])
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
print("TOKEVENT_DONE")
