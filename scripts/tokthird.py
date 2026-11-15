#!/usr/bin/env python3
"""Is the gain from a larger input tied to how small the event is? (2026-11-15, CPU only, existing score files.)
ComplexVAD: videos are split into thirds by the median number of tokens on the event at 180p (annotated boxes on the
6x11 grid, as in tokevent.py). Per third and model: mean per-video AUC at 180p, and the change with bicubic 2x and with
real 360p (bootstrap over videos, 95% half-width).   python scripts/tokthird.py --root /work/u9117820/datasets/complexvad
"""
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); a = p.parse_args()
key = lambda n: n[:-4] if n.endswith(".mp4") else n
rng = np.random.default_rng(0)
def load(name):
    ds = sorted(x for x in glob.glob(f"results/nano4/{name}/*/") if glob.glob(x + "run.json"))
    return {(key(r["video"]), r["window"]): r for r in map(json.loads, open(ds[-1] + "scores.jsonl"))}
def U(pos, neg):
    neg = np.sort(neg); return (np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right")).sum() / 2 / (len(pos) * len(neg))
def per_video(R):
    by = defaultdict(lambda: ([], []))
    for (v, _), r in R.items(): by[v][r["label"]].append(r["score"])
    return {v: 100 * U(np.array(po), np.array(ne)) for v, (ne, po) in by.items() if ne and po}
ann = {}
def boxes_of(v):
    if v not in ann:
        by = defaultdict(list)
        for x in json.loads((a.root / "annotations" / f"{v}_annotation.json").read_text())["annotations"]: by[int(x["frame_id"])].append(x["bbox"])
        ann[v] = (by, json.loads((a.root / "meta" / f"{v}.json").read_text())["fps"])
    return ann[v]
def count(v, r, gh=6, gw=11):
    by, fps = boxes_of(v); t0, L = r["start_s"], r["end_s"] - r["start_s"]; tot = 0
    for j in range(8):
        f = round((t0 + (j + 0.5) * L / 8) * fps); m = np.zeros((gh, gw), bool)
        for ff in range(f - 2, f + 3):
            for x, y, w, h in by.get(ff, []):
                c0, c1 = int(x / 1920 * gw), min(int(np.ceil((x + w) / 1920 * gw)), gw); r0, r1 = int(y / 1080 * gh), min(int(np.ceil((y + h) / 1080 * gh)), gh)
                m[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)] = True
        tot += int(m.sum())
    return tot
def ci(d):
    d = np.asarray(d); b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]; lo, hi = np.percentile(b, [2.5, 97.5])
    return f"{d.mean():+5.1f} ± {(hi - lo) / 2:3.1f}"
MODELS = {"d3-nano zero-shot": "cvadall_zs_{}", "d3-nano fine-tuned": "cvadall_f0_{}", "d3-flash zero-shot": "cvadall_flzs_{}", "d3-flash fine-tuned s1": "cvadall_flf1_{}", "d3-flash fine-tuned s2": "cvadall_flf2_{}", "Qwen3-VL-4B neutral": "svcvad_qwen3-vl-4b_yn_{}"}
base = load("cvadall_zs_v180"); T = defaultdict(list)
for (v, _), r in base.items():
    if r["label"] == 1: T[v].append(count(v, r))
med = {v: float(np.median(x)) for v, x in T.items()}
for name, pat in MODELS.items():
    A = {t: per_video(load(pat.format(t))) for t in ("v180", "v180up2", "v360")}
    vids = sorted((v for v in A["v180"] if v in med and v in A["v180up2"] and v in A["v360"]), key=lambda v: med[v]); n = len(vids)
    print(f"\n{name}: {n} videos")
    print(f"{'third (tokens on event, 8 frames)':36s} {'n':>3s} {'180p':>6s} {'bicubic 2x':>12s} {'real 360p':>12s} {'real - bicubic':>14s}")
    for lab, g in (("fewest", vids[: n // 3]), ("middle", vids[n // 3: 2 * n // 3]), ("most", vids[2 * n // 3:]), ("all", vids)):
        b = np.array([A["v180"][v] for v in g]); u = np.array([A["v180up2"][v] for v in g]); r = np.array([A["v360"][v] for v in g])
        rngT = f"{med[g[0]]:.0f}-{med[g[-1]]:.0f}"
        print(f"{lab + ' (' + rngT + ')':36s} {len(g):3d} {b.mean():6.1f} {ci(u - b):>12s} {ci(r - b):>12s} {ci(r - u):>14s}")
print("TOKTHIRD_DONE")
