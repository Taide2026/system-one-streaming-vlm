#!/usr/bin/env python3
"""Does the token count explain detection? (2026-11-14, CPU only.) Follows tokevent.py, removing between-video offsets.

r = share of the SAME video's negative windows that a positive window outscores (within-video rank, 0..1).
between slope: at one input, across windows, r per doubling of tokens on the event.
within slope : for one window, across inputs (only the token count changes), r per doubling.
If token count were the common currency, the two slopes would agree. Bootstrap over videos, 95%.
"""
import importlib.util, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
what = sys.argv[1]; rng = np.random.default_rng(0)
import glob
key = lambda n: n[:-4] if n.endswith(".mp4") else n
def load(name):
    ds = sorted(x for x in glob.glob(f"results/nano4/{name}/*/") if glob.glob(x + "run.json"))
    return {(key(r["video"]), r["window"]): r for r in map(json.loads, open(ds[-1] + "scores.jsonl"))}
def ranks(R):
    neg = defaultdict(list)
    for (v, _), r in R.items():
        if r["label"] == 0: neg[v].append(r["score"])
    out = {}
    for k, r in R.items():
        if r["label"] == 1 and len(neg[k[0]]) >= 5:
            n = np.array(neg[k[0]]); out[k] = ((n < r["score"]).sum() + 0.5 * (n == r["score"]).sum()) / len(n)
    return out
def slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); x = x - x.mean()
    return float((x * (y - y.mean())).sum() / max((x * x).sum(), 1e-12))
def boot(vids, f):
    vids = sorted(vids); est = f(vids); bs = [f(list(rng.choice(vids, len(vids)))) for _ in range(500)]
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
def report(title, tags, r, T):
    """r[tag][k] rank, T[tag][k] tokens (>0)."""
    ks = [k for k in r[tags[0]] if all(k in r[t] and T[t].get(k, 0) > 0 for t in tags)]
    byv = defaultdict(list)
    for k in ks: byv[k[0]].append(k)
    print(f"\n{title}: {len(ks)} positive windows in {len(byv)} videos")
    for t in tags:
        f = lambda vs: 100 * slope([np.log2(T[t][k]) for v in vs for k in byv[v]], [r[t][k] for v in vs for k in byv[v]])
        e, lo, hi = boot(byv, f); m = 100 * np.mean([r[t][k] for k in ks])
        print(f"  between, {t:16s}: mean rank {m:5.1f}; {e:+5.2f} [{lo:+.2f},{hi:+.2f}] rank points per doubling of tokens")
    def within(vs):
        xs, ys = [], []
        for v in vs:
            for k in byv[v]:
                x = np.array([np.log2(T[t][k]) for t in tags]); y = np.array([r[t][k] for t in tags]); xs += list(x - x.mean()); ys += list(y - y.mean())
        return 100 * slope(xs, ys)
    e, lo, hi = boot(byv, within); print(f"  within (same window, {tags[0]} .. {tags[-1]}): {e:+5.2f} [{lo:+.2f},{hi:+.2f}] rank points per doubling")
    def vw(t, sel=None):
        return 100 * np.mean([np.mean([r[t][k] for k in byv[v] if sel is None or k in sel]) for v in byv if sel is None or any(k in sel for k in byv[v])])
    print("  video-weighted mean rank (= Macro-AUC on these videos): " + ", ".join(f"{t} {vw(t):.1f}" for t in tags))
    def within_v(vs):
        out = []
        for v in vs:
            xs, ys = [], []
            for k in byv[v]:
                x = np.array([np.log2(T[t][k]) for t in tags]); y = np.array([r[t][k] for t in tags]); xs += list(x - x.mean()); ys += list(y - y.mean())
            out.append(slope(xs, ys))
        return 100 * np.mean(out)
    e, lo, hi = boot(byv, within_v); print(f"  within, video-weighted: {e:+5.2f} [{lo:+.2f},{hi:+.2f}] rank points per doubling")
    med = {v: np.median([T[tags[0]][k] for k in byv[v]]) for v in byv}; qs = np.quantile(list(med.values()), [0, 1/3, 2/3, 1])
    for a, b in zip(qs[:-1], qs[1:]):
        vs = [v for v in byv if a <= med[v] <= b]; g = [np.mean([r[tags[-1]][k] - r[tags[0]][k] for k in byv[v]]) for v in vs]
        bs = [np.mean(rng.choice(g, len(g))) for _ in range(1000)]
        print(f"    videos with median tokens at {tags[0]} {a:6.1f}-{b:6.1f} (n{len(vs):4d}): per-video AUC {100*np.mean([np.mean([r[tags[0]][k] for k in byv[v]]) for v in vs]):5.1f}, gain at {tags[-1]} {100*np.mean(g):+5.2f} [{100*np.percentile(bs,2.5):+.2f},{100*np.percentile(bs,97.5):+.2f}]")
    # the same windows, split by tokens at the first input: gain from first to last input
    t0, t1 = tags[0], tags[-1]; q = np.quantile([T[t0][k] for k in ks], [0, .25, .5, .75, 1])
    for a, b in zip(q[:-1], q[1:]):
        sel = [k for k in ks if a <= T[t0][k] <= b]
        print(f"    tokens at {t0} {a:6.1f}-{b:6.1f} (n{len(sel):5d}): rank {100*np.mean([r[t0][k] for k in sel]):5.1f} -> {100*np.mean([r[t1][k] for k in sel]):5.1f} at {t1}")
if what == "ucf":
    ev = defaultdict(list)
    for l in open(sys.argv[2]):
        e = json.loads(l); ev[key(e["video"])].append((e["event"][0], e["event"][1], e["area_frac"]))
    TPF = {"0.5x": 70, "0.75x": 70, "1x": 80, "1.25x": 108, "1.5x": 165, "1.75x": 234, "2x": 300}
    NAME = {"0.5x": "dn05", "0.75x": "dn075", "1x": "frames", "1.25x": "1p25", "1.5x": "1p5", "1.75x": "1p75", "2x": "all2x"}
    runs = {t: [load(f"up290_floor_s{s}_{n}") for s in range(3)] for t, n in NAME.items()}
    r = {}
    for t in NAME:
        rs = [ranks(R) for R in runs[t]]; r[t] = {k: float(np.mean([x[k] for x in rs])) for k in rs[0] if all(k in x for x in rs)}
    base = {}
    for k, w in runs["1x"][0].items():
        if w["label"] != 1: continue
        t0, L = w["start_s"], w["end_s"] - w["start_s"]; lo, hi = t0 * 30, w["end_s"] * 30 - 1
        hit = [(b, e, ar) for b, e, ar in ev.get(k[0], []) if lo <= e and b <= hi]
        if not hit: continue
        n = sum(any(b <= round((t0 + (j + 0.5) * L / 8) * 30) <= e for b, e, _ in hit) for j in range(8))
        base[k] = min(ar for _, _, ar in hit) * n
    T = {t: {k: a * TPF[t] for k, a in base.items()} for t in NAME}
    report("UCF-Crime fine-tuned, 1x .. 2x", ["1x", "1.25x", "1.5x", "1.75x", "2x"], r, T)
    report("UCF-Crime fine-tuned, 0.5x vs 1x (70 vs 80 tokens per frame)", ["0.5x", "1x"], r, T)
else:
    root = Path(sys.argv[2]); GRID = {"180p": (6, 11), "bicubic 2x": (11, 20), "real 360p": (11, 20)}; NAME = {"180p": "v180", "bicubic 2x": "v180up2", "real 360p": "v360"}
    ann = {}
    def count(v, w, grid):
        if v not in ann:
            by = defaultdict(list)
            for x in json.loads((root / "annotations" / f"{v}_annotation.json").read_text())["annotations"]: by[int(x["frame_id"])].append(x["bbox"])
            ann[v] = (by, json.loads((root / "meta" / f"{v}.json").read_text())["fps"])
        by, fps = ann[v]; gh, gw = grid; t0, L = w["start_s"], w["end_s"] - w["start_s"]; tot = 0
        for j in range(8):
            f = round((t0 + (j + 0.5) * L / 8) * fps); m = np.zeros((gh, gw), bool)
            for ff in range(f - 2, f + 3):
                for x, y, bw, bh in by.get(ff, []):
                    c0, c1 = int(x / 1920 * gw), min(int(np.ceil((x + bw) / 1920 * gw)), gw); r0, r1 = int(y / 1080 * gh), min(int(np.ceil((y + bh) / 1080 * gh)), gh)
                    m[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)] = True
            tot += int(m.sum())
        return tot
    for mt, mn in (("zs", "zero-shot"), ("f0", "fine-tuned on UCF-Crime")):
        runs = {t: load(f"cvadall_{mt}_{n}") for t, n in NAME.items()}; r = {t: ranks(R) for t, R in runs.items()}
        T = {t: {k: count(k[0], w, GRID[t]) for k, w in runs["180p"].items() if w["label"] == 1} for t in GRID}
        report(f"ComplexVAD 113, {mn}, 180p vs bicubic 2x", ["180p", "bicubic 2x"], r, T)
        ks = [k for k in r["bicubic 2x"] if k in r["real 360p"]]
        print(f"  same tokens, different pixels: bicubic 2x rank {100*np.mean([r['bicubic 2x'][k] for k in ks]):.1f}, real 360p {100*np.mean([r['real 360p'][k] for k in ks]):.1f}")
print("TOKSLOPE_DONE")
