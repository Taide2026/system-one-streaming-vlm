#!/usr/bin/env python3
"""Does a change-driven selection see the anomaly? (2026-11-13) CPU only.

ComplexVAD has boxes. For every anomalous window of the listed videos, on the 180p frames our runs use:
  crop        the bounding box of the moving blocks (the key-frame layout's crop)
  motion      the moving blocks themselves (patch rows "same motion map")
  changed     the 2x2-token groups that changed most against the previous frame, at the token budget of the crop
and the share of the annotated box area each selection covers (a window's boxes: all annotated in its two seconds).
A track is "static" when its box centre moves less than half the box diagonal over the whole track.

  python scripts/cvad-cover.py <root> [--list list30.txt] [--scores-a results/nano4/<whole>/<run>/scores.jsonl --scores-b <layout run>]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import s1stream.plugins  # noqa: F401
from s1stream.input_forms.frames import read_frames
from s1stream.input_forms.keycrop import motion_mask
from s1stream.registry import DATASETS
from s1stream.windows import plan_windows

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path); ap.add_argument("--list", default="list30.txt"); ap.add_argument("--variant", default="v180")
ap.add_argument("--scores-a", type=Path); ap.add_argument("--scores-b", type=Path); ap.add_argument("--out", type=Path)
args = ap.parse_args()
items = DATASETS.build({"name": "complexvad", "path": str(args.root), "variant": args.variant, "videos": args.list}).items()
W0, H0 = 1920, 1080
rows, per_video = [], defaultdict(list)
for item in items:
    meta = json.loads((args.root / "meta" / f"{item.name}.json").read_text()); fps = meta["fps"]
    ann = json.loads((args.root / "annotations" / f"{item.name}_annotation.json").read_text())["annotations"]
    tracks = defaultdict(list)
    for a in ann:
        tracks[a["track_id"]].append(a)
    static = {}
    for tid, t in tracks.items():
        c = np.array([[b["bbox"][0] + b["bbox"][2] / 2, b["bbox"][1] + b["bbox"][3] / 2] for b in t])
        diag = float(np.mean([np.hypot(b["bbox"][2], b["bbox"][3]) for b in t]))
        static[tid] = bool(np.linalg.norm(c.max(0) - c.min(0)) < 0.5 * diag)
    by_frame = defaultdict(list)
    for a in ann:
        by_frame[int(a["frame_id"])].append(a)
    for w in plan_windows(item, 2.0, 2.0, 8):
        if w.label != 1:
            continue
        lo, hi = round(w.start_s * fps), round(w.end_s * fps) - 1
        boxes = [a for f in range(lo, hi + 1) for a in by_frame.get(f, [])]
        if not boxes:
            continue
        images = read_frames(item.path, w.frame_indices)
        arrays = [np.asarray(im.convert("RGB")) for im in images]
        h, wd = arrays[0].shape[:2]; sx, sy = wd / W0, h / H0
        hb, wb = -(-h // 32), -(-wd // 32)
        mask = motion_mask(arrays)
        fine = cv2.resize(mask, (wb, hb), interpolation=cv2.INTER_AREA) > 0.05
        sel = {"crop": np.zeros((h, wd), bool), "motion": np.zeros((h, wd), bool), "changed": np.zeros((h, wd), bool)}
        n_cells = 0
        if fine.any():
            rr, cc = np.where(fine)
            sel["crop"][rr.min() * 32:(rr.max() + 1) * 32, cc.min() * 32:(cc.max() + 1) * 32] = True
            sel["motion"] = cv2.resize(fine.astype(np.uint8), (wb * 32, hb * 32), interpolation=cv2.INTER_NEAREST)[:h, :wd].astype(bool)
            n_cells = int((rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)) * 4          # 16-pixel groups in the crop
        grey = np.stack([cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255 for a in arrays])
        d = np.abs(grey[1:] - grey[:-1]).mean(0)                                             # mean change over the window
        gh, gw = -(-h // 16), -(-wd // 16)
        cell = cv2.resize(d, (gw, gh), interpolation=cv2.INTER_AREA)
        if n_cells:
            thr = np.sort(cell.ravel())[::-1][min(n_cells, cell.size) - 1]
            sel["changed"] = cv2.resize((cell >= thr).astype(np.uint8), (gw * 16, gh * 16), interpolation=cv2.INTER_NEAREST)[:h, :wd].astype(bool)
        seen = {}
        for a in boxes:                                                                       # one box per track: the track's union in the window
            x, y, bw, bh = a["bbox"]; k = a["track_id"]
            r = seen.setdefault(k, [x, y, x + bw, y + bh, a["object_type"]])
            r[0], r[1], r[2], r[3] = min(r[0], x), min(r[1], y), max(r[2], x + bw), max(r[3], y + bh)
        for tid, (x0, y0, x1, y1, typ) in seen.items():
            a0, b0, a1, b1 = int(x0 * sx), int(y0 * sy), max(int(np.ceil(x1 * sx)), int(x0 * sx) + 1), max(int(np.ceil(y1 * sy)), int(y0 * sy) + 1)
            area = (a1 - a0) * (b1 - b0)
            cov = {k: float(v[b0:b1, a0:a1].sum()) / area for k, v in sel.items()}
            rows.append({"video": item.name, "window": w.index, "track": tid, "type": typ, "static": static[tid], "area_pct": 100 * area / (h * wd), **cov})
            per_video[item.name].append(cov["crop"])
    print(item.name, "windows so far", len({(r['video'], r['window']) for r in rows}), flush=True)

R = rows
print(f"\n{len(R)} annotated objects in {len({(r['video'], r['window']) for r in R})} anomalous windows of {len(items)} videos; selected share of the frame (crop): "
      f"{100 * np.mean([r['crop'] >= 0 for r in R]):.0f}% rows")
def line(name, rs):
    if not rs:
        return
    c = lambda k: np.array([r[k] for r in rs])
    print(f"{name:28s} n {len(rs):5d} | box area covered, mean: crop {100 * c('crop').mean():5.1f}%  motion blocks {100 * c('motion').mean():5.1f}%  most-changed {100 * c('changed').mean():5.1f}% | "
          f"missed entirely (<10% covered): crop {100 * (c('crop') < 0.1).mean():5.1f}%  motion {100 * (c('motion') < 0.1).mean():5.1f}%  changed {100 * (c('changed') < 0.1).mean():5.1f}%")
line("all objects", R)
line("static tracks", [r for r in R if r["static"]]); line("moving tracks", [r for r in R if not r["static"]])
for t in sorted({r["type"] for r in R}, key=lambda t: -sum(r["type"] == t for r in R))[:8]:
    line(f"type {t}", [r for r in R if r["type"] == t])
if args.scores_a and args.scores_b:
    def per(path):
        by = defaultdict(lambda: ([], []))
        for l in path.open():
            r = json.loads(l); by[r["video"]][r["label"]].append(r["score"])
        out = {}
        for v, (n, p) in by.items():
            if n and p:
                n_, p_ = np.sort(n), np.array(p)
                out[v] = (np.searchsorted(n_, p_, "left") + np.searchsorted(n_, p_, "right")).sum() / 2 / (len(p_) * len(n_))
        return out
    A, B = per(args.scores_a), per(args.scores_b)
    vs = [v for v in per_video if v in A and v in B]
    cov = np.array([np.mean(per_video[v]) for v in vs]); delta = np.array([B[v] - A[v] for v in vs])
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    rho = float(np.corrcoef(rank(cov), rank(delta))[0, 1]); rng = np.random.default_rng(0)
    p = float(np.mean([abs(np.corrcoef(rank(cov), rank(rng.permutation(delta)))[0, 1]) >= abs(rho) for _ in range(5000)]))   # permutation test
    lo, hi = cov < np.median(cov), cov >= np.median(cov)
    print(f"\nper video ({len(vs)}): Spearman(coverage of the boxes by the crop, Macro-AUC change layout - whole) = {rho:+.2f} (p {p:.3f}); "
          f"mean change, low-coverage half {100 * delta[lo].mean():+.1f}, high-coverage half {100 * delta[hi].mean():+.1f}")
if args.out:
    args.out.write_text("\n".join(json.dumps(r) for r in R))
print("COVER_DONE")
