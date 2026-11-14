#!/usr/bin/env python3
"""How often does the foveation gate fire? (2026-11-06) Samples test windows from a run's manifest, computes the TGM
hot-area fraction exactly as scripts/tgm-size.py / input_forms/foveated.py do, and reports the share of windows
below several gates, split by labelled / unlabelled windows.
  python firing-rate.py <run dir with manifest.json + scores.jsonl> <root with videos> <n windows> <out.jsonl>"""
import json, random, sys
from pathlib import Path
import cv2, numpy as np

run, root, n, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
man = {m["name"]: m for m in json.loads((run / "manifest.json").read_text())}
rows = [json.loads(l) for l in (run / "scores.jsonl").read_text().splitlines() if l.strip()]
rng = random.Random(0); rng.shuffle(rows); rows = rows[:n]


def heat_of(frames):
    grey = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32) for f in frames]
    return cv2.GaussianBlur(np.sum([np.abs(a - b) for a, b in zip(grey[1:], grey[:-1])], axis=0), (31, 31), 0)


res = []
caps = {}
for i, r in enumerate(rows):
    m = man[r["video"]]
    p = root / m["path"]
    cap = caps.get(p)
    if cap is None:
        for c in caps.values():
            c.release()
        caps = {p: cv2.VideoCapture(str(p))}; cap = caps[p]
    fps = m["fps"]; f0 = int(round(r["start_s"] * fps)); f1 = int(round(r["end_s"] * fps))
    idx = np.linspace(f0, max(f1 - 1, f0), 8).round().astype(int)
    frames = []
    for k in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, fr = cap.read()
        if ok:
            frames.append(fr)
    if len(frames) < 2:
        continue
    h = heat_of(frames); peak = float(h.max())
    area = float((h > 0.5 * peak).mean()) if peak > 0 else 0.0
    res.append({"video": r["video"], "window": r["window"], "label": r["label"], "area": area})
    if i % 200 == 0:
        print(i, flush=True)
out.write_text("\n".join(json.dumps(x) for x in res) + "\n")
a = np.array([x["area"] for x in res]); lab = np.array([x["label"] for x in res], bool)
for g in (0.03, 0.045, 0.06, 0.0695):
    print(f"gate {g:.4f}: fires on {100*(a < g).mean():.1f}% of windows (labelled {100*(a[lab] < g).mean():.1f}%, unlabelled {100*(a[~lab] < g).mean():.1f}%)  n={len(a)}")
