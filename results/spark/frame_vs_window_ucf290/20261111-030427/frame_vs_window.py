"""Why frame AP < window AP: positive rates and boundary frames (label 0 frames inside a label 1 window)."""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
from s1stream import metrics
from s1stream.windows import window_frame_range
man = json.loads(Path(sys.argv[1]).read_text()); scores = Path(sys.argv[2])
fps = {m["name"]: float(m["fps"]) for m in man}
rows = [json.loads(l) for l in scores.read_text().splitlines()]
wl = np.array([r["label"] for r in rows]); ws = np.array([r["score"] for r in rows])
print(f"window: n={len(rows)} positive {wl.mean()*100:.1f}%  AP {metrics.ap(ws, wl)*100:.1f}")
S, L, B = [], [], []
byv = {}
for r in rows: byv.setdefault(r["video"], []).append(r)
for m in man:
    n = int(m["n_frames"]); fs = np.full(n, -np.inf); inpos = np.zeros(n, bool); fl = np.zeros(n, int)
    for r in byv[m["name"]]:
        a, b = window_frame_range(r["start_s"], r["end_s"], fps[m["name"]]); a, b = max(a, 0), min(b, n - 1)
        fs[a:b+1] = np.maximum(fs[a:b+1], r["score"])
        if r["label"]: inpos[a:b+1] = True
    for e0, e1 in m["events"]: fl[max(e0, 0):min(e1, n - 1) + 1] = 1
    c = np.isfinite(fs); S.append(fs[c]); L.append(fl[c]); B.append(inpos[c])
S, L, B = map(np.concatenate, (S, L, B))
bnd = (L == 0) & B
print(f"frame: n={len(S)} positive {L.mean()*100:.1f}%")
print(f"boundary frames (label 0 inside a label-1 window): {bnd.sum()} = {bnd.sum()/(L==0).sum()*100:.2f}% of negatives, mean score {S[bnd].mean():.3f} vs other negatives {S[(L==0)&~B].mean():.3f}")
pos_in = (L == 1) & ~B
print(f"positive frames not covered by any label-1 window: {pos_in.sum()} = {pos_in.sum()/(L==1).sum()*100:.1f}% of positives, mean score {S[pos_in].mean():.3f} vs other positives {S[(L==1)&B].mean():.3f}")
ap = metrics.ap
print(f"frame AP {ap(S, L)*100:.1f}; without boundary negatives {ap(S[~bnd], L[~bnd])*100:.1f}; without both {ap(S[~bnd & ~pos_in], L[~bnd & ~pos_in])*100:.1f}")
