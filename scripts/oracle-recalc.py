#!/usr/bin/env python3
"""Oracle placement, re-derived from the stored rows (2026-11-14). CPU only.
AUC of small-event positives and of all positives against the sampled negatives, per layout; paired differences with a
bootstrap over videos (95% half-width).   python scripts/oracle-recalc.py <rows.jsonl> ..."""
import json, sys
from collections import Counter, defaultdict
import numpy as np
names = ("coarse", "rand_k5", "tight", "touch", "abs_k5", "abs_k24", "full2x", "label_k5")
def U(p, n):
    n = np.sort(n); return 100 * (np.searchsorted(n, p, "left") + np.searchsorted(n, p, "right")).sum() / 2 / (len(p) * len(n))
for path in sys.argv[1:]:
    R = [json.loads(l) for l in open(path)]
    print(path, len(R), "windows;", dict(Counter(r["group"] for r in R)))
    lab = np.array([r["label"] for r in R]); small = np.array([r["group"] == "small+" for r in R]); vid = np.array([r["video"] for r in R])
    M = {n: np.array([r["margin"][n] for r in R]) for n in names}
    tok = {n: int(np.mean([r["tokens"][n] for r in R])) for n in names}
    def auc(n, idx=None, which="small"):
        i = np.arange(len(R)) if idx is None else idx
        pos = (small[i] if which == "small" else lab[i] == 1); neg = lab[i] == 0
        return U(M[n][i][pos], M[n][i][neg]) if pos.any() and neg.any() else np.nan
    print(f"{'layout':10s} tokens  small   all")
    for n in names:
        print(f"{n:10s} {tok[n]:6d}  {auc(n):5.1f}  {auc(n, which='all'):5.1f}")
    by = defaultdict(list)
    for i, v in enumerate(vid): by[v].append(i)
    vs = sorted(by); rng = np.random.default_rng(0)
    boots = [np.concatenate([by[v] for v in rng.choice(vs, len(vs))]) for _ in range(1000)]
    for a, b in (("abs_k5", "tight"), ("abs_k24", "touch"), ("tight", "rand_k5"), ("tight", "full2x"), ("label_k5", "tight")):
        for which in ("small", "all"):
            d = auc(a, which=which) - auc(b, which=which); bs = [auc(a, i, which) - auc(b, i, which) for i in boots]
            lo, hi = np.nanpercentile(bs, [2.5, 97.5]); print(f"  {a} - {b}, {which:5s}: {d:+.1f} ± {(hi - lo) / 2:.1f}   [{lo:+.1f}, {hi:+.1f}]")
    eff = np.array([r["effect"] for r in R]); top5 = np.sort(np.abs(eff), 1)[:, -5:].sum(1) / np.abs(eff).sum(1)
    hit = [np.mean([b in set(np.argsort(-np.abs(e))[:5]) for b in np.flatnonzero(np.array(r["mask"]["tight"]).ravel())]) if np.array(r["mask"]["tight"]).any() else np.nan for r, e in zip(R, eff)]
    print(f"  share of total |effect| in the five most influential blocks: {100 * np.mean(top5):.0f}%; motion-map (tight) blocks that are among those five: {100 * np.nanmean(hit):.0f}% (chance {100 * 5 / 80:.0f}%)")
print("ORACLE_RECALC_DONE")
