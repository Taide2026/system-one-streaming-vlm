#!/usr/bin/env python3
"""Four score-file analyses of the train x test runs (2026-11-11). Numpy only.
  python scripts/four.py --events results/pro6000/jev-hard-events.jsonl --tag floor|fovfull2x
For seeds 0-2 of one training input it reads results/nano4/up290_<tag>_s<k>_{frames,full2x}/<latest>/scores.jsonl
(A = original test input, B = 4+4 mix) and prints:
 1 size as a continuous variable   per-event AUC (event windows vs all negatives), change B-A against event area
 2 complementarity                 correlation, fusions, per-window oracle
 3 operating point and calibration fixed threshold from A (5% of normal-video windows alarm), recalibrated, ECE
 4 large-tier stability            leave-one-event-out influence on large AP, concentration of windows
"""
import argparse, glob, json
from collections import defaultdict
import numpy as np

p = argparse.ArgumentParser(); p.add_argument("--events", required=True); p.add_argument("--tag", default="floor"); a_ = p.parse_args()
key = lambda n: n[:-4] if n.endswith(".mp4") else n
E = [json.loads(l) for l in open(a_.events)]
ar = sorted(e["area_frac"] for e in E); t1, t2 = ar[len(ar) // 3], ar[2 * len(ar) // 3]
ev = defaultdict(list)
for i, e in enumerate(E): ev[key(e["video"])].append((e["event"][0], e["event"][1], i))
area = np.array([e["area_frac"] for e in E]); tier = np.where(area < t1, 0, np.where(area < t2, 1, 2)); TN = ["small", "medium", "large"]

def load(name):
    d = sorted(x for x in glob.glob(f"results/nano4/{name}/*/") if glob.glob(x + "run.json"))[-1]
    return {(key(r["video"]), r["window"]): r for r in map(json.loads, open(d + "scores.jsonl"))}
lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
def ap(y, s):
    o = np.argsort(-s, kind="stable"); y = y[o]; return 100 * (np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / max(y.sum(), 1)
def auc(pos, neg_sorted):
    return (np.searchsorted(neg_sorted, pos, "left") + np.searchsorted(neg_sorted, pos, "right")).sum() / 2 / (len(pos) * len(neg_sorted))

S = []
for k in range(3):
    A, B = load(f"up290_{a_.tag}_s{k}_frames"), load(f"up290_{a_.tag}_s{k}_full2x"); keys = sorted(A); assert keys == sorted(B)
    y = np.array([A[q]["label"] for q in keys]); sa = np.array([A[q]["score"] for q in keys]); sb = np.array([B[q]["score"] for q in keys])
    vid = np.array([q[0] for q in keys]); eid = np.full(len(keys), -1)
    for i, q in enumerate(keys):
        if y[i] == 1:
            r = A[q]; lo, hi = r["start_s"] * 30, r["end_s"] * 30 - 1
            eid[i] = next((j for s, e, j in ev.get(q[0], []) if lo <= e and s <= hi), -1)
    anom = set(vid[y == 1]); normal = np.array([v not in anom for v in vid])
    S.append(dict(y=y, a=sa, b=sb, vid=vid, eid=eid, normal=normal))
y, eid, vid, normal = S[0]["y"], S[0]["eid"], S[0]["vid"], S[0]["normal"]
wt = np.where(eid >= 0, tier[np.clip(eid, 0, None)], -1); neg = y == 0
print(f"tag {a_.tag}: windows {len(y)}, positives per tier {[int((wt == t).sum()) for t in range(3)]}, unmatched positives {int(((y == 1) & (eid < 0)).sum())}")
def tiers(s):
    return [ap(y[neg | (wt == t)], s[neg | (wt == t)]) for t in range(3)] + [ap(y, s)]
def macro(s):
    out = []
    for v in np.unique(vid[y == 1]):
        m = vid == v
        if 0 < y[m].sum() < m.sum(): out.append(auc(s[m][y[m] == 1], np.sort(s[m][y[m] == 0])))
    return 100 * float(np.mean(out))

print("\n== 1 size as a continuous variable (per-event AUC against all negatives; mean of 3 seeds)")
ids = [j for j in range(len(E)) if (eid == j).any()]
da = np.zeros((3, len(ids))); base = np.zeros((3, len(ids)))
for k, s in enumerate(S):
    na, nb = np.sort(s["a"][neg]), np.sort(s["b"][neg])
    for c, j in enumerate(ids):
        m = eid == j; base[k, c] = 100 * auc(s["a"][m], na); da[k, c] = 100 * auc(s["b"][m], nb) - base[k, c]
d = da.mean(0); x = np.log10(area[ids]); nwin = np.array([(eid == j).sum() for j in ids])
rk = lambda v: np.argsort(np.argsort(v)).astype(float)
rho = np.corrcoef(rk(x), rk(d))[0, 1]; rng = np.random.default_rng(0)
bs = [np.corrcoef(rk(x[i]), rk(d[i]))[0, 1] for i in (rng.integers(0, len(ids), len(ids)) for _ in range(2000))]
print(f"events {len(ids)}; Spearman(change, log area) = {rho:+.3f}  95% [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]; per seed {[round(float(np.corrcoef(rk(x), rk(da[k]))[0, 1]), 3) for k in range(3)]}")
sl, ic = np.polyfit(x, d, 1); print(f"linear fit: change = {sl:+.2f} * log10(area) {ic:+.2f}; zero crossing at area {100 * 10 ** (-ic / sl):.2f}% of the frame" if sl < 0 else f"linear fit slope {sl:+.2f} (not negative)")
print("  area quintile   area range (%)   events  base AUC   change B-A   share of events improved")
qs = np.quantile(area[ids], [0, .2, .4, .6, .8, 1])
for i in range(5):
    m = (area[ids] >= qs[i]) & ((area[ids] < qs[i + 1]) | (i == 4))
    print(f"  Q{i + 1}              {100 * qs[i]:5.2f}-{100 * qs[i + 1]:5.2f}      {m.sum():3d}     {base.mean(0)[m].mean():5.1f}     {d[m].mean():+5.2f}        {100 * (d[m] > 0).mean():3.0f}%")
hi = base.mean(0) > 95
print(f"  ceiling check: events with base AUC > 95: {hi.sum()} (mean change {d[hi].mean():+.2f}); others: {(~hi).sum()} (mean change {d[~hi].mean():+.2f})")
print(f"  partial: Spearman(change, log area) among events with base AUC <= 95: {np.corrcoef(rk(x[~hi]), rk(d[~hi]))[0, 1]:+.3f};  Spearman(change, base AUC) = {np.corrcoef(rk(base.mean(0)), rk(d))[0, 1]:+.3f}")

print("\n== 2 complementarity (AP small / medium / large / all | Macro-AUC), mean of 3 seeds")
names = ["A original", "B mix", "mean logit", "max p", "mean logit, each centred", "oracle per window"]; acc = {n: [] for n in names}; cors = []
for s in S:
    la, lb = lg(s["a"]), lg(s["b"]); cors.append(np.corrcoef(rk(la), rk(lb))[0, 1])
    v = {"A original": s["a"], "B mix": s["b"], "mean logit": (la + lb) / 2, "max p": np.maximum(s["a"], s["b"]),
         "mean logit, each centred": (la - np.median(la[normal]) + lb - np.median(lb[normal])) / 2,
         "oracle per window": np.where(y == 1, np.maximum(s["a"], s["b"]), np.minimum(s["a"], s["b"]))}
    for n in names: acc[n].append(tiers(v[n]) + [macro(v[n])])
print(f"rank correlation of A and B scores per seed: {[round(float(c), 3) for c in cors]}")
for n in names:
    m = np.mean(acc[n], 0); print(f"  {n:26s} {m[0]:5.1f} {m[1]:5.1f} {m[2]:5.1f} {m[3]:5.1f} | {m[4]:5.1f}")

print("\n== 3 operating point and calibration (threshold: 5% of normal-video windows alarm on input A; per seed)")
def ece(pv, yv):
    b = np.minimum((pv * 10).astype(int), 9); return 100 * sum(abs(pv[b == i].mean() - yv[b == i].mean()) * (b == i).mean() for i in range(10) if (b == i).any())
for k, s in enumerate(S):
    ta = np.quantile(s["a"][normal], 0.95); tb = np.quantile(s["b"][normal], 0.95)
    def op(sc, th):
        fa = 100 * (sc[normal] > th).mean(); far_anom = 100 * (sc[neg & ~normal] > th).mean()
        rec = [100 * np.mean([(sc[eid == j] > th).any() for j in ids if tier[j] == t]) for t in range(3)]
        wrec = [100 * (sc[wt == t] > th).mean() for t in range(3)]
        return fa, far_anom, rec, wrec
    for label, sc, th in (("A, threshold from A", s["a"], ta), ("B, threshold from A", s["b"], ta), ("B, recalibrated   ", s["b"], tb)):
        fa, fa2, rec, wrec = op(sc, th)
        print(f"  seed {k} {label} thr {th:.3f}: alarms on normal-video windows {fa:4.1f}%, on unlabelled windows of anomaly videos {fa2:4.1f}%; event recall S/M/L {rec[0]:.0f}/{rec[1]:.0f}/{rec[2]:.0f}; window recall {wrec[0]:.0f}/{wrec[1]:.0f}/{wrec[2]:.0f}")
    print(f"  seed {k} ECE A {ece(s['a'], y):.1f}  B {ece(s['b'], y):.1f}; mean p on normal-video windows A {s['a'][normal].mean():.3f}  B {s['b'][normal].mean():.3f}")

print("\n== 4 large-tier stability (input A)")
L = [j for j in ids if tier[j] == 2]; nw = np.array([(eid == j).sum() for j in L]); o = np.argsort(-nw)
print(f"large events {len(L)}, positive windows {nw.sum()}; the 3 longest events hold {100 * nw[o[:3]].sum() / nw.sum():.0f}% of them, the 10 longest {100 * nw[o[:10]].sum() / nw.sum():.0f}%; median windows per event {int(np.median(nw))}")
full = []; infl = np.zeros((3, len(L)))
for k, s in enumerate(S):
    m = neg | (wt == 2); f = ap(y[m], s["a"][m]); full.append(f)
    for c, j in enumerate(L):
        mm = m & (eid != j); infl[k, c] = f - ap(y[mm], s["a"][mm])
    t = np.argsort(-np.abs(infl[k]))[:3]
    print(f"  seed {k}: large AP {f:.1f}; leave-one-event-out changes range {(-infl[k]).min():+.1f} .. {(-infl[k]).max():+.1f}; top 3 influential: " + ", ".join(f"{E[L[c]]['video'][:14]} ({nw[c]} win, {infl[k, c]:+.1f})" for c in t))
gap = infl[0] - infl[2]; t = np.argsort(-np.abs(gap))[:5]
print(f"  seed 0 vs seed 2 gap {full[0] - full[2]:+.1f} AP; events whose contribution differs most: " + ", ".join(f"{E[L[c]]['video'][:14]} ({nw[c]} win, {gap[c]:+.1f})" for c in t))
pe = np.array([[100 * auc(s["a"][eid == j], np.sort(s["a"][neg])) for j in L] for s in S])
print(f"  per-event AUC, large events: mean over events of the across-seed range {np.mean(pe.max(0) - pe.min(0)):.1f}; events with range > 10: {(pe.max(0) - pe.min(0) > 10).sum()} of {len(L)}")
print(f"  unweighted mean per-event AUC (each event counts once), seeds: {[round(float(v), 1) for v in pe.mean(1)]} -> range {pe.mean(1).max() - pe.mean(1).min():.1f}")
