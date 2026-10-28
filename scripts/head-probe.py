#!/usr/bin/env python3
"""HeadHunt-style probe on the features of scripts/head-features.py (2026-10-28), nothing trained in the model:

  uv run --with numpy --with scikit-learn python scripts/head-probe.py --train <train.npz> --test <test.npz>
      --manifest <test run's manifest.json> [--k 5] [--n-train 200] [--seed 0]

1. Per head (attention and DeltaNet, and h_T as one 'head'): Fisher ratio and train AUC of its best linear direction
   (mean difference) on the n-train training windows. 2. Keep the K best heads by Fisher ratio, concatenate.
3. Three read-outs on the same features: logistic regression, shrinkage LDA, mean-difference direction.
4. Test: window AUC, frame AUC / AP (UCF convention) and ECE. Baselines printed alongside: h_T probe, the model's own
   logits (zero-shot). Prints a Typst row per setting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def frame_metrics(video, window, score, manifest, length_s=2.0, stride_s=2.0):
    S, Y = [], []
    by: dict[str, list] = {}
    for v, w, s in zip(video, window, score):
        by.setdefault(str(v), []).append((int(w), float(s)))
    for v, ws in by.items():
        m = manifest[v]; n, fps = m["n_frames"], m["fps"]
        fs = np.full(n, -np.inf); fl = np.zeros(n, int)
        for w, s in ws:
            a = int(round(w * stride_s * fps)); b = int(round((w * stride_s + length_s) * fps)) - 1
            a, b = max(a, 0), min(b, n - 1)
            if b >= a: fs[a:b + 1] = np.maximum(fs[a:b + 1], s)
        for e0, e1 in m["events"]: fl[max(e0, 0):min(e1, n - 1) + 1] = 1
        c = np.isfinite(fs); S.append(fs[c]); Y.append(fl[c])
    S, Y = np.concatenate(S), np.concatenate(Y)
    return 100 * roc_auc_score(Y, S), 100 * average_precision_score(Y, S)


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1); tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): tot += abs(y[m].mean() - p[m].mean()) * m.mean()
    return tot


def heads(z):
    """(N, L, H, D) -> list of (name, (N, D)) per head; h_T as one head."""
    out = []
    for l in range(z["attn"].shape[1]):
        for h in range(z["attn"].shape[2]): out.append((f"attn L{l} h{h}", z["attn"][:, l, h].astype(np.float32)))
    for l in range(z["linear"].shape[1]):
        for h in range(z["linear"].shape[2]): out.append((f"delta L{l} h{h}", z["linear"][:, l, h].astype(np.float32)))
    return out


def fisher(x, y):
    a, b = x[y == 1], x[y == 0]
    d = a.mean(0) - b.mean(0); w = d / (a.var(0) + b.var(0) + 1e-6)  # diagonal LDA direction
    proj = x @ w
    return float((proj[y == 1].mean() - proj[y == 0].mean()) ** 2 / (proj[y == 1].var() + proj[y == 0].var() + 1e-9)), w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True); ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", choices=["attn", "all"], default="all", help="candidate heads: attention only, or DeltaNet too")
    args = ap.parse_args()
    tr, te = np.load(args.train), np.load(args.test)
    manifest = {v["name"]: v for v in json.loads(args.manifest.read_text())}
    rng = np.random.default_rng(args.seed)
    ytr_all = tr["label"].astype(int); yte = te["label"].astype(int)
    pos, neg = np.where(ytr_all == 1)[0], np.where(ytr_all == 0)[0]
    n = min(args.n_train // 2, len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    ytr = ytr_all[sel]
    htr, hte = heads(tr), heads(te)
    if args.pool == "attn":
        keep = [i for i, (name, _) in enumerate(htr) if name.startswith("attn")]
        htr, hte = [htr[i] for i in keep], [hte[i] for i in keep]
    scores = [(fisher(x[sel], ytr)[0], i) for i, (_, x) in enumerate(htr)]
    scores.sort(reverse=True)
    top = [i for _, i in scores[:args.k]]
    print(f"train {len(sel)} windows ({n} + {n}); test {len(yte)} windows; candidates {len(htr)} heads; top-{args.k}: "
          + ", ".join(f"{htr[i][0]} (F {f:.2f})" for f, i in scores[:args.k]), flush=True)

    def evaluate(name, Xtr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        rows = []
        lr = LogisticRegression(C=0.5, max_iter=2000).fit(Xtr, ytr); rows.append(("logistic", lr.predict_proba(Xte)[:, 1]))
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr); rows.append(("LDA (shrinkage)", lda.predict_proba(Xte)[:, 1]))
        w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0); d = Xte @ w; dtr = Xtr @ w
        p = 1 / (1 + np.exp(-(d - dtr.mean()) / (dtr.std() + 1e-6) * 2)); rows.append(("mean difference", p))
        for clf, p in rows:
            fa, fp = frame_metrics(te["video"], te["window"], p, manifest)
            print(f"  {name:34s} {clf:16s} window AUC {100 * roc_auc_score(yte, p):5.1f}  frame AUC {fa:5.1f}  AP {fp:5.1f}  ECE {ece(p, yte):.3f}")
            print(f"    // Typst: [{name}, {clf}], [{fa:.1f}], [{fp:.1f}], [{ece(p, yte):.3f}],")

    z = te["logits"]; p0 = np.exp(z[:, 0]) / np.exp(z).sum(1)
    fa, fp = frame_metrics(te["video"], te["window"], p0, manifest)
    print(f"  {'model logits (zero-shot)':34s} {'':16s} window AUC {100 * roc_auc_score(yte, p0):5.1f}  frame AUC {fa:5.1f}  AP {fp:5.1f}  ECE {ece(p0, yte):.3f}")
    evaluate("h_T", tr["hT"][sel].astype(np.float32), te["hT"].astype(np.float32))
    evaluate(f"top-{args.k} heads ({args.pool})", np.concatenate([htr[i][1][sel] for i in top], 1), np.concatenate([hte[i][1] for i in top], 1))
    evaluate(f"all attention heads, mean-pooled", tr["attn"][sel].astype(np.float32).mean((1, 2)), te["attn"].astype(np.float32).mean((1, 2)))


if __name__ == "__main__":
    main()
