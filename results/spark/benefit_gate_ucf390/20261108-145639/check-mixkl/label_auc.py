#!/usr/bin/env python3
"""Question: does the benefit head fail because it cannot predict the zoom-benefit label? (2026-11-08)

Cross-fitted (5-fold by video) head scores on the 390 training videos, mix+KL label model, per weight:
  AUC up      = AUC of the head score for up = [b > 0]
  AUC up |b|  = the same AUC with each window weighted by |b| (pairs weighted by the product)
  AUC Holmes  = AUC of the head score for the Holmes event label (is it an anomaly detector?)
  up rate     = share of b > 0 among the windows at or above the top-20% cutoff, against the base rate
  uv run --frozen python <this file> --config <run>/check-mixkl/config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from benefit import REPO, aligned, features, fit_score, folds_of, targets, top_cutoff


def weighted_auc(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """P(score_pos > score_neg) over pairs weighted by w_pos * w_neg; ties count 1/2."""
    order = np.argsort(score, kind="stable")
    s, yy, ww = score[order], y[order], w[order]
    _, start, counts = np.unique(s, return_index=True, return_counts=True)
    total = 0.0
    neg_below = 0.0
    for a, n in zip(start, counts):
        pos_w = ww[a:a + n][yy[a:a + n] == 1].sum()
        neg_w = ww[a:a + n][yy[a:a + n] == 0].sum()
        total += pos_w * (neg_below + 0.5 * neg_w)
        neg_below += neg_w
    return total / (ww[y == 1].sum() * ww[y == 0].sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(p.parse_args().config.read_text())
    x, meta = features(Path(cfg["train"]["features"]), ["train_anomaly", "train_normal"], cfg["feature"])
    y = np.array([m["holmes"] for m in meta])
    fold = folds_of([m["video"] for m in meta], cfg["folds"])
    root = REPO / cfg["train"]["scores"] / "mixkl"
    s1, s2 = aligned(meta, root / "1x" / "scores.jsonl"), aligned(meta, root / "2x" / "scores.jsonl")
    b = np.where(y == 1, s2 - s1, s1 - s2)
    up = (b > 0).astype(int)
    tg = targets(y, s1, s2, cfg["ap_clip_percentile"])
    ones = np.ones(len(y))
    print(f"{len(y)} windows; base rate of b > 0: {100 * up.mean():.1f}%")
    print(f"{'weight':8}  AUC up  AUC up |b|  AUC Holmes  up rate in top 20%")
    for name in cfg["weights"]:
        label, weight = tg[name]
        held = np.empty(len(y))
        for k in range(cfg["folds"]["n"]):
            tr, te = fold != k, fold == k
            _, held[te] = fit_score(x[tr], label[tr], None if weight is None else weight[tr], name, cfg["head"], x[te])
        top = held >= top_cutoff(held, cfg["upscale_rate"])
        print(f"{name:8}  {weighted_auc(held, up, ones):.3f}   {weighted_auc(held, up, np.abs(b)):.3f}       "
              f"{weighted_auc(held, y, ones):.3f}       {100 * up[top].mean():.1f}%", flush=True)


if __name__ == "__main__":
    main()
