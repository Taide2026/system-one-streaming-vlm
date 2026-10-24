#!/usr/bin/env python3
"""Post-hoc calibration of a fine-tuned run, without touching the GPU again (2026-10-24).

  python scripts/calibrate.py --val-scores <val-adapter.jsonl> --val-coverage <soft run's val.jsonl>
                              --calibration <train run's calibration.json> --test <merged eval dir> [--out calib.json]

The fine-tuned models rank well (AUC) but put too much probability on the label-0 windows around events, and their
one temperature T_old was fitted on held-out training windows labelled by coverage >= 0.5, while the test windows are
labelled by the runner's rule (any overlap with an event, inclusive). Two fixes, both fitted on the held-out windows
relabelled by the test rule (coverage > 0 -> 1) and applied to the test logits:
  1. T refit:  p = sigmoid(d / T)
  2. Platt:    p = sigmoid(a * d + b)
where d is the two-option logit difference z_Yes - z_No, recovered from the stored probabilities as
d = T_old * logit(p) (the backbone divided by T_old before the softmax). The ranking (AUC) is unchanged by either.

Prints, for the test set, frame AUC (unchanged), window AUC, 10-bin ECE and NLL for the original, T-refit and Platt
probabilities, overall and split into normal videos / label-0 windows inside anomaly videos / abnormal windows, plus
the fitted parameters; writes them to --out when given, and a Typst row per variant.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += abs(y[m].mean() - p[m].mean()) * m.mean()
    return float(total)


def fit_temperature(d: np.ndarray, y: np.ndarray) -> float:
    """Golden-section search of T in [0.1, 10] (on log T) minimising the NLL."""
    lo, hi = math.log(0.1), math.log(10.0)
    phi = (math.sqrt(5) - 1) / 2
    f = lambda t: nll(sigmoid(d / math.exp(t)), y)  # noqa: E731
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = f(a), f(b)
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = f(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = f(b)
    return float(math.exp((lo + hi) / 2))


def fit_platt(d: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Logistic regression of y on d (two parameters a, b) by Newton's method."""
    a, b = 1.0, 0.0
    for _ in range(100):
        z = a * d + b
        p = sigmoid(z)
        w = p * (1 - p) + 1e-9
        g = np.array([((p - y) * d).sum(), (p - y).sum()])
        h = np.array([[(w * d * d).sum(), (w * d).sum()], [(w * d).sum(), w.sum()]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-scores", type=Path, required=True)
    ap.add_argument("--val-coverage", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    t_old = float(json.loads(args.calibration.read_text())["temperature"])
    coverage = {(r["video"], r["window"]): r["y"] for r in map(json.loads, args.val_coverage.read_text().splitlines()) if r}
    val = [json.loads(l) for l in args.val_scores.read_text().splitlines() if l.strip()]
    d_val = t_old * logit(np.array([r["p"] for r in val]))
    y_val = np.array([1.0 if coverage[(r["video"], r["window"])] > 0 else 0.0 for r in val])  # the test rule

    t_new = fit_temperature(d_val, y_val)
    a, b = fit_platt(d_val, y_val)

    rows = [json.loads(l) for l in (args.test / "scores.jsonl").read_text().splitlines() if l.strip()]
    manifest = {v["name"]: v for v in json.loads((args.test / "manifest.json").read_text())}
    p0 = np.array([r["score"] for r in rows])
    y = np.array([r["label"] for r in rows], dtype=float)
    d = t_old * logit(p0)
    variants = {"original (T_old=%.2f)" % t_old: p0, "T refit (T=%.2f)" % t_new: sigmoid(d / t_new),
                "Platt (a=%.2f, b=%.2f)" % (a, b): sigmoid(a * d + b)}
    normal_video = np.array([not manifest[r["video"]]["events"] for r in rows])
    subsets = {"all": np.ones(len(rows), bool), "normal videos": normal_video,
               "label-0 in anomaly videos": (~normal_video) & (y == 0), "abnormal windows": y == 1}
    result = {"t_old": t_old, "t_new": t_new, "platt": [a, b], "val_windows": len(val), "variants": {}}
    print(f"{'variant':28s} {'ECE all':>8s} {'NLL':>7s} {'ECE norm':>9s} {'ECE lab0-anom':>13s} {'ECE abn':>8s} {'FPR@.5 norm':>11s} {'rec@.5':>7s}")
    for name, p in variants.items():
        e = {k: ece(p[m], y[m]) for k, m in subsets.items()}
        fpr_n = float((p[normal_video] >= 0.5).mean())
        rec = float((p[y == 1] >= 0.5).mean())
        result["variants"][name] = {"ece": e, "nll": nll(p, y), "fpr_normal_0.5": fpr_n, "recall_0.5": rec}
        print(f"{name:28s} {e['all']:8.3f} {nll(p, y):7.3f} {e['normal videos']:9.3f} {e['label-0 in anomaly videos']:13.3f} "
              f"{e['abnormal windows']:8.3f} {fpr_n:11.3f} {rec:7.3f}")
    print("\n// Typst: [variant], [ECE all], [ECE normal videos], [ECE label-0 anomaly videos], [ECE abnormal], [FPR@0.5 normal], [recall@0.5]")
    for name, v in result["variants"].items():
        e = v["ece"]
        print(f"[{name}], [{e['all']:.3f}], [{e['normal videos']:.3f}], [{e['label-0 in anomaly videos']:.3f}], "
              f"[{e['abnormal windows']:.3f}], [{100 * v['fpr_normal_0.5']:.1f}], [{100 * v['recall_0.5']:.1f}],")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
