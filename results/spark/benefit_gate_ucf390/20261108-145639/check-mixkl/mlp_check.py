#!/usr/bin/env python3
"""Question: is the linear head too weak for the zoom-benefit label, or do the features not hold it? (2026-11-08)

mix+KL label model, logistic weight (BCE with pos_weight, y = [b > 0]), 5-fold by video on the 390 training videos.
Two heads on the same standardized layer-18 patch_mean features and the same training loop (Adam lr 1e-3,
1,000 full-batch steps, + lam * sum of squared weights):
  linear  = benefit.py fit_linear (zero init)
  mlp     = Linear(d, 256) -> ReLU -> Linear(256, 1), PyTorch default init, seed 20261022
Per head: in-sample AUC (train folds, mean over folds) and held-out AUC (pooled), for up = [b > 0] and for Holmes.
  uv run --frozen python <this file> --config <run>/check-mixkl/config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from benefit import REPO, aligned, features, fit_linear, folds_of, loss_for, standardize
from label_auc import weighted_auc

HIDDEN = 256


def fit_mlp(x: torch.Tensor, loss_fn, cfg: dict) -> torch.nn.Module:
    torch.manual_seed(cfg["seed"])
    head = torch.nn.Sequential(torch.nn.Linear(x.shape[1], HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, 1))
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg["lr"])
    weights = [m.weight for m in head if isinstance(m, torch.nn.Linear)]
    for _ in range(cfg["steps"]):
        optimizer.zero_grad()
        loss = loss_fn(head(x).squeeze(1)) + cfg["lam"] * sum(w.square().sum() for w in weights)
        loss.backward()
        optimizer.step()
    return head


FITTERS = {"linear": fit_linear, "mlp": fit_mlp}


def auc(score: np.ndarray, y: np.ndarray) -> float:
    return weighted_auc(score, y, np.ones(len(y)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(p.parse_args().config.read_text())
    x, meta = features(Path(cfg["train"]["features"]), ["train_anomaly", "train_normal"], cfg["feature"])
    y = np.array([m["holmes"] for m in meta])
    fold = folds_of([m["video"] for m in meta], cfg["folds"])
    root = REPO / cfg["train"]["scores"] / "mixkl"
    s1, s2 = aligned(meta, root / "1x" / "scores.jsonl"), aligned(meta, root / "2x" / "scores.jsonl")
    up = (np.where(y == 1, s2 - s1, s1 - s2) > 0).astype(np.float32)
    print(f"{len(y)} windows, feature dim {x.shape[1]}, base rate of b > 0: {100 * up.mean():.1f}%")
    print(f"{'head':6}  in-sample AUC up  held-out AUC up  in-sample AUC Holmes  held-out AUC Holmes")
    for name, fitter in FITTERS.items():
        held = np.empty(len(y))
        in_up, in_holmes = [], []
        for k in range(cfg["folds"]["n"]):
            tr, te = fold != k, fold == k
            xt, xh = standardize(x[tr], x[te])
            head = fitter(xt, loss_for("logistic", up[tr], None), cfg["head"])
            with torch.inference_mode():
                fit_s, held[te] = head(xt).squeeze(1).numpy(), head(xh).squeeze(1).numpy()
            in_up.append(auc(fit_s, up[tr]))
            in_holmes.append(auc(fit_s, y[tr]))
        print(f"{name:6}  {np.mean(in_up):.3f}             {auc(held, up):.3f}            "
              f"{np.mean(in_holmes):.3f}                {auc(held, y):.3f}", flush=True)


if __name__ == "__main__":
    main()
