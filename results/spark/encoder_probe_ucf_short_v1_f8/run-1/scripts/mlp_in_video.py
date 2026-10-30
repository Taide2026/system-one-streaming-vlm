"""Diagnostic: does a non-linear head generalize better inside a video than the linear head?

Question: train_in_video.py shows the linear head reaches in-video AUC 0.81-0.99 on its train windows but only
0.59-0.68 on validation and test. If an MLP head lifts validation and test, the linear head is the limit; if it lifts
only train, the limit is generalization to new scenes.
Head: Linear(hidden, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1), Adam lr 1e-3, full batch, window BCE with
pos_weight (as pairwise_fit.fit_head with alpha 0). For each fit in FITS, the state is the one train_in_video.py used
(alpha 0); weight decay in WEIGHT_DECAYS and the step count in CHECKPOINTS are chosen on the validation in-video AUC.
Values (window-level in-video AUC, label > 0.5), as in train_in_video.py:
  train / val / test   head fit on the fit split for val, on all train windows for train and test
  oracle_cv            head fit on test windows with ground-truth labels, 5 folds split by test video
Usage: uv run python <this file>
Writes run-1/analysis-pairwise/mlp-in-video.json and prints one line per fit.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from pairwise_fit import in_video_auc, read_meta
from s1stream.probe.fit import SEED, standardize
from train_in_video import FITS, FOLDS, chosen

RUN = Path(__file__).resolve().parents[1]
HIDDEN = 256
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2)
CHECKPOINTS = (250, 500, 1000)
DEVICE = "cuda"


def make_head(width: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(width, HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(DROPOUT),
                               torch.nn.Linear(HIDDEN, 1))


def fit_mlp(x: torch.Tensor, y: np.ndarray, weight_decay: float) -> dict[int, torch.nn.Module]:
    """Train one MLP head; return a CPU copy of it at each step count in CHECKPOINTS."""
    x, y = x.to(DEVICE), torch.tensor(y).to(DEVICE)
    pos = y.mean().clamp(min=1e-6)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).detach())
    torch.manual_seed(SEED)
    head = make_head(x.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=weight_decay)
    heads = {}
    for step in range(1, max(CHECKPOINTS) + 1):
        head.train()
        optimizer.zero_grad()
        bce(head(x).squeeze(1), y).backward()
        optimizer.step()
        if step in CHECKPOINTS:
            heads[step] = copy.deepcopy(head).cpu().eval()
    return heads


def scores_of(head: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return head(x).squeeze(1).numpy()


def oracle_cv(x: np.ndarray, labels: np.ndarray, videos: list[str], anomaly: np.ndarray,
              weight_decay: float, steps: int) -> float:
    """Out-of-fold in-video AUC on the test videos; same folds as train_in_video.oracle_cv."""
    names = sorted(set(videos))
    fold_of = {name: i % FOLDS for i, name in enumerate(np.random.default_rng(SEED).permutation(names))}
    folds = np.array([fold_of[video] for video in videos])
    out = np.empty(len(videos), dtype=np.float32)
    for fold in range(FOLDS):
        held = folds == fold
        fit_x, held_x = standardize(x[~held], x[held])
        out[held] = scores_of(fit_mlp(fit_x, labels[~held], weight_decay)[steps], held_x)
    return in_video_auc(out, labels, anomaly)


def diagnose(name: str) -> dict:
    fit = json.loads((RUN / "analysis-pairwise" / name / "fit.json").read_text())
    features, kind = Path(fit["features"]), fit["kinds"][0]
    state, _ = chosen(fit, 0.0)
    meta = read_meta(features)
    train_meta = meta["train_anomaly"] + meta["train_normal"]
    is_val = np.array([row["video"] in set(fit["val_videos"]) for row in train_meta])
    labels = np.array([row["holmes"] for row in train_meta], dtype=np.float32)
    anomaly = np.array([row["category"] != "Normal" for row in train_meta])
    test_videos = [row["video"] for row in meta["test"]]
    test_labels = np.array([row["label"] for row in meta["test"]], dtype=np.float32)
    test_anomaly = np.array([row["category"] != "Normal" for row in meta["test"]])
    arrays = {s: np.asarray(np.load(features / s / f"{kind}.npy", mmap_mode="r")[:, state], dtype=np.float32)
              for s in ("train_anomaly", "train_normal", "test")}
    train = np.concatenate([arrays["train_anomaly"], arrays["train_normal"]])

    # Choose weight decay and step count on the validation videos.
    fit_x, val_x = standardize(train[~is_val], train[is_val])
    grid = {}
    for weight_decay in WEIGHT_DECAYS:
        for steps, head in fit_mlp(fit_x, labels[~is_val], weight_decay).items():
            grid[f"{weight_decay:g}/{steps}"] = in_video_auc(scores_of(head, val_x), labels[is_val], anomaly[is_val])
    best = max(grid, key=grid.get)
    weight_decay, steps = float(best.split("/")[0]), int(best.split("/")[1])

    all_x, test_x = standardize(train, arrays["test"])
    _, train_x = standardize(train, train)
    head = fit_mlp(all_x, labels, weight_decay)[steps]
    result = {
        "state": state, "weight_decay": weight_decay, "steps": steps, "val_grid": grid,
        "train": in_video_auc(scores_of(head, train_x), labels, anomaly),
        "val": grid[best],
        "test": in_video_auc(scores_of(head, test_x), test_labels, test_anomaly),
        "oracle_cv": oracle_cv(arrays["test"], test_labels, test_videos, test_anomaly, weight_decay, steps),
    }
    print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in result.items() if k != "val_grid"},
          flush=True)
    return result


def main() -> None:
    torch.cuda.set_per_process_memory_fraction(0.2)
    results = {name: diagnose(name) for name in FITS}
    (RUN / "analysis-pairwise" / "mlp-in-video.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
