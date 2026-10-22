"""CPU stage of probing: `s1stream probe-fit` (finetune-layer-probing decision 10).

Input: one features run of probe-extract (<set>/last.npy, <set>/visual.npy, <set>/meta.jsonl for train_anomaly,
train_normal and test). For every token kind (KINDS) and state (0 .. n_states - 1) three probes give one score per
test window:
  mil:    standardized features -> Linear(hidden, 1); video score = mean of the top-k window scores; BCE on the video
          labels + lam * |w|^2. k in KS, lam in LAMS; the pair with the lowest validation video BCE is chosen.
  holmes: the same head trained per window on the Holmes labels (BCE with pos_weight = negatives / positives);
          lam in LAMS with the highest validation window AUC on the Holmes labels is chosen.
  knn:    mean distance to the K_NEIGHBORS nearest train-normal windows after PCA (min(PCA_DIM, hidden, n) components)
          fit on train-normal windows. No setting to choose; its validation score is the validation video AUC with
          video score = mean of the top-3 window scores.
Validation: VAL_SHARE of the train videos of each category (split, seed). Choices read only train and validation
windows. The chosen setting is then refit on all train windows and scores the test windows.

Writes into out:
  fit.json          {"methods": METHODS, "kinds": KINDS, "n_states": S, "features": <features dir>,
                     "val_videos": [...], "choices": {"<method>/<kind>/<state>": {"setting": {...}, "validation": x}}}
  test_scores.npy   float32 (len(METHODS), len(KINDS), S, n_test), in METHODS and KINDS order
  test_meta.jsonl   the test meta.jsonl of the features run, unchanged
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

METHODS = ("mil", "holmes", "knn")
KINDS = ("last", "visual")
SETS = ("train_anomaly", "train_normal", "test")
KS = ("1", "3", "frac8")
LAMS = (1e-4, 1e-3, 1e-2)
STEPS, LR, SEED, VAL_SHARE, TOP = 200, 1e-2, 20261022, 0.2, 3
K_NEIGHBORS, PCA_DIM = 5, 256


def bags(videos: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """(index [V, L] of window rows, mask [V, L], video names in first-seen order); padding entries point at row 0."""
    raise NotImplementedError


def topk_pool(scores: torch.Tensor, index: torch.Tensor, mask: torch.Tensor, k: str) -> torch.Tensor:
    """Mean of the top-k window scores of each bag; scores [N] -> [V]. k "1" or "3": that many, capped by the bag
    size; "frac8": ceil(bag size / 8), at least 1."""
    raise NotImplementedError


def standardize(train: np.ndarray, *others: np.ndarray) -> list[torch.Tensor]:
    """(a - mean) / (std + 1e-4) with mean and std of train, for train and each of others; float32 tensors."""
    raise NotImplementedError


def fit_mil(x: torch.Tensor, video_y: torch.Tensor, index: torch.Tensor, mask: torch.Tensor, k: str,
            lam: float) -> torch.nn.Linear:
    """Zero-initialized Linear(x.shape[1], 1), Adam lr LR, STEPS full-batch steps, torch.manual_seed(SEED); loss =
    BCE-with-logits(topk_pool(head(x), index, mask, k), video_y) + lam * |w|^2."""
    raise NotImplementedError


def fit_window(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.nn.Linear:
    """As fit_mil, but the loss is the per-window BCE-with-logits with pos_weight = (1 - mean(y)) / mean(y)."""
    raise NotImplementedError


def knn_scores(train_normal: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mean distance of each row of x to its K_NEIGHBORS nearest rows of train_normal after a PCA fit on
    train_normal (random_state 0)."""
    raise NotImplementedError


def split(videos: list[str], category: dict[str, str], seed: int = SEED) -> set[str]:
    """Validation videos: for each category (sorted), random.Random(seed).sample of max(1, round(VAL_SHARE * n)) of
    its name-sorted videos."""
    raise NotImplementedError


def fit_condition(features: Path, out: Path, seed: int = SEED, log: Callable[[str], None] = print) -> None:
    """Fit every method, kind and state on one features run and write fit.json, test_scores.npy and test_meta.jsonl
    into out (see the module docstring). Raise FileExistsError when out/fit.json exists. Load one state at a time
    (np.load with mmap_mode="r") and cast it to float32."""
    raise NotImplementedError
