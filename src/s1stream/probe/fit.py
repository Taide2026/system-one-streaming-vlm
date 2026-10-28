"""Probe stage of probing: `s1stream probe-fit` (finetune-layer-probing decision 10).

Input: one features run of probe-extract (<set>/<kind>.npy and <set>/meta.jsonl for train_anomaly, train_normal and
test, with token kinds from read_kinds). For every token kind and state (0 .. n_states - 1) three probes give one score
per test window:
  mil:    standardized features -> Linear(hidden, 1); video score = mean of the top-k window scores; BCE on the video
          labels + lam * |w|^2. k in KS, lam in LAMS; the pair with the lowest validation video BCE is chosen.
  holmes: the same head trained per window on the Holmes labels (BCE with pos_weight = negatives / positives);
          lam in LAMS with the highest validation window AUC on the Holmes labels is chosen.
  knn:    mean distance to the K_NEIGHBORS nearest train-normal windows after PCA (min(PCA_DIM, hidden, n) components)
          fit on train-normal windows. No setting to choose; its validation score is the validation video AUC with
          video score = mean of the top-3 window scores.
Validation: VAL_SHARE of the train videos of each category (split, seed). Choices read only train and validation
windows. The chosen setting is then refit on all train windows and scores the test windows.
mil and holmes heads train on `device` ("cpu" or "cuda"): Adam lr LR, STEPS full-batch steps (run-2 setting; run-1
used lr 1e-2 and 200 steps, which oscillated on condition C). knn always runs on the CPU with scikit-learn.

Writes into out:
  fit.json          {"methods": METHODS, "kinds": list(kinds), "n_states": S, "features": <features dir>,
                     "val_videos": [...], "choices": {"<method>/<kind>/<state>": {"setting": {...}, "validation": x}}}
                    mil and holmes choices also hold "final_loss" and "min_loss": the training loss of the last
                    step (before its update) and the lowest training loss over all steps of the head refit on all train windows
  test_scores.npy   float32 (len(METHODS), len(kinds), S, n_test), in METHODS and kinds order
  test_meta.jsonl   the test meta.jsonl of the features run, unchanged
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

METHODS = ("mil", "holmes", "knn")
KINDS = ("last", "visual")
SETS = ("train_anomaly", "train_normal", "test")
KS = ("1", "3", "frac8")
LAMS = (1e-4, 1e-3, 1e-2)
STEPS, LR, SEED, VAL_SHARE, TOP = 1000, 1e-3, 20261022, 0.2, 3
K_NEIGHBORS, PCA_DIM = 5, 256


class Fitted(NamedTuple):
    """A trained head on the CPU, the training loss (data loss + lam * |w|^2) of the last step (computed before that
    step's update), and the lowest training loss over all STEPS steps."""
    head: torch.nn.Linear
    final_loss: float
    min_loss: float


def bags(videos: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """(index [V, L] of window rows, mask [V, L], video names in first-seen order); padding entries point at row 0."""
    order: list[str] = []
    rows: dict[str, list[int]] = defaultdict(list)
    for i, video in enumerate(videos):
        if video not in rows:
            order.append(video)
        rows[video].append(i)
    width = max(map(len, rows.values()))
    index = torch.zeros((len(order), width), dtype=torch.long)
    mask = torch.zeros((len(order), width), dtype=torch.bool)
    for i, video in enumerate(order):
        index[i, :len(rows[video])] = torch.tensor(rows[video])
        mask[i, :len(rows[video])] = True
    return index, mask, order


def topk_pool(scores: torch.Tensor, index: torch.Tensor, mask: torch.Tensor, k: str) -> torch.Tensor:
    """Mean of the top-k window scores of each bag; scores [N] -> [V]. k "1" or "3": that many, capped by the bag
    size; "frac8": ceil(bag size / 8), at least 1."""
    ordered = scores[index].masked_fill(~mask, -torch.inf).sort(dim=1, descending=True).values
    sizes = mask.sum(dim=1)
    if k == "frac8":
        counts = (sizes + 7) // 8
    else:
        counts = sizes.clamp(max=int(k))
    keep = torch.arange(ordered.shape[1], device=ordered.device)[None, :] < counts[:, None]
    return torch.where(keep, ordered, 0).sum(dim=1) / counts


def standardize(train: np.ndarray, *others: np.ndarray) -> list[torch.Tensor]:
    """(a - mean) / (std + 1e-4) with mean and std of train, for train and each of others; float32 tensors."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    return [torch.tensor((a - mean) / (std + 1e-4), dtype=torch.float32) for a in (train, *others)]


def _fit_linear(x: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor], lam: float,
                device: str) -> Fitted:
    x = x.to(device)
    torch.manual_seed(SEED)
    head = torch.nn.Linear(x.shape[1], 1).to(device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    min_loss = float("inf")
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = loss_fn(head(x).squeeze(1)) + lam * head.weight.square().sum()
        final_loss = float(loss.detach())
        min_loss = min(min_loss, final_loss)
        loss.backward()
        optimizer.step()
    return Fitted(head.cpu(), final_loss, min_loss)


def fit_mil(x: torch.Tensor, video_y: torch.Tensor, index: torch.Tensor, mask: torch.Tensor, k: str,
            lam: float, device: str = "cpu") -> Fitted:
    """Zero-initialized Linear(x.shape[1], 1), Adam lr LR, STEPS full-batch steps, torch.manual_seed(SEED); loss =
    BCE-with-logits(topk_pool(head(x), index, mask, k), video_y) + lam * |w|^2. The inputs (CPU tensors) move to
    device for training; the returned head is on the CPU."""
    video_y, index, mask = video_y.to(device), index.to(device), mask.to(device)
    bce = torch.nn.BCEWithLogitsLoss()
    return _fit_linear(x, lambda scores: bce(topk_pool(scores, index, mask, k), video_y), lam, device)


def fit_window(x: torch.Tensor, y: torch.Tensor, lam: float, device: str = "cpu") -> Fitted:
    """As fit_mil, but the loss is the per-window BCE-with-logits with pos_weight = (1 - mean(y)) / mean(y)."""
    y = y.to(device)
    pos = y.mean().clamp(min=1e-6)
    pos_weight = ((1 - pos) / pos).detach()
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return _fit_linear(x, lambda scores: bce(scores, y), lam, device)


def knn_scores(train_normal: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mean distance of each row of x to its K_NEIGHBORS nearest rows of train_normal after a PCA fit on
    train_normal (random_state 0)."""
    components = min(PCA_DIM, train_normal.shape[1], len(train_normal))
    pca = PCA(n_components=components, random_state=0).fit(train_normal)
    neighbors = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, len(train_normal)))
    neighbors.fit(pca.transform(train_normal))
    return neighbors.kneighbors(pca.transform(x))[0].mean(axis=1)


def split(videos: list[str], category: dict[str, str], seed: int = SEED) -> set[str]:
    """Validation videos: for each category (sorted), random.Random(seed).sample of max(1, round(VAL_SHARE * n)) of
    its name-sorted videos."""
    groups: dict[str, list[str]] = defaultdict(list)
    for video in sorted(set(videos)):
        groups[category[video]].append(video)
    val: set[str] = set()
    for name in sorted(groups):
        group = groups[name]
        val.update(random.Random(seed).sample(group, max(1, round(VAL_SHARE * len(group)))))
    return val


def read_kinds(features: Path) -> tuple[str, ...]:
    """Return run.json token kinds when they are a list of strings, else KINDS.

    A missing run.json, a run.json without "kinds", or kinds as a dict uses KINDS.
    """
    run_path = features / "run.json"
    if not run_path.exists():
        return KINDS
    kinds = json.loads(run_path.read_text()).get("kinds")
    if isinstance(kinds, list) and all(isinstance(kind, str) for kind in kinds):
        return tuple(kinds)
    return KINDS


def fit_condition(features: Path, out: Path, seed: int = SEED, log: Callable[[str], None] = print,
                  device: str = "cpu") -> None:
    """Fit every method, kind (from read_kinds) and state on one features run and write fit.json, test_scores.npy and
    test_meta.jsonl into out (see the module docstring); fit.json "kinds" and the kind axis of test_scores.npy follow
    that list. Raise FileExistsError when out/fit.json exists. A kind whose state count differs from the first kind
    raises ValueError naming that kind. Load one state at a time (np.load with mmap_mode="r") and cast it to float32.
    mil and holmes heads train on device (fit_mil, fit_window); test scores are float32 numpy arrays on the CPU. knn
    does not depend on device."""
    if (out / "fit.json").exists():
        raise FileExistsError(out / "fit.json")
    kinds = read_kinds(features)
    meta_text = {name: (features / name / "meta.jsonl").read_text() for name in SETS}
    meta = {name: [json.loads(line) for line in text.splitlines()] for name, text in meta_text.items()}
    train_meta = meta["train_anomaly"] + meta["train_normal"]
    videos = [row["video"] for row in train_meta]
    video_y = {row["video"]: float(row["category"] != "Normal") for row in train_meta}
    val = split(videos, {row["video"]: row["category"] for row in train_meta}, seed)
    is_val = np.array([video in val for video in videos])
    normal = np.array([row["category"] == "Normal" for row in train_meta])
    holmes = torch.tensor([row["holmes"] for row in train_meta], dtype=torch.float32)

    def bag_part(selected: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index, mask, order = bags([video for video, keep in zip(videos, selected, strict=True) if keep])
        return index, mask, torch.tensor([video_y[name] for name in order], dtype=torch.float32)

    fit_index, fit_mask, fit_labels = bag_part(~is_val)
    val_index, val_mask, val_labels = bag_part(is_val)
    all_index, all_mask, all_labels = bag_part(np.ones(len(videos), dtype=bool))
    val_videos = [video for video, keep in zip(videos, is_val, strict=True) if keep]
    bce = torch.nn.BCEWithLogitsLoss()
    arrays = {kind: {name: np.load(features / name / f"{kind}.npy", mmap_mode="r") for name in SETS}
              for kind in kinds}
    n_states = arrays[kinds[0]]["test"].shape[1]
    for kind in kinds[1:]:
        if arrays[kind]["test"].shape[1] != n_states:
            raise ValueError(f"{kind} has a different state count")
    n_test = len(meta["test"])
    scores = np.empty((len(METHODS), len(kinds), n_states, n_test), dtype=np.float32)
    choices: dict[str, dict] = {}
    for kind_i, kind in enumerate(kinds):
        for state in range(n_states):
            started = time.monotonic()
            train = np.concatenate([np.asarray(arrays[kind][name][:, state], dtype=np.float32)
                                    for name in ("train_anomaly", "train_normal")])
            fit_x, val_x = standardize(train[~is_val], train[is_val])
            best_mil = (float("inf"), "1", LAMS[0])
            for k in KS:
                for lam in LAMS:
                    head = fit_mil(fit_x, fit_labels, fit_index, fit_mask, k, lam, device=device).head
                    with torch.no_grad():
                        pooled = topk_pool(head(val_x).squeeze(1), val_index, val_mask, k)
                        value = float(bce(pooled, val_labels))
                    if value < best_mil[0]:
                        best_mil = (value, k, lam)
            choices[f"mil/{kind}/{state}"] = {"setting": {"k": best_mil[1], "lam": best_mil[2]},
                                              "validation": best_mil[0]}

            best_holmes = (-float("inf"), LAMS[0])
            for lam in LAMS:
                head = fit_window(fit_x, holmes[~is_val], lam, device=device).head
                with torch.no_grad():
                    val_scores = head(val_x).squeeze(1).numpy()
                value = float(roc_auc_score(holmes[is_val].numpy(), val_scores))
                if value > best_holmes[0]:
                    best_holmes = (value, lam)
            choices[f"holmes/{kind}/{state}"] = {"setting": {"lam": best_holmes[1]},
                                                 "validation": best_holmes[0]}

            fit_normal = train[~is_val & normal]
            knn_val = knn_scores(fit_normal, train[is_val])
            vi, vm, order = bags(val_videos)
            pooled = topk_pool(torch.tensor(knn_val, dtype=torch.float32), vi, vm, str(TOP)).numpy()
            knn_auc = float(roc_auc_score([video_y[name] for name in order], pooled))
            n_normal = int(normal.sum())
            choices[f"knn/{kind}/{state}"] = {
                "setting": {"k_neighbors": K_NEIGHBORS, "pca_dim": min(PCA_DIM, train.shape[1], n_normal)},
                "validation": knn_auc}

            test = np.asarray(arrays[kind]["test"][:, state], dtype=np.float32)
            all_x, test_x = standardize(train, test)
            mil_head = fit_mil(all_x, all_labels, all_index, all_mask, best_mil[1], best_mil[2], device=device)
            holmes_head = fit_window(all_x, holmes, best_holmes[1], device=device)
            choices[f"mil/{kind}/{state}"].update(final_loss=mil_head.final_loss, min_loss=mil_head.min_loss)
            choices[f"holmes/{kind}/{state}"].update(final_loss=holmes_head.final_loss, min_loss=holmes_head.min_loss)
            with torch.no_grad():
                scores[0, kind_i, state] = mil_head.head(test_x).squeeze(1).numpy()
                scores[1, kind_i, state] = holmes_head.head(test_x).squeeze(1).numpy()
            scores[2, kind_i, state] = knn_scores(train[normal], test)
            log(f"{kind} state {state}: {time.monotonic() - started:.2f}s")

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "test_scores.npy", scores)
    (out / "test_meta.jsonl").write_text(meta_text["test"])
    result = {"methods": METHODS, "kinds": list(kinds), "n_states": n_states, "features": str(features),
              "val_videos": sorted(val), "choices": choices}
    (out / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
