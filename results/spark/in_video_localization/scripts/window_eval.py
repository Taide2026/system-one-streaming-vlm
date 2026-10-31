"""Shared helpers for the temporal-difference and motion analyses (tempdiff_fit.py, motion_fit.py).

evaluate(): the test report of one score per test window: frame-weighted all / in_video / per-video median (as
centered_fit.report) and the event-phase AUC against normal videos (event_phase_auc.phase_auc).
fit_bce(): the window BCE linear head of pairwise_fit.fit_head (alpha 0), with lam chosen on the validation videos
by the validation in-video AUC, then refit on all train windows.
previous_rows(): for each window, the row of the previous window of the same video (-1 for the first window).
"""

import json
from pathlib import Path

import numpy as np
import torch

from paths import ENCODER_RUN
from event_phase_auc import phase_auc
from pairwise_fit import fit_head, in_video_auc
from s1stream.encoder.extract import load_encoder_config
from s1stream.probe.compare import _weighted_curve, _window_counts
from s1stream.probe.fit import LAMS, standardize
from s1stream.registry import DATASETS

NO_PAIRS = (torch.zeros((0, 1), dtype=torch.long), torch.zeros((0, 1), dtype=torch.bool),
            torch.zeros((0, 1), dtype=torch.bool))


def test_items() -> dict:
    return {item.name: item for item in DATASETS.build(load_encoder_config(ENCODER_RUN / "config.yaml").test).items()}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def previous_rows(meta: list[dict]) -> np.ndarray:
    """Row index of the window that ends where this one starts in the same video, or -1."""
    by_end = {(row["video"], round(row["end_s"], 3)): i for i, row in enumerate(meta)}
    return np.array([by_end.get((row["video"], round(row["start_s"], 3)), -1) for row in meta])


def evaluate(values: np.ndarray, meta: list[dict], items: dict) -> dict:
    values = np.asarray(values, dtype=np.float64)
    positive, negative = _window_counts(meta, items)
    anomaly = np.array([row["category"] != "Normal" for row in meta])
    video = np.array([row["video"] for row in meta])
    per_video = []
    for name in dict.fromkeys(video[anomaly]):
        keep = video == name
        if positive[keep].sum() and negative[keep].sum():
            per_video.append(_weighted_curve(values[keep], positive[keep][None], negative[keep][None])[0][0])
    phases = phase_auc(meta, values, items)
    return {
        "all": float(_weighted_curve(values, positive[None], negative[None])[0][0]),
        "in_video": float(_weighted_curve(values, (positive * anomaly)[None], (negative * anomaly)[None])[0][0]),
        "per_video_median": float(np.median(per_video)),
        "per_video_below_half": int((np.array(per_video) < 0.5).sum()),
        "phase": {name: phases[name]["auc_vs_normal"] for name in ("before", "during", "after")},
        "during_minus_before": phases["during"]["auc_vs_normal"] - phases["before"]["auc_vs_normal"],
    }


def line(name: str, result: dict) -> str:
    phase = result["phase"]
    return (f"{name:40s} all {result['all']:.3f}  in_video {result['in_video']:.3f}  "
            f"per-video median {result['per_video_median']:.3f}  before {phase['before']:.3f}  "
            f"during {phase['during']:.3f}  after {phase['after']:.3f}  gap {result['during_minus_before']:.3f}")


def fit_bce(train: np.ndarray, test: np.ndarray, labels: np.ndarray, anomaly: np.ndarray, is_val: np.ndarray,
            device: str) -> tuple[float, float, np.ndarray]:
    """(validation in-video AUC, lam, test scores) of the BCE head on one feature matrix."""
    fit_x, val_x = standardize(train[~is_val], train[is_val])
    all_x, test_x = standardize(train, test)
    y = torch.tensor(labels)
    best = (-np.inf, LAMS[0])
    for lam in LAMS:
        head = fit_head(fit_x, y[~is_val], NO_PAIRS, 0.0, lam, device)
        with torch.no_grad():
            value = in_video_auc(head(val_x).squeeze(1).numpy(), labels[is_val], anomaly[is_val])
        best = max(best, (value, lam))
    head = fit_head(all_x, y, NO_PAIRS, 0.0, best[1], device)
    with torch.no_grad():
        return best[0], best[1], head(test_x).squeeze(1).numpy()
