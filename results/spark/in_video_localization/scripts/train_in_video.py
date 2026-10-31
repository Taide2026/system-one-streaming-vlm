"""Diagnostic: is the in-video AUC of the linear head limited by fitting or by generalization?

Question: the pairwise fits (pairwise_fit.py) stay at in-video AUC 0.6-0.68 on test. If the head also stays near that
on its own train windows, a linear head on these features cannot rank windows inside a video (underfit). If train is
much higher, the head fits train but does not transfer (generalization; Holmes label noise, few videos).
For each fit in 04-loss-changes/pairwise/ and alpha 0 and 1, with the state and lam chosen in its fit.json:
  train    head fit on all train windows, scored on the train anomaly videos (Holmes labels)
  val      head fit on the fit split, scored on the validation anomaly videos (Holmes labels)
  test     head fit on all train windows, scored on the test anomaly videos (ground-truth labels)
  oracle   head fit on the test windows with ground-truth labels, scored on the same windows (linear ceiling)
  oracle_cv  as oracle, but 5 folds split by test video; each fold scored by a head that did not see it
Usage: uv run python <this file>
All five values are window-level AUC (label > 0.5), so test differs by up to 0.03 from the frame-weighted report.
Writes 02-generalization/train-in-video.json and prints one line per fit and alpha.
"""

import json
from pathlib import Path

import numpy as np
import torch

from paths import GENERALIZATION, LOSS_CHANGES
from pairwise_fit import fit_head, in_video_auc, pair_masks, read_meta
from s1stream.probe.fit import SEED, standardize

FITS = ("vjepa2-vitl-raw", "vjepa2-vitl-residual-past10", "qwen35-vit-raw", "qwen35-vit-residual-past10",
        "qwen35-9b-A-last-raw", "qwen35-9b-A-last-residual-past10",
        "qwen35-9b-A-visual-raw", "qwen35-9b-A-visual-residual-past10")
ALPHAS = (0.0, 1.0)
FOLDS = 5
DEVICE = "cuda"


def chosen(fit: dict, alpha: float) -> tuple[int, float]:
    """(state, lam) with the best validation in-video AUC for this alpha, as the report picks it."""
    kind = fit["kinds"][0]
    rows = [(fit["choices"][f"pair_a{alpha:g}/{kind}/{state}"], state) for state in range(fit["n_states"])]
    best, state = max(rows, key=lambda row: row[0]["validation"])
    return state, best["setting"]["lam"]


def scores_of(head: torch.nn.Linear, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return head(x).squeeze(1).numpy()


def fit_and_score(fit_x, fit_y, fit_videos, fit_anomaly, score_x, alpha, lam) -> np.ndarray:
    """Fit one head on (fit_x, fit_y) with its own same-video pairs, return its scores on score_x."""
    pairs = pair_masks(fit_videos, fit_y, fit_anomaly)
    head = fit_head(fit_x, torch.tensor(fit_y), pairs, alpha, lam, DEVICE)
    return scores_of(head, score_x)


def oracle_cv(x: np.ndarray, labels: np.ndarray, videos: list[str], anomaly: np.ndarray, alpha, lam) -> float:
    """In-video AUC of out-of-fold scores; folds split the test videos (round robin over sorted names)."""
    names = sorted(set(videos))
    fold_of = {name: i % FOLDS for i, name in enumerate(np.random.default_rng(SEED).permutation(names))}
    folds = np.array([fold_of[video] for video in videos])
    out = np.empty(len(videos), dtype=np.float32)
    for fold in range(FOLDS):
        held = folds == fold
        fit_x, held_x = standardize(x[~held], x[held])
        fit_videos = [video for video, keep in zip(videos, held, strict=True) if not keep]
        out[held] = fit_and_score(fit_x, labels[~held], fit_videos, anomaly[~held], held_x, alpha, lam)
    return in_video_auc(out, labels, anomaly)


def diagnose(name: str) -> dict:
    fit = json.loads((LOSS_CHANGES / "pairwise" / name / "fit.json").read_text())
    features, kind = Path(fit["features"]), fit["kinds"][0]
    meta = read_meta(features)
    train_meta = meta["train_anomaly"] + meta["train_normal"]
    videos = [row["video"] for row in train_meta]
    is_val = np.array([video in set(fit["val_videos"]) for video in videos])
    labels = np.array([row["holmes"] for row in train_meta], dtype=np.float32)
    anomaly = np.array([row["category"] != "Normal" for row in train_meta])
    test_videos = [row["video"] for row in meta["test"]]
    test_labels = np.array([row["label"] for row in meta["test"]], dtype=np.float32)
    test_anomaly = np.array([row["category"] != "Normal" for row in meta["test"]])
    fit_videos = [video for video, keep in zip(videos, is_val, strict=True) if not keep]

    result = {}
    for alpha in ALPHAS:
        state, lam = chosen(fit, alpha)
        arrays = {s: np.asarray(np.load(features / s / f"{kind}.npy", mmap_mode="r")[:, state], dtype=np.float32)
                  for s in ("train_anomaly", "train_normal", "test")}
        train = np.concatenate([arrays["train_anomaly"], arrays["train_normal"]])
        test = arrays["test"]

        all_x, test_x = standardize(train, test)
        _, train_x = standardize(train, train)
        pairs = pair_masks(videos, labels, anomaly)
        head = fit_head(all_x, torch.tensor(labels), pairs, alpha, lam, DEVICE)
        fit_x, val_x = standardize(train[~is_val], train[is_val])
        val_scores = fit_and_score(fit_x, labels[~is_val], fit_videos, anomaly[~is_val], val_x, alpha, lam)
        own_x, _ = standardize(test, test)
        oracle_scores = fit_and_score(own_x, test_labels, test_videos, test_anomaly, own_x, alpha, lam)

        result[f"a{alpha:g}"] = {
            "state": state, "lam": lam,
            "train": in_video_auc(scores_of(head, train_x), labels, anomaly),
            "val": in_video_auc(val_scores, labels[is_val], anomaly[is_val]),
            "test": in_video_auc(scores_of(head, test_x), test_labels, test_anomaly),
            "oracle": in_video_auc(oracle_scores, test_labels, test_anomaly),
            "oracle_cv": oracle_cv(test, test_labels, test_videos, test_anomaly, alpha, lam),
        }
        print(name, f"a{alpha:g}", {k: round(v, 3) if isinstance(v, float) else v
                                    for k, v in result[f"a{alpha:g}"].items()}, flush=True)
    return result


def main() -> None:
    torch.cuda.set_per_process_memory_fraction(0.2)
    results = {name: diagnose(name) for name in FITS}
    (GENERALIZATION / "train-in-video.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
