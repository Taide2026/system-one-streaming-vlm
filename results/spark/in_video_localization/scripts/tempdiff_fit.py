"""Temporal difference of stored window features: does "what changed since the last window" localize the event?

Question: heads on the window feature x_t learn the scene (event_phase_auc.py, shortcut_fit.py). The difference
d_t = x_t - x_(t-1) to the previous 2 s window of the same video cancels what stays constant (the scene) and keeps
what changes. It uses only the current and the previous window, so it is causal. The first window of a video has no
previous window; its d_t is the zero vector (and its change score 0).
Inputs (per state):
  x            BCE head on x_t (the reference; same as shortcut_fit.py bce)
  diff         BCE head on d_t
  x_diff       BCE head on [x_t, d_t]
  change_l2    no training: |d_t| (L2 norm of the standardized difference)
  change_cos   no training: 1 - cos(x_t, x_(t-1))
Heads: window_eval.fit_bce (Holmes labels, lam on validation). For every input the state with the best validation
in-video AUC is used for the test report (window_eval.evaluate).
Usage: uv run python <this file> <features run dir> <out name> [kind]   (default patch_mean)
Writes in_video_localization/06-motion-tempdiff/tempdiff/<out name>.json and prints one line per input.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from paths import MOTION
from pairwise_fit import in_video_auc, read_meta
from s1stream.probe.fit import split
from window_eval import evaluate, fit_bce, line, previous_rows, test_items

INPUTS = ("x", "diff", "x_diff", "change_l2", "change_cos")


def difference(x: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """x_t - x_(t-1); zero where the window has no previous window."""
    out = x - x[np.maximum(previous, 0)]
    out[previous < 0] = 0
    return out


def change_scores(x: np.ndarray, previous: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    """Training-free change scores; mean and std (of the train windows) standardize x first."""
    z = (x - mean) / std
    prev = z[np.maximum(previous, 0)]
    cos = (z * prev).sum(1) / (np.linalg.norm(z, axis=1) * np.linalg.norm(prev, axis=1) + 1e-8)
    l2 = np.linalg.norm(z - prev, axis=1)
    first = previous < 0
    return {"change_l2": np.where(first, 0.0, l2), "change_cos": np.where(first, 0.0, 1 - cos)}


def main(features: str, name: str, kind: str = "patch_mean", device: str = "cuda") -> None:
    features_dir = Path(features)
    out = MOTION / "tempdiff" / f"{name}.json"
    if out.exists():
        raise FileExistsError(out)
    torch.cuda.set_per_process_memory_fraction(0.2)
    meta = read_meta(features_dir)
    train_meta = meta["train_anomaly"] + meta["train_normal"]
    videos = [row["video"] for row in train_meta]
    val = split(videos, {row["video"]: row["category"] for row in train_meta})
    is_val = np.array([video in val for video in videos])
    labels = np.array([row["holmes"] for row in train_meta], dtype=np.float32)
    anomaly = np.array([row["category"] != "Normal" for row in train_meta])
    train_prev, test_prev = previous_rows(train_meta), previous_rows(meta["test"])
    print(f"windows without a previous window: train {int((train_prev < 0).sum())}, "
          f"test {int((test_prev < 0).sum())}", flush=True)

    arrays = {s: np.load(features_dir / s / f"{kind}.npy", mmap_mode="r")
              for s in ("train_anomaly", "train_normal", "test")}
    n_states = arrays["test"].shape[1]
    best = {name: (-np.inf, None, None, None) for name in INPUTS}  # (validation, state, lam, test scores)
    for state in range(n_states):
        train = np.concatenate([np.asarray(arrays[s][:, state], dtype=np.float32)
                                for s in ("train_anomaly", "train_normal")])
        test = np.asarray(arrays["test"][:, state], dtype=np.float32)
        train_d, test_d = difference(train, train_prev), difference(test, test_prev)
        heads = {"x": (train, test), "diff": (train_d, test_d),
                 "x_diff": (np.concatenate([train, train_d], 1), np.concatenate([test, test_d], 1))}
        for input_name, (tr, te) in heads.items():
            value, lam, scores = fit_bce(tr, te, labels, anomaly, is_val, device)
            best[input_name] = max(best[input_name], (value, state, lam, scores), key=lambda b: b[0])
        mean, std = train.mean(0), train.std(0) + 1e-6
        train_change = change_scores(train, train_prev, mean, std)
        test_change = change_scores(test, test_prev, mean, std)
        for input_name in ("change_l2", "change_cos"):
            value = in_video_auc(train_change[input_name][is_val], labels[is_val], anomaly[is_val])
            best[input_name] = max(best[input_name], (value, state, None, test_change[input_name]),
                                   key=lambda b: b[0])
        print(f"state {state} done", flush=True)

    items = test_items()
    results = {}
    for input_name in INPUTS:
        value, state, lam, scores = best[input_name]
        results[input_name] = {"state": state, "lam": lam, "validation_in_video": value,
                               **evaluate(scores, meta["test"], items)}
        print(line(f"{name}/{input_name} state {state} val {value:.3f}", results[input_name]), flush=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"features": str(features_dir), "kind": kind, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
