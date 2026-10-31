"""Bias-expert (product-of-experts) training on stored window features: let a scene-only model absorb the shortcut.

Question: every head learns "this footage looks like an anomaly video". Subtracting the whole-video mean feature
raised the in-video AUC (0.668 -> 0.763, V-JEPA 2) but needs future windows at inference. Here the whole video is
used only during training: a bias model sees only the scene, and the main head is trained together with it, so the
main head gets gradient only where the scene does not explain the label (Clark et al. 2019, "Don't Take the Easy
Way Out"; He et al. 2019, DRiFt). At inference only the main head is used, on x_t alone, so it stays causal.
  bias model b   Linear on the mean feature of all windows of the video (same input for every window of a video),
                 window BCE with Holmes labels; fit first, then frozen
  main head f    Linear on x_t; loss = BCE(f(x_t) + c * b(video), y) + lam * |w|^2 (pos_weight as probe.fit)
  bce            the same head without the bias model (reference)
Methods: bce and poe_c<c> for c in SCALES. For each state and method, lam in probe.fit.LAMS is chosen on the
validation videos by the validation in-video AUC of f alone; the bias model is fit on the same windows as f (fit split
for validation, all train windows for test) with lam BIAS_LAM. Test report: f alone (window_eval.evaluate).
Usage: uv run python <this file> <features run dir> <out name> [kind]   (default patch_mean)
Writes in_video_localization/08-bias-expert/<out name>.json and prints one line per method.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from paths import BIAS_EXPERT
from pairwise_fit import in_video_auc, read_meta
from s1stream.probe.fit import LAMS, LR, SEED, STEPS, split, standardize
from window_eval import evaluate, line, test_items

SCALES = (0.5, 1.0, 2.0)
METHODS = ("bce", *(f"poe_c{c:g}" for c in SCALES))
BIAS_LAM = 1e-3


def video_means(x: np.ndarray, videos: list[str]) -> np.ndarray:
    """Each row replaced by the mean of all rows of its video."""
    names = np.array(videos)
    out = np.empty_like(x)
    for name in dict.fromkeys(videos):
        rows = names == name
        out[rows] = x[rows].mean(0)
    return out


def fit_linear(x: torch.Tensor, y: torch.Tensor, offset: torch.Tensor, lam: float, device: str) -> torch.nn.Linear:
    """Zero-initialized Linear(hidden, 1) on BCE(head(x) + offset, y) with pos_weight, Adam, STEPS full-batch steps."""
    x, y, offset = x.to(device), y.to(device), offset.to(device)
    pos = y.mean().clamp(min=1e-6)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).detach())
    torch.manual_seed(SEED)
    head = torch.nn.Linear(x.shape[1], 1).to(device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = bce(head(x).squeeze(1) + offset, y) + lam * head.weight.square().sum()
        loss.backward()
        optimizer.step()
    return head.cpu()


def scores_of(head: torch.nn.Linear, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return head(x).squeeze(1).numpy()


def bias_logits(x: np.ndarray, videos: list[str], labels: np.ndarray, device: str) -> np.ndarray:
    """The frozen bias model's logits on its own training windows (input: per-video mean feature)."""
    means = video_means(x, videos)
    z, _ = standardize(means, means[:1])
    y = torch.tensor(labels)
    head = fit_linear(z, y, torch.zeros(len(y)), BIAS_LAM, device)
    return scores_of(head, z)


def main(features: str, name: str, kind: str = "patch_mean", device: str = "cuda") -> None:
    features_dir = Path(features)
    out = BIAS_EXPERT / f"{name}.json"
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
    fit_videos = [v for v, keep in zip(videos, is_val, strict=True) if not keep]

    arrays = {s: np.load(features_dir / s / f"{kind}.npy", mmap_mode="r")
              for s in ("train_anomaly", "train_normal", "test")}
    n_states = arrays["test"].shape[1]
    best = {m: (-np.inf, None, None, None) for m in METHODS}  # (validation, state, lam, test scores)
    for state in range(n_states):
        train = np.concatenate([np.asarray(arrays[s][:, state], dtype=np.float32)
                                for s in ("train_anomaly", "train_normal")])
        test = np.asarray(arrays["test"][:, state], dtype=np.float32)
        fit_x, val_x = standardize(train[~is_val], train[is_val])
        all_x, test_x = standardize(train, test)
        bias = {"fit": torch.tensor(bias_logits(train[~is_val], fit_videos, labels[~is_val], device)),
                "all": torch.tensor(bias_logits(train, videos, labels, device))}
        y_fit, y_all = torch.tensor(labels[~is_val]), torch.tensor(labels)
        for method in METHODS:
            c = 0.0 if method == "bce" else float(method.removeprefix("poe_c"))
            choice = (-np.inf, LAMS[0])
            for lam in LAMS:
                head = fit_linear(fit_x, y_fit, c * bias["fit"], lam, device)
                value = in_video_auc(scores_of(head, val_x), labels[is_val], anomaly[is_val])
                choice = max(choice, (value, lam))
            if choice[0] > best[method][0]:
                head = fit_linear(all_x, y_all, c * bias["all"], choice[1], device)
                best[method] = (choice[0], state, choice[1], scores_of(head, test_x))
        print(f"state {state} done", flush=True)

    items = test_items()
    results = {}
    for method in METHODS:
        value, state, lam, scores = best[method]
        results[method] = {"state": state, "lam": lam, "validation_in_video": value,
                           **evaluate(scores, meta["test"], items)}
        print(line(f"{name}/{method} state {state} val {value:.3f}", results[method]), flush=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"features": str(features_dir), "kind": kind, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
