"""Linear head trained with window BCE + a same-video pairwise ranking loss, on raw or [feature, residual] features.

Question: with BCE alone the head mostly learns "is this an anomaly video" (residual_fit.py: in-video AUC 0.62-0.65).
A pairwise loss inside each train anomaly video asks directly that event windows score above the other windows of
the same video:
  loss = BCE(window scores, Holmes labels; pos_weight as probe.fit.fit_window)
         + alpha * mean over videos of mean over (event, non-event) pairs of softplus(s_neg - s_pos)
         + lam * |w|^2
Event windows: Holmes label > 0.5; non-event windows: Holmes label 0; only train anomaly videos with both give pairs.
For each state and alpha in ALPHAS, lam in probe.fit.LAMS is chosen on the validation videos (probe.fit.split) by the
validation in-video AUC (anomaly videos only, Holmes labels); then the head is refit on all train windows. Each alpha
is one "method" pair_a<alpha> in fit.json, so alpha 0 is the BCE-only head under the same selection rule.
Usage: uv run python <this file> <features run dir> <out name>
Writes run-1/analysis-pairwise/<out name>/ (fit.json, test_scores.npy, test_meta.jsonl in the probe.fit format) and
prints the report of centered_fit.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from centered_fit import report
from s1stream.probe.fit import LAMS, LR, SEED, SETS, STEPS, bags, split, standardize

RUN = Path(__file__).resolve().parents[1]
ALPHAS = (0.0, 1.0, 4.0)
KIND = "patch_mean"


def read_meta(features: Path) -> dict[str, list[dict]]:
    return {name: [json.loads(line) for line in (features / name / "meta.jsonl").read_text().splitlines()]
            for name in SETS}


def pair_masks(videos: list[str], labels: np.ndarray, anomaly: np.ndarray):
    """(index [V, L], event mask [V, L], non-event mask [V, L]) for the anomaly videos that have both kinds."""
    index, mask, _ = bags(videos)
    y = torch.tensor(labels, dtype=torch.float32)[index]
    is_anomaly = torch.tensor(anomaly)[index] & mask
    event = (y > 0.5) & is_anomaly
    other = (y == 0) & is_anomaly
    keep = event.any(dim=1) & other.any(dim=1)
    return index[keep], event[keep], other[keep]


def pairwise_loss(scores: torch.Tensor, index: torch.Tensor, event: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Mean over videos of the mean softplus(s_other - s_event) over all (event, other) pairs of the video."""
    s = scores[index]
    diff = s[:, None, :] - s[:, :, None]  # [V, event position, other position]: s_other - s_event
    pairs = event[:, :, None] & other[:, None, :]
    per_video = (torch.nn.functional.softplus(diff) * pairs).sum(dim=(1, 2)) / pairs.sum(dim=(1, 2))
    return per_video.mean()


def fit_head(x: torch.Tensor, y: torch.Tensor, pairs: tuple, alpha: float, lam: float,
             device: str) -> torch.nn.Linear:
    """Zero-initialized Linear(hidden, 1), Adam lr LR, STEPS full-batch steps (as probe.fit._fit_linear)."""
    x, y = x.to(device), y.to(device)
    index, event, other = (t.to(device) for t in pairs)
    pos = y.mean().clamp(min=1e-6)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).detach())
    torch.manual_seed(SEED)
    head = torch.nn.Linear(x.shape[1], 1).to(device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    for _ in range(STEPS):
        optimizer.zero_grad()
        scores = head(x).squeeze(1)
        loss = bce(scores, y) + lam * head.weight.square().sum()
        if alpha > 0:
            loss = loss + alpha * pairwise_loss(scores, index, event, other)
        loss.backward()
        optimizer.step()
    return head.cpu()


def in_video_auc(scores: np.ndarray, labels: np.ndarray, anomaly: np.ndarray) -> float:
    """Window AUC on the anomaly videos only, event = Holmes label > 0.5."""
    return float(roc_auc_score(labels[anomaly] > 0.5, scores[anomaly]))


def main(features: str, name: str, device: str = "cuda") -> None:
    features_dir = Path(features)
    out = RUN / "analysis-pairwise" / name
    if (out / "fit.json").exists():
        report(out, KIND)
        return
    torch.cuda.set_per_process_memory_fraction(0.2)
    meta = read_meta(features_dir)
    train_meta = meta["train_anomaly"] + meta["train_normal"]
    videos = [row["video"] for row in train_meta]
    val = split(videos, {row["video"]: row["category"] for row in train_meta})
    is_val = np.array([video in val for video in videos])
    labels = np.array([row["holmes"] for row in train_meta], dtype=np.float32)
    anomaly = np.array([row["category"] != "Normal" for row in train_meta])
    y = torch.tensor(labels)
    fit_videos = [video for video, keep in zip(videos, is_val, strict=True) if not keep]
    fit_pairs = pair_masks(fit_videos, labels[~is_val], anomaly[~is_val])
    all_pairs = pair_masks(videos, labels, anomaly)

    arrays = {set_name: np.load(features_dir / set_name / f"{KIND}.npy", mmap_mode="r") for set_name in SETS}
    n_states = arrays["test"].shape[1]
    scores = np.empty((len(ALPHAS), 1, n_states, len(meta["test"])), dtype=np.float32)
    choices = {}
    for state in range(n_states):
        train = np.concatenate([np.asarray(arrays[s][:, state], dtype=np.float32)
                                for s in ("train_anomaly", "train_normal")])
        test = np.asarray(arrays["test"][:, state], dtype=np.float32)
        fit_x, val_x = standardize(train[~is_val], train[is_val])
        all_x, test_x = standardize(train, test)
        for alpha_i, alpha in enumerate(ALPHAS):
            best = (-np.inf, LAMS[0])
            for lam in LAMS:
                head = fit_head(fit_x, y[~is_val], fit_pairs, alpha, lam, device)
                with torch.no_grad():
                    value = in_video_auc(head(val_x).squeeze(1).numpy(), labels[is_val], anomaly[is_val])
                best = max(best, (value, lam))
            choices[f"pair_a{alpha:g}/{KIND}/{state}"] = {"setting": {"alpha": alpha, "lam": best[1]},
                                                         "validation": best[0]}
            head = fit_head(all_x, y, all_pairs, alpha, best[1], device)
            with torch.no_grad():
                scores[alpha_i, 0, state] = head(test_x).squeeze(1).numpy()
        print(f"state {state} done", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "test_scores.npy", scores)
    (out / "test_meta.jsonl").write_text((features_dir / "test" / "meta.jsonl").read_text())
    fit = {"methods": [f"pair_a{alpha:g}" for alpha in ALPHAS], "kinds": [KIND], "n_states": n_states,
           "features": str(features_dir), "val_videos": sorted(val), "choices": choices}
    (out / "fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    report(out, KIND)


if __name__ == "__main__":
    main(*sys.argv[1:])
