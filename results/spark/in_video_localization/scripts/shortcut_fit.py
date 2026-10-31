"""Two training signals against the scene shortcut, on stored window features (survey area10, items 1 and 2).

Question: event_phase_auc.py shows that windows BEFORE the event already score far above normal videos; heads learn
"this footage looks like an anomaly video". Can a training signal that targets those windows teach the event?
Methods (each one "method" in fit.json; linear head, Adam, STEPS full-batch steps as probe.fit):
  bce         window BCE with Holmes labels (pos_weight as probe.fit.fit_window); the reference
  before_a<a> bce + a * same-video pairwise loss, event windows against the BEFORE-event windows of the same train
              anomaly video only (pairwise_fit.py used all non-event windows, also the ones after the event)
              event = Holmes label > 0.5; before = label 0 and start before the first event window of the video
  jtt_up<k>   Just Train Twice (Liu et al., ICML 2021): stage 1 = the bce head; error set = training windows that
              stage 1 classifies wrong (logit > 0 vs label > 0.5); stage 2 = bce with weight k on the error set
For each state and method, lam in probe.fit.LAMS is chosen on the validation videos by the validation in-video AUC
(as pairwise_fit.py); the head is then refit on all train windows. The jtt stage 1 uses the lam that bce chose.
Test report: all / in_video / per-video median (centered_fit.report) and the event-phase AUC against normal videos
(event_phase_auc.phase_auc); the during-minus-before gap shows whether the shortcut shrank.
Usage: uv run python <this file> <features run dir> <out name> [kind]   (default patch_mean)
Writes in_video_localization/04-loss-changes/shortcut/<out name>/ (fit.json, test_scores.npy, test_meta.jsonl, report.json).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from paths import ENCODER_RUN, LOSS_CHANGES
from centered_fit import report
from event_phase_auc import phase_auc
from pairwise_fit import fit_head, in_video_auc, read_meta
from s1stream.encoder.compare import select_state
from s1stream.encoder.extract import load_encoder_config
from s1stream.probe.fit import LAMS, LR, SEED, SETS, STEPS, bags, split, standardize
from s1stream.registry import DATASETS

BEFORE_ALPHAS = (1.0, 4.0)
JTT_UPWEIGHTS = (5.0, 20.0)
METHODS = ("bce", *(f"before_a{a:g}" for a in BEFORE_ALPHAS), *(f"jtt_up{k:g}" for k in JTT_UPWEIGHTS))


def before_rows(videos: list[str], labels: np.ndarray, anomaly: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """True for the label-0 windows of an anomaly video that start before its first event window."""
    first_event: dict[str, float] = {}
    for video, label, start in zip(videos, labels, starts, strict=True):
        if label > 0.5:
            first_event[video] = min(first_event.get(video, np.inf), start)
    return np.array([bool(is_anomaly) and label == 0 and start < first_event.get(video, -np.inf)
                     for video, label, is_anomaly, start in zip(videos, labels, anomaly, starts, strict=True)])


def before_pairs(videos: list[str], labels: np.ndarray, anomaly: np.ndarray, starts: np.ndarray):
    """(index [V, L], event mask, before mask) for the anomaly videos with both event and before windows."""
    index, mask, _ = bags(videos)
    event = torch.tensor((labels > 0.5) & anomaly)[index] & mask
    before = torch.tensor(before_rows(videos, labels, anomaly, starts))[index] & mask
    keep = event.any(dim=1) & before.any(dim=1)
    return index[keep], event[keep], before[keep]


def fit_weighted(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, lam: float, device: str) -> torch.nn.Linear:
    """As pairwise_fit.fit_head with alpha 0, but with one BCE weight per window."""
    x, y, weight = x.to(device), y.to(device), weight.to(device)
    pos = y.mean().clamp(min=1e-6)
    bce = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=((1 - pos) / pos).detach())
    torch.manual_seed(SEED)
    head = torch.nn.Linear(x.shape[1], 1).to(device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = bce(head(x).squeeze(1), y) + lam * head.weight.square().sum()
        loss.backward()
        optimizer.step()
    return head.cpu()


def scores_of(head: torch.nn.Linear, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return head(x).squeeze(1).numpy()


class Trainer:
    """Fits one method on one split: the windows, labels and same-video pairs of that split."""

    def __init__(self, x: torch.Tensor, labels: np.ndarray, videos: list[str], anomaly: np.ndarray,
                 starts: np.ndarray, device: str):
        self.x, self.labels, self.device = x, labels, device
        self.y = torch.tensor(labels)
        self.pairs = before_pairs(videos, labels, anomaly, starts)
        self.no_pairs = tuple(t[:0] for t in self.pairs)

    def bce(self, lam: float) -> torch.nn.Linear:
        return fit_head(self.x, self.y, self.no_pairs, 0.0, lam, self.device)

    def before(self, alpha: float, lam: float) -> torch.nn.Linear:
        return fit_head(self.x, self.y, self.pairs, alpha, lam, self.device)

    def jtt(self, upweight: float, lam: float, stage1_lam: float) -> torch.nn.Linear:
        wrong = (scores_of(self.bce(stage1_lam), self.x) > 0) != (self.labels > 0.5)
        weight = torch.tensor(np.where(wrong, upweight, 1.0), dtype=torch.float32)
        return fit_weighted(self.x, self.y, weight, lam, self.device)

    def fit(self, method: str, lam: float, bce_lam: float) -> torch.nn.Linear:
        if method == "bce":
            return self.bce(lam)
        if method.startswith("before_a"):
            return self.before(float(method.removeprefix("before_a")), lam)
        return self.jtt(float(method.removeprefix("jtt_up")), lam, bce_lam)


def test_report(out: Path, kind: str) -> dict:
    """Print centered_fit.report and the event-phase AUC per method; return the phase values."""
    report(out, kind)
    items = {item.name: item for item in DATASETS.build(load_encoder_config(ENCODER_RUN / "config.yaml").test).items()}
    fit = json.loads((out / "fit.json").read_text())
    scores = np.load(out / "test_scores.npy")
    meta = [json.loads(line) for line in (out / "test_meta.jsonl").read_text().splitlines()]
    phases = {}
    for method in fit["methods"]:
        state = select_state(fit, method, kind)
        values = np.asarray(scores[fit["methods"].index(method), 0, state], dtype=np.float64)
        phases[method] = phase_auc(meta, values, items)
        cells = {name: round(phases[method][name]["auc_vs_normal"], 3) for name in ("before", "during", "after")}
        gap = phases[method]["during"]["auc_vs_normal"] - phases[method]["before"]["auc_vs_normal"]
        print(f"{out.name}/{kind}/{method} phase vs normal {cells}  during-before {gap:.3f}")
    (out / "report.json").write_text(json.dumps(phases, indent=2) + "\n")
    return phases


def main(features: str, name: str, kind: str = "patch_mean", device: str = "cuda") -> None:
    features_dir = Path(features)
    out = LOSS_CHANGES / "shortcut" / name
    if (out / "fit.json").exists():
        test_report(out, kind)
        return
    torch.cuda.set_per_process_memory_fraction(0.2)
    meta = read_meta(features_dir)
    train_meta = meta["train_anomaly"] + meta["train_normal"]
    videos = [row["video"] for row in train_meta]
    val = split(videos, {row["video"]: row["category"] for row in train_meta})
    is_val = np.array([video in val for video in videos])
    labels = np.array([row["holmes"] for row in train_meta], dtype=np.float32)
    anomaly = np.array([row["category"] != "Normal" for row in train_meta])
    starts = np.array([row["start_s"] for row in train_meta])
    fit_videos = [video for video, keep in zip(videos, is_val, strict=True) if not keep]
    print(f"before windows: {int(before_rows(videos, labels, anomaly, starts).sum())} of {int(anomaly.sum())} "
          f"anomaly-video windows", flush=True)

    arrays = {set_name: np.load(features_dir / set_name / f"{kind}.npy", mmap_mode="r") for set_name in SETS}
    n_states = arrays["test"].shape[1]
    scores = np.empty((len(METHODS), 1, n_states, len(meta["test"])), dtype=np.float32)
    choices = {}
    for state in range(n_states):
        train = np.concatenate([np.asarray(arrays[s][:, state], dtype=np.float32)
                                for s in ("train_anomaly", "train_normal")])
        test = np.asarray(arrays["test"][:, state], dtype=np.float32)
        fit_x, val_x = standardize(train[~is_val], train[is_val])
        all_x, test_x = standardize(train, test)
        fit_split = Trainer(fit_x, labels[~is_val], fit_videos, anomaly[~is_val], starts[~is_val], device)
        all_split = Trainer(all_x, labels, videos, anomaly, starts, device)
        bce_lam = LAMS[0]
        for method_i, method in enumerate(METHODS):
            best = (-np.inf, LAMS[0])
            for lam in LAMS:
                head = fit_split.fit(method, lam, bce_lam)
                value = in_video_auc(scores_of(head, val_x), labels[is_val], anomaly[is_val])
                best = max(best, (value, lam))
            if method == "bce":
                bce_lam = best[1]
            choices[f"{method}/{kind}/{state}"] = {"setting": {"lam": best[1]}, "validation": best[0]}
            scores[method_i, 0, state] = scores_of(all_split.fit(method, best[1], bce_lam), test_x)
        print(f"state {state} done", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "test_scores.npy", scores)
    (out / "test_meta.jsonl").write_text((features_dir / "test" / "meta.jsonl").read_text())
    fit = {"methods": list(METHODS), "kinds": [kind], "n_states": n_states, "features": str(features_dir),
           "val_videos": sorted(val), "choices": choices}
    (out / "fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    test_report(out, kind)


if __name__ == "__main__":
    main(*sys.argv[1:])
