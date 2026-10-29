"""Probe fit on per-video centered features: each window feature minus the mean feature of its video, per state.

Question: is the in-video AUC low because the window features are dominated by the scene of the video? Centering
removes the video mean (the scene), so a head can use only the variation inside a video. A fixed linear head gives
the same ranking inside a video with or without centering; the difference comes from training (the heads are refit).
Modes:
  full      minus the mean of all windows of the video, also later ones (an analysis, not a streaming method)
  past<N>   minus the mean of the windows that start in [start - N s, start), i.e. only earlier windows (causal);
            the first window of a video has no earlier window and becomes the zero vector
  pastall   minus the mean of all earlier windows of the video (causal)
Usage: uv run python <this file> <feature folder> [kind] [mode]   (default patch_mean full)
Writes centered features to $S1_FEATURES/encoder_probe_ucf_short_v1_f8_centered[_<mode>]/<folder>/run-1/ (kind only)
and the fit to run-1/analysis-centered/fit-<folder>[-<mode>]/ (probe.fit.fit_condition, device cuda); the suffix is
left out for mode full. Then prints all, in-video and per-video AUC as in per_category_auc.py and per_video_auc.py.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import s1stream.plugins  # noqa: F401  (registers the datasets)
from s1stream.encoder.compare import select_state
from s1stream.encoder.extract import load_encoder_config
from s1stream.probe.compare import _weighted_curve, _window_counts
from s1stream.probe.fit import SETS, fit_condition
from s1stream.registry import DATASETS

RUN = Path(__file__).resolve().parents[1]
FEATURES = Path(os.environ["S1_FEATURES"])


def reference(block: np.ndarray, starts: np.ndarray, mode: str) -> np.ndarray:
    """The mean each window of one video is centered on; block [windows, states, hidden], starts in seconds."""
    if mode == "full":
        return block.mean(axis=0, keepdims=True)
    span = np.inf if mode == "pastall" else float(mode.removeprefix("past"))
    out = np.zeros_like(block)
    for i, start in enumerate(starts):
        past = (starts < start) & (starts >= start - span)
        out[i] = block[past].mean(axis=0) if past.any() else block[i]
    return out


def center(source: Path, target: Path, kind: str, mode: str) -> None:
    """Write target/<set>/<kind>.npy (float32, reference(...) removed per state) and copy meta.jsonl; run.json kinds
    [kind]. Rows of one video are contiguous in meta.jsonl."""
    for name in SETS:
        (target / name).mkdir(parents=True, exist_ok=True)
        shutil.copy(source / name / "meta.jsonl", target / name / "meta.jsonl")
        meta = [json.loads(line) for line in (source / name / "meta.jsonl").read_text().splitlines()]
        videos = np.array([row["video"] for row in meta])
        starts = np.array([row["start_s"] for row in meta])
        x = np.load(source / name / f"{kind}.npy", mmap_mode="r")
        out = np.lib.format.open_memmap(target / name / f"{kind}.npy", mode="w+", dtype=np.float32, shape=x.shape)
        for video in dict.fromkeys(videos):
            rows = np.flatnonzero(videos == video)
            block = np.asarray(x[rows[0]:rows[-1] + 1], dtype=np.float32)
            out[rows[0]:rows[-1] + 1] = block - reference(block, starts[rows[0]:rows[-1] + 1], mode)
        out.flush()
    run = json.loads((source / "run.json").read_text())
    run.update(kinds=[kind], centered_from=str(source), centering=mode)
    (target / "run.json").write_text(json.dumps(run, indent=1) + "\n")


def report(fit_dir: Path, kind: str) -> None:
    items = {item.name: item for item in DATASETS.build(load_encoder_config(RUN / "config.yaml").test).items()}
    fit = json.loads((fit_dir / "fit.json").read_text())
    scores = np.load(fit_dir / "test_scores.npy")
    meta = [json.loads(line) for line in (fit_dir / "test_meta.jsonl").read_text().splitlines()]
    positive, negative = _window_counts(meta, items)
    anomaly = np.array([row["category"] != "Normal" for row in meta])
    video = np.array([row["video"] for row in meta])
    for method in fit["methods"]:
        state = select_state(fit, method, kind)
        values = scores[fit["methods"].index(method), fit["kinds"].index(kind), state]
        overall = _weighted_curve(values, positive[None], negative[None])[0][0]
        in_video = _weighted_curve(values, (positive * anomaly)[None], (negative * anomaly)[None])[0][0]
        per_video = []
        for name in dict.fromkeys(video[anomaly]):
            keep = video == name
            if positive[keep].sum() and negative[keep].sum():
                per_video.append(_weighted_curve(values[keep], positive[keep][None], negative[keep][None])[0][0])
        per_video = np.array(per_video)
        print(f"{fit_dir.name}/{kind}/{method} state {state:2d}: all {overall:.3f}  in_video {in_video:.3f}  "
              f"per-video median {np.median(per_video):.3f}  below 0.5 {int((per_video < 0.5).sum())}/{len(per_video)}")


def main(folder: str, kind: str = "patch_mean", mode: str = "full") -> None:
    suffix = "" if mode == "full" else f"_{mode}"
    source = FEATURES / "encoder_probe_ucf_short_v1_f8" / folder / "run-1"
    target = FEATURES / f"encoder_probe_ucf_short_v1_f8_centered{suffix}" / folder / "run-1"
    fit_dir = RUN / "analysis-centered" / f"fit-{folder}{suffix.replace('_', '-')}"
    if not (target / "run.json").exists():
        center(source, target, kind, mode)
    if not (fit_dir / "fit.json").exists():
        torch.cuda.set_per_process_memory_fraction(0.2)
        fit_condition(target, fit_dir, device="cuda")
    report(fit_dir, kind)


if __name__ == "__main__":
    main(*sys.argv[1:])
