"""Probe fit on per-video centered features: each window feature minus the mean feature of its video, per state.

Question: is the in-video AUC low because the window features are dominated by the scene of the video? Centering
removes the video mean (the scene), so a head can use only the variation inside a video. A fixed linear head gives
the same ranking inside a video with or without centering; the difference comes from training (the heads are refit).
The test centering uses all windows of the video, also later ones, so this is an analysis, not a streaming method.
Usage: uv run python <this file> <feature folder> [kind]   (default kind patch_mean)
Writes centered features to $S1_FEATURES/encoder_probe_ucf_short_v1_f8_centered/<folder>/run-1/ (kind only) and the
fit to run-1/analysis-centered/fit-<folder>/ (probe.fit.fit_condition, device cuda), then prints all, in-video and
per-video AUC as in per_category_auc.py and per_video_auc.py.
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


def center(source: Path, target: Path, kind: str) -> None:
    """Write target/<set>/<kind>.npy (float32, video mean removed per state) and copy meta.jsonl; run.json kinds [kind]."""
    for name in SETS:
        (target / name).mkdir(parents=True, exist_ok=True)
        shutil.copy(source / name / "meta.jsonl", target / name / "meta.jsonl")
        videos = np.array([json.loads(line)["video"] for line in (source / name / "meta.jsonl").read_text().splitlines()])
        x = np.load(source / name / f"{kind}.npy", mmap_mode="r")
        out = np.lib.format.open_memmap(target / name / f"{kind}.npy", mode="w+", dtype=np.float32, shape=x.shape)
        for video in dict.fromkeys(videos):
            rows = np.flatnonzero(videos == video)
            block = np.asarray(x[rows[0]:rows[-1] + 1], dtype=np.float32)
            out[rows[0]:rows[-1] + 1] = block - block.mean(axis=0, keepdims=True)
        out.flush()
    run = json.loads((source / "run.json").read_text())
    run.update(kinds=[kind], centered_from=str(source))
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


def main(folder: str, kind: str = "patch_mean") -> None:
    source = FEATURES / "encoder_probe_ucf_short_v1_f8" / folder / "run-1"
    target = FEATURES / "encoder_probe_ucf_short_v1_f8_centered" / folder / "run-1"
    fit_dir = RUN / "analysis-centered" / f"fit-{folder}"
    if not (target / "run.json").exists():
        center(source, target, kind)
    if not (fit_dir / "fit.json").exists():
        torch.cuda.set_per_process_memory_fraction(0.2)
        fit_condition(target, fit_dir, device="cuda")
    report(fit_dir, kind)


if __name__ == "__main__":
    main(*sys.argv[1:])
