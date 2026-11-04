"""Last stage of encoder probing: `s1stream encoder-compare` (encoder-probing decisions 7, 8).

Input: a results run folder with one fit folder per encoder feature folder (fit-<encoder>, fit-<encoder>-head; written
by probe-fit), and the encoder config (for the test videos). Each fit has its own kinds and state count (fit.json).

Per fit folder, kind and method, on the test windows:
  curve:          frame AUC for every state, with N_BOOT video resamples (seed SEED), for the plots
  state:          select_state (validation only, decision 7)
  metrics:        bootstrap.metric_summary of every name in bootstrap.VIDEO_METRICS at that state
  difference:     the same metrics minus the baseline entry (BASELINE: fit-qwen35-vit, kind patch_mean, decision 8) of
                  the same method at its own selected state, paired over the resamples; zero for the baseline entry
  baseline_state: the selected state of that baseline entry
Videos come from report.frame_videos on the test_meta.jsonl rows and the items (fps of the items).

Writes into the run folder:
  encoder_compare.json {"fits": [...], "methods": [...], "n_boot": N_BOOT, "seed": SEED,
                        "baseline": {"fit": "qwen35-vit", "kind": "patch_mean"},
                        "results": {"<fit>/<kind>/<method>": {
                            "n_states": S, "curve": {"value": [S], "low": [S], "high": [S]},
                            "state": i, "baseline_state": j,
                            "metrics": {metric: {"value", "low", "high"}},
                            "difference": {metric: {"value", "low", "high"}}}}}
                       <fit> is the folder name without "fit-". An existing encoder_compare.json is overwritten.
  <method>.png         frame AUC per state, one line per fit and kind.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from s1stream.bootstrap import metric_summary
from s1stream.probe.compare import _interval, _weighted_curve, _window_counts, _window_keys
from s1stream.report import frame_videos
from s1stream.types import VideoItem

N_BOOT, SEED = 1000, 20261022
BASELINE = ("qwen35-vit", "patch_mean")


def select_state(fit: dict, method: str, kind: str) -> int:
    """The state chosen on validation from fit.json choices "<method>/<kind>/<state>" "validation":
    mil the lowest value (video BCE), holmes and knn the highest (AUC). A tie goes to the lower state."""
    values = [fit["choices"][f"{method}/{kind}/{state}"]["validation"]
              for state in range(fit["n_states"])]
    return (min if method == "mil" else max)(range(len(values)), key=values.__getitem__)


def run_encoder_compare(config_path: Path, run_dir: Path, items: dict[str, VideoItem] | None = None,
                        log: Callable[[str], None] = print) -> Path:
    """Read every run_dir/fit-*/ (fit.json, test_scores.npy, test_meta.jsonl) and write encoder_compare.json and the PNG
    files (matplotlib, Agg backend). Return run_dir / "encoder_compare.json".
    Overwrite an existing encoder_compare.json; raise ValueError when fit-qwen35-vit is missing or has no
    patch_mean kind; ValueError when the probe.compare.WINDOW_KEYS of the test_meta.jsonl rows differ between two
    fit folders. items None: build the test items from the config (encoder.extract.load_encoder_config, registry
        DATASETS); given items: the config is not read."""
    out = run_dir / "encoder_compare.json"
    folders = sorted(folder for folder in run_dir.glob("fit-*") if folder.is_dir())
    baseline_folder = run_dir / f"fit-{BASELINE[0]}"
    if baseline_folder not in folders:
        raise ValueError(f"Missing baseline fit: {baseline_folder}")
    if items is None:
        from s1stream.encoder.extract import load_encoder_config
        from s1stream.registry import DATASETS

        items = {item.name: item for item in DATASETS.build(load_encoder_config(config_path).test).items()}

    fits = {}
    scores = {}
    meta = None
    for folder in folders:
        name = folder.name.removeprefix("fit-")
        try:
            fit = json.loads((folder / "fit.json").read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid fit metadata in {folder}") from exc
        try:
            rows = [json.loads(line) for line in (folder / "test_meta.jsonl").read_text().splitlines()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid test metadata in {folder}") from exc
        if meta is not None and _window_keys(rows) != _window_keys(meta):
            raise ValueError(f"Test metadata differs for {name}")
        meta = rows
        fits[name] = fit
        scores[name] = np.load(folder / "test_scores.npy", mmap_mode="r")
        log(f"{name}: loaded")
    if BASELINE[1] not in fits[BASELINE[0]]["kinds"]:
        raise ValueError(f"Missing baseline kind: {BASELINE[1]}")

    assert meta is not None
    positive, negative = _window_counts(meta, items)
    videos = list(dict.fromkeys(row["video"] for row in meta))
    video_index = {name: i for i, name in enumerate(videos)}
    window_video = np.array([video_index[row["video"]] for row in meta])
    rng = np.random.default_rng(SEED)
    weights = np.zeros((N_BOOT + 1, len(videos)), dtype=np.int64)
    weights[0] = 1
    for i in range(1, N_BOOT + 1):
        draw = rng.integers(0, len(videos), len(videos))
        weights[i] = np.bincount(draw, minlength=len(videos))
    pos = weights[:, window_video] * positive
    neg = weights[:, window_video] * negative

    results = {}
    for name, fit in fits.items():
        for method_i, method in enumerate(fit["methods"]):
            for kind_i, kind in enumerate(fit["kinds"]):
                key = f"{name}/{kind}/{method}"
                samples = np.stack([_weighted_curve(state_scores, pos, neg)[0]
                                    for state_scores in scores[name][method_i, kind_i]], axis=1)
                curve = _interval(samples[0], samples[1:])
                chosen = select_state(fit, method, kind)
                results[key] = {"n_states": fit["n_states"], "state": chosen, "curve": curve}

    add_metrics(results, fits, scores, meta, items)

    methods = list(dict.fromkeys(method for fit in fits.values() for method in fit["methods"]))
    payload = {"fits": list(fits), "methods": methods, "n_boot": N_BOOT, "seed": SEED,
               "baseline": {"fit": BASELINE[0], "kind": BASELINE[1]}, "results": results}
    for method in methods:
        fig, ax = plt.subplots()
        for name, fit in fits.items():
            if method not in fit["methods"]:
                continue
            for kind in fit["kinds"]:
                values = results[f"{name}/{kind}/{method}"]["curve"]["value"]
                ax.plot(range(len(values)), values, marker="o" if len(values) == 1 else None,
                        label=f"{name}/{kind}")
        ax.set(xlabel="State", ylabel="Frame AUC", title=method)
        ax.legend()
        fig.savefig(run_dir / f"{method}.png")
        plt.close(fig)
    with out.open("w") as stream:
        json.dump(payload, stream, indent=2)
    return out


def add_metrics(results: dict, fits: dict, scores: dict, meta: list[dict], items: dict[str, VideoItem]) -> None:
    """Add selected-state metrics and paired differences to every entry (see the module docstring)."""
    manifest = {name: {"fps": item.fps, "n_frames": item.n_frames,
                       "events": [list(event) for event in item.events]} for name, item in items.items()}
    fps_values = {item.fps for item in items.values()}
    if len(fps_values) != 1:
        raise ValueError("Test videos must share one fps")
    fps = fps_values.pop()
    methods = list(dict.fromkeys(method for fit in fits.values() for method in fit["methods"]))
    for method in methods:
        baseline_key = f"{BASELINE[0]}/{BASELINE[1]}/{method}"
        runs = {}
        for name, fit in fits.items():
            if method not in fit["methods"]:
                continue
            method_i = fit["methods"].index(method)
            for kind_i, kind in enumerate(fit["kinds"]):
                key = f"{name}/{kind}/{method}"
                state = results[key]["state"]
                values = scores[name][method_i, kind_i, state]
                rows = [dict(row, score=float(score)) for row, score in zip(meta, values, strict=True)]
                runs[key] = frame_videos(rows, manifest)
        summary = metric_summary(runs, fps, baseline=baseline_key)
        for key in runs:
            entry = results[key]
            entry["baseline_state"] = results[baseline_key]["state"]
            entry["metrics"] = summary["runs"][key]
            if key == baseline_key:
                entry["difference"] = {
                    metric: {"value": 0.0, "low": 0.0, "high": 0.0} for metric in summary["runs"][key]}
            else:
                entry["difference"] = summary["differences"][key]
