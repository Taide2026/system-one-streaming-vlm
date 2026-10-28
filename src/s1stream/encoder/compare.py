"""Last stage of encoder probing: `s1stream encoder-compare` (encoder-probing decisions 7, 8).

Input: a results run folder with one fit folder per encoder feature folder (fit-<encoder>, fit-<encoder>-head; written
by probe-fit), and the encoder config (for the test videos). Each fit has its own kinds and state count (fit.json).

Per fit folder, kind and method, on the test windows: the frame AUC curve over the states, with N_BOOT video resamples
(seed SEED; the same resamples, drawn as in probe.compare.bootstrap, for every fit, so differences are paired).
Three states are reported:
  selected: select_state (validation only, decision 7)
  last:     the last state
  peak:     the state with the highest test frame AUC (reference only; the first one on a tie)
Difference against the baseline (BASELINE: fit-qwen35-vit, kind patch_mean, decision 8), per method: frame AUC of this
fit, kind and method at its selected state minus the baseline frame AUC at the baseline's selected state of the same
method, paired over the resamples.

Writes into the run folder:
  encoder_compare.json {"fits": [...], "methods": [...], "n_boot": N_BOOT, "seed": SEED,
                        "baseline": {"fit": "qwen35-vit", "kind": "patch_mean"},
                        "results": {"<fit>/<kind>/<method>": {
                            "n_states": S, "curve": {"value": [S], "low": [S], "high": [S]},
                            "selected": R, "last": R, "peak": R,
                            "difference": {"state": i, "baseline_state": j, "value": x, "low": x, "high": x}}}}
                       where R = {"state": i, "value": x, "low": x, "high": x}; <fit> is the folder name without "fit-".
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

from s1stream.probe.compare import _interval, _weighted_curve, _window_counts, _window_keys
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
    Raise FileExistsError when encoder_compare.json exists; ValueError when fit-qwen35-vit is missing or has no
    patch_mean kind; ValueError when the probe.compare.WINDOW_KEYS of the test_meta.jsonl rows differ between two
    fit folders. items None: build the test items from the config (encoder.extract.load_encoder_config, registry
        DATASETS); given items: the config is not read."""
    out = run_dir / "encoder_compare.json"
    if out.exists():
        raise FileExistsError(out)
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
    selected_samples = {}
    for name, fit in fits.items():
        for method_i, method in enumerate(fit["methods"]):
            for kind_i, kind in enumerate(fit["kinds"]):
                key = f"{name}/{kind}/{method}"
                samples = np.stack([_weighted_curve(state_scores, pos, neg)[0]
                                    for state_scores in scores[name][method_i, kind_i]], axis=1)
                curve = _interval(samples[0], samples[1:])
                chosen = select_state(fit, method, kind)
                selected_samples[key] = samples[:, chosen]
                entry = {"n_states": fit["n_states"], "curve": curve}
                for label, state in (("selected", chosen), ("last", fit["n_states"] - 1),
                                     ("peak", int(np.argmax(np.nan_to_num(samples[0], nan=-np.inf))))):
                    entry[label] = {"state": state, **{field: curve[field][state]
                                                       for field in ("value", "low", "high")}}
                results[key] = entry

    for key, entry in results.items():
        _, _, method = key.rsplit("/", 2)
        baseline_key = f"{BASELINE[0]}/{BASELINE[1]}/{method}"
        diff = selected_samples[key] - selected_samples[baseline_key]
        entry["difference"] = {"state": entry["selected"]["state"],
                               "baseline_state": results[baseline_key]["selected"]["state"],
                               **_interval(diff[0], diff[1:])}

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
    with out.open("x") as stream:
        json.dump(payload, stream, indent=2)
    return out
