"""Last stage of probing: `s1stream probe-compare` (finetune-layer-probing decisions 9, 11, 12).

Input: a results run folder with one fit folder per condition (fit-A, fit-B, fit-C, written by probe-fit), and the
probe config (for the test videos: fps, frame count and annotated events).

Metrics per condition, method, token kind and state, on the test windows:
  frame_auc (main), frame_ap: s1stream.metrics conventions (frame ranges from windows.window_frame_range);
  window_auc: s1stream.metrics.auc on the window scores and window labels.
P(Yes) of every condition (meta "p_yes") is scored the same way, as a reference line (decision 9).

Bootstrap: N_BOOT resamples of the test videos with replacement (numpy default_rng(SEED)); every condition, method,
kind and state uses the same resamples, so differences are paired. A resample whose videos have no abnormal frame
or no normal frame gives NaN for that metric; intervals are the 2.5 and 97.5 percentiles over the finite values
(np.nanpercentile). The point value is the metric on the original test set.

Layer summary (decision 12) of a frame AUC curve over the states: layer_summary. Its onset state, peak state and
peak value are also computed on every resample, which gives their intervals and the paired intervals of their
differences between conditions.

Writes into the run folder:
  compare.json  {"conditions": [...], "methods": [...], "kinds": [...], "n_states": S, "n_boot": N_BOOT,
                 "seed": SEED, "fits": {<condition>: <fit dir>},
                 "metrics": {"<cond>/<method>/<kind>/<metric>": {"value": [S], "low": [S], "high": [S]}},
                 "p_yes": {"<cond>/<metric>": {"value": x, "low": x, "high": x}},
                 "differences": {"<pair>/<method>/<kind>/<metric>": {"value": [S], "low": [S], "high": [S]}},
                 "summary": {"<cond>/<method>/<kind>": {"onset": I, "peak": I, "peak_value": I}},
                 "summary_differences": {"<pair>/<method>/<kind>": {"onset": I, "peak": I, "peak_value": I}},
                 "cka": {"<pair>/<kind>": [S]}}
                where I = {"value": x, "low": x, "high": x}; pair is one of PAIRS, e.g. "C-B" = C minus B.
  <method>_<kind>.png  frame AUC per state, one line per condition with its 95 % band, P(Yes) as dashed lines.
  cka.png              linear CKA per state for every pair and kind.
Conditions missing from the run folder are left out, together with their pairs.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from s1stream import metrics
from s1stream.probe.fit import KINDS, METHODS
from s1stream.types import VideoItem
from s1stream.windows import window_frame_range

METRICS = ("frame_auc", "frame_ap", "window_auc")
PAIRS = (("A", "B"), ("C", "B"), ("C", "A"))
WINDOW_KEYS = ("i", "video", "window", "start_s", "end_s", "label")  # p_yes and seconds differ per condition
N_BOOT, SEED = 1000, 20261022


def frame_metric(scores: Sequence[float], meta: list[dict], items: dict[str, VideoItem], metric: str) -> float | None:
    """One metric of METRICS for one score per test window. meta: the test_meta.jsonl rows (video, start_s, end_s,
    label), in the order of scores; items: test VideoItem by name (fps, n_frames, events). frame_auc and frame_ap
    use metrics.frame_auc / metrics.frame_ap with window_frame_range(start_s, end_s, fps); window_auc uses
    metrics.auc with meta "label"."""
    if metric == "window_auc":
        return metrics.auc(scores, [row["label"] for row in meta])
    if metric not in ("frame_auc", "frame_ap"):
        raise ValueError(f"Unknown metric: {metric}")
    videos = []
    for name in dict.fromkeys(row["video"] for row in meta):
        item = items[name]
        windows = [(*window_frame_range(row["start_s"], row["end_s"], item.fps), score)
                   for row, score in zip(meta, scores, strict=True) if row["video"] == name]
        videos.append({"n_frames": item.n_frames, "events": item.events, "windows": windows})
    return metrics.frame_auc(videos) if metric == "frame_auc" else metrics.frame_ap(videos)


def layer_summary(curve: Sequence[float]) -> dict[str, float | int]:
    """{"onset": first state with value >= 0.5 + 0.9 * (max - 0.5), "peak": first state with the max value,
    "peak_value": max}. NaN values are ignored; all-NaN gives NaN for all three."""
    values = np.asarray(curve, dtype=float)
    if not np.any(np.isfinite(values)):
        return {"onset": np.nan, "peak": np.nan, "peak_value": np.nan}
    peak_value = np.nanmax(values).item()
    qualifying = np.flatnonzero(values >= 0.5 + 0.9 * (peak_value - 0.5))
    onset = qualifying[0].item() if qualifying.size else np.nan
    return {"onset": onset, "peak": np.nanargmax(values).item(), "peak_value": peak_value}


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA of two (n, p) and (n, q) matrices: columns centered, ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F),
    computed in float64."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return np.asarray(np.linalg.norm(y.T @ x) ** 2 / denominator).item() if denominator else np.nan


def _window_counts(meta: list[dict], items: dict[str, VideoItem]) -> tuple[np.ndarray, np.ndarray]:
    """Count covered abnormal and normal frames per window; overlapping windows raise ValueError."""
    positive = np.zeros(len(meta), dtype=np.int64)
    negative = np.zeros(len(meta), dtype=np.int64)
    covered = {name: np.zeros(item.n_frames, dtype=bool) for name, item in items.items()}
    labels = {}
    for name, item in items.items():
        labels[name] = np.zeros(item.n_frames, dtype=bool)
        for start, end in item.events:
            labels[name][max(0, start):min(item.n_frames, end + 1)] = True
    for i, row in enumerate(meta):
        name = row["video"]
        item = items[name]
        first, last = window_frame_range(row["start_s"], row["end_s"], item.fps)
        span = slice(max(first, 0), min(last + 1, item.n_frames))
        if np.any(covered[name][span]):
            raise ValueError(f"Overlapping windows for {name}")
        covered[name][span] = True
        positive[i] = np.count_nonzero(labels[name][span])
        negative[i] = labels[name][span].size - positive[i]
    return positive, negative


def _weighted_curve(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUC and AP for each row of weights, with equal scores grouped as one threshold."""
    order = np.argsort(-scores, kind="mergesort")
    boundaries = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    pos = np.add.reduceat(positive[:, order], boundaries, axis=1).astype(float)
    neg = np.add.reduceat(negative[:, order], boundaries, axis=1).astype(float)
    total_pos = pos.sum(axis=1)
    total_neg = neg.sum(axis=1)
    cumulative_pos = np.cumsum(pos, axis=1)
    cumulative_neg = np.cumsum(neg, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc_values = np.sum(pos * (total_neg[:, None] - cumulative_neg + neg / 2), axis=1) / (total_pos * total_neg)
        precision = cumulative_pos / (cumulative_pos + cumulative_neg)
        ap_values = np.nansum(pos * precision, axis=1) / total_pos
    return auc_values, ap_values


def _interval(point: np.ndarray | float, samples: np.ndarray) -> dict:
    values = np.asarray(point).tolist()
    if not np.any(np.isfinite(samples)):
        low = high = np.full(np.shape(point), np.nan).tolist()
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low, high = (np.nanpercentile(samples, q, axis=0).tolist() for q in (2.5, 97.5))
    return {"value": values, "low": low, "high": high}


def bootstrap(scores: dict[str, np.ndarray], p_yes: dict[str, np.ndarray], meta: list[dict],
              items: dict[str, VideoItem], n: int = N_BOOT, seed: int = SEED) -> dict:
    """The "metrics", "p_yes", "differences", "summary" and "summary_differences" parts of compare.json (see the
    module docstring). scores: per condition an array (methods, kinds, S, n_test) as in test_scores.npy; p_yes: per
    condition an array (n_test,); method and kind names in the keys come from probe.fit.METHODS and KINDS.
    The frame metrics must equal frame_metric exactly; precompute per video the frame counts of each window and add
    them per resample instead of building frame arrays again."""
    positive, negative = _window_counts(meta, items)
    videos = list(dict.fromkeys(row["video"] for row in meta))
    video_index = {name: i for i, name in enumerate(videos)}
    window_video = np.array([video_index[row["video"]] for row in meta])
    rng = np.random.default_rng(seed)
    draws = np.array([rng.integers(0, len(videos), len(videos)) for _ in range(n)], dtype=int)
    weights = np.zeros((n + 1, len(videos)), dtype=np.int64)
    weights[0] = 1
    for i, draw in enumerate(draws, 1):
        weights[i] = np.bincount(draw, minlength=len(videos))
    window_weights = weights[:, window_video]
    pos = window_weights * positive
    neg = window_weights * negative
    labels = np.array([row["label"] == 1 for row in meta])
    window_pos = window_weights * labels
    window_neg = window_weights * ~labels
    result: dict = {key: {} for key in ("metrics", "p_yes", "differences", "summary", "summary_differences")}
    curves = {}
    summaries = {}
    for condition, array in scores.items():
        for method_i, method in enumerate(METHODS):
            for kind_i, kind in enumerate(KINDS):
                base = f"{condition}/{method}/{kind}"
                metric_curves = {metric: [] for metric in METRICS}
                for state_scores in array[method_i, kind_i]:
                    frame_auc, frame_ap = _weighted_curve(state_scores, pos, neg)
                    window_auc, _ = _weighted_curve(state_scores, window_pos, window_neg)
                    for metric, values in zip(METRICS, (frame_auc, frame_ap, window_auc), strict=True):
                        metric_curves[metric].append(values)
                for metric, values in metric_curves.items():
                    curve = np.stack(values, axis=1)
                    curves[base, metric] = curve
                    result["metrics"][f"{base}/{metric}"] = _interval(curve[0], curve[1:])
                fields = {field: [] for field in ("onset", "peak", "peak_value")}
                for curve in curves[base, "frame_auc"]:
                    summary = layer_summary(curve)
                    for field, values in fields.items():
                        values.append(summary[field])
                summaries[base] = {field: np.asarray(values, dtype=float) for field, values in fields.items()}
                result["summary"][base] = {field: _interval(values[0], values[1:])
                                           for field, values in summaries[base].items()}
        p = np.asarray(p_yes[condition])
        frame_auc, frame_ap = _weighted_curve(p, pos, neg)
        window_auc, _ = _weighted_curve(p, window_pos, window_neg)
        for metric, values in zip(METRICS, (frame_auc, frame_ap, window_auc), strict=True):
            result["p_yes"][f"{condition}/{metric}"] = _interval(values[0], values[1:])
    for left, right in PAIRS:
        if left not in scores or right not in scores:
            continue
        for method in METHODS:
            for kind in KINDS:
                base = f"{left}-{right}/{method}/{kind}"
                for metric in METRICS:
                    diff = curves[f"{left}/{method}/{kind}", metric] - curves[f"{right}/{method}/{kind}", metric]
                    result["differences"][f"{base}/{metric}"] = _interval(diff[0], diff[1:])
                result["summary_differences"][base] = {}
                for field in ("onset", "peak", "peak_value"):
                    diff = summaries[f"{left}/{method}/{kind}"][field] - summaries[f"{right}/{method}/{kind}"][field]
                    result["summary_differences"][base][field] = _interval(diff[0], diff[1:])
    return result


def _window_keys(rows: list[dict]) -> list[tuple]:
    return [tuple(row.get(key) for key in WINDOW_KEYS) for row in rows]


def run_compare(config_path: Path, run_dir: Path, items: dict[str, VideoItem] | None = None,
                log: Callable[[str], None] = print) -> Path:
    """Read run_dir/fit-<condition>/ (fit.json, test_scores.npy, test_meta.jsonl) for each condition present;
    raise ValueError when the WINDOW_KEYS of the test_meta.jsonl rows of two conditions differ, FileNotFoundError without any fit
    folder and FileExistsError when run_dir/compare.json exists. items None: build the test items from the config
    (probe.extract.load_probe_config, registry DATASETS); given items: the config is not read. CKA reads the test features of each fit (fit.json
    "features" / "test" / "<kind>.npy") one state at a time (mmap). Write compare.json and the PNG files with
    matplotlib (Agg backend). Return run_dir / "compare.json"."""
    out = run_dir / "compare.json"
    if out.exists():
        raise FileExistsError(out)
    fits = {name: run_dir / f"fit-{name}" for name in ("A", "B", "C") if (run_dir / f"fit-{name}").is_dir()}
    if not fits:
        raise FileNotFoundError(f"No fit folders in {run_dir}")
    if items is None:
        from s1stream.probe.extract import load_probe_config
        from s1stream.registry import DATASETS
        config = load_probe_config(config_path)
        items = {item.name: item for item in DATASETS.build(config.test).items()}
    scores = {}
    p_yes = {}
    features = {}
    meta = None
    n_states = 0
    for condition, folder in fits.items():
        started = time.monotonic()
        try:
            fit = json.loads((folder / "fit.json").read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid fit metadata in {folder}") from exc
        try:
            rows = [json.loads(line) for line in (folder / "test_meta.jsonl").read_text().splitlines()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid test metadata in {folder}") from exc
        if meta is not None and _window_keys(rows) != _window_keys(meta):
            raise ValueError(f"Test metadata differs for {condition}")
        meta = rows
        features[condition] = Path(fit["features"])
        scores[condition] = np.load(folder / "test_scores.npy", mmap_mode="r")
        n_states = fit["n_states"]
        p_yes[condition] = np.asarray([row["p_yes"] for row in rows])
        log(f"{condition}: loaded in {time.monotonic() - started:.2f}s")
    assert meta is not None
    results = bootstrap(scores, p_yes, meta, items)
    cka = {}
    for left, right in PAIRS:
        if left not in fits or right not in fits:
            continue
        started = time.monotonic()
        for kind in KINDS:
            x = np.load(features[left] / "test" / f"{kind}.npy", mmap_mode="r")
            y = np.load(features[right] / "test" / f"{kind}.npy", mmap_mode="r")
            cka[f"{left}-{right}/{kind}"] = [linear_cka(np.asarray(x[:, state], dtype=np.float32),
                                                          np.asarray(y[:, state], dtype=np.float32))
                                              for state in range(n_states)]
        log(f"{left}-{right}: CKA in {time.monotonic() - started:.2f}s")
    results.update({"conditions": list(fits), "methods": list(METHODS), "kinds": list(KINDS),
                    "n_states": n_states, "n_boot": N_BOOT, "seed": SEED,
                    "fits": {name: str(folder) for name, folder in fits.items()}, "cka": cka})
    for method in METHODS:
        for kind in KINDS:
            fig, ax = plt.subplots()
            for condition in fits:
                curve = results["metrics"][f"{condition}/{method}/{kind}/frame_auc"]
                line, = ax.plot(curve["value"], label=condition)
                color = line.get_color()
                ax.fill_between(range(n_states), curve["low"], curve["high"], color=color, alpha=0.2)
                ax.axhline(results["p_yes"][f"{condition}/frame_auc"]["value"], color=color, linestyle="--",
                           label=f"{condition} P(Yes)")
            ax.set(xlabel="State", ylabel="Frame AUC", title=f"{method} / {kind}")
            ax.legend()
            fig.savefig(run_dir / f"{method}_{kind}.png")
            plt.close(fig)
    fig, ax = plt.subplots()
    for name, values in cka.items():
        ax.plot(values, label=name)
    ax.set(xlabel="State", ylabel="Linear CKA")
    if cka:
        ax.legend()
    fig.savefig(run_dir / "cka.png")
    plt.close(fig)
    with out.open("x") as stream:
        json.dump(results, stream, indent=2)
    return out
