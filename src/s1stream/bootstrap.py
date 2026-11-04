"""Paired bootstrap over videos for any metric of s1stream.metrics.

Input: runs = {run name: videos}, every run a list of video dicts in the format of metrics.frame_auc, with the same
videos in the same order (same length and the same "events" at each index). A metric is a function videos -> float or
None (for example metrics.macro_auc).

Resampling: anomaly videos (non-empty "events") and normal videos are drawn separately, so every draw keeps both
counts. rng = numpy.random.default_rng(seed). For each draw, in this order:
  anomaly picks = rng.integers(0, A, size=A), then normal picks = rng.integers(0, N, size=N)
(A, N = counts of anomaly and normal videos; positions are indices into those two lists in input order). The draw is
the anomaly picks followed by the normal picks. Every run uses the same draws, so the differences are paired.

Intervals: numpy.percentile (default method) of the draw values at q = 100 x (1 - level) / 2 and 100 - q.
The metric is called once on all videos and once per draw, for each run. A draw where a metric value is None is
left out of that run's interval and of every pair that uses it.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from s1stream import metrics

Metric = Callable[[Sequence[dict]], float | None]

N_BOOT, SEED = 1000, 20261022
VIDEO_METRICS = ("frame_auc", "frame_ap", "macro_auc", "median_video_auc", "within_auc", "anomaly_auc",
                 "recall_at_1fa_h")


def metric_summary(runs: dict[str, Sequence[dict]], fps: float, baseline: str | None = None, n: int = N_BOOT,
                   seed: int = SEED) -> dict:
    """All VIDEO_METRICS of every run with paired_bootstrap intervals (same n, seed and level 0.95).

    recall_at_1fa_h = metrics.recall_at_fa_per_hour(videos, fps, 1.0).
    Returns {"runs": {name: {metric: {"value", "low", "high"}}},
             "differences": {name: {metric: {"value", "low", "high"}}}}
    value = the point value (paired_bootstrap "point" or pair "diff"). differences holds name - baseline for every
    run except the baseline; it is {} when baseline is None.
    """
    if baseline is not None and baseline not in runs:
        raise ValueError(f"Missing baseline run: {baseline}")
    cached: dict[str, Sequence[dict]] = {}
    for name, videos in runs.items():
        prepared = []
        for video in videos:
            frames = metrics._video_frames(video)
            abnormal = int(np.count_nonzero(frames[1]))
            prepared.append(dict(video, _cached_frames=frames, _cached_auc=metrics.video_aucs([video])[0],
                                 _cached_pairs=abnormal * (len(frames[1]) - abnormal)))
        cached[name] = prepared
    ordered = cached if baseline is None else {baseline: cached[baseline], **cached}
    _prepare_histograms(cached)
    functions = {
        "frame_auc": lambda videos: _pooled_metric(videos, "auc"),
        "frame_ap": lambda videos: _pooled_metric(videos, "ap"),
        "macro_auc": metrics.macro_auc,
        "median_video_auc": metrics.median_video_auc,
        "within_auc": metrics.within_auc,
        "anomaly_auc": lambda videos: _pooled_metric(videos, "anomaly_auc"),
        "recall_at_1fa_h": lambda videos: metrics.recall_at_fa_per_hour(videos, fps, 1.0),
    }
    summary = {"runs": {name: {} for name in runs},
               "differences": {name: {} for name in runs if baseline is not None and name != baseline}}
    for metric_name, metric in functions.items():
        result = paired_bootstrap(ordered, metric, n=n, seed=seed)
        for name in runs:
            point = result["runs"][name]
            summary["runs"][name][metric_name] = {"value": point["point"], "low": point["low"],
                                                    "high": point["high"]}
        for name in summary["differences"]:
            pair = result["pairs"][f"{name} - {baseline}"]
            summary["differences"][name][metric_name] = {"value": pair["diff"], "low": pair["low"],
                                                           "high": pair["high"]}
    return summary


def _prepare_histograms(runs: dict[str, Sequence[dict]]) -> None:
    """Store covered-frame counts at each pooled score once, before resampling videos."""
    videos = [video for run in runs.values() for video in run]
    all_scores = [video["_cached_frames"][0] for video in videos]
    scores = np.unique(np.concatenate(all_scores)) if all_scores else np.array([])
    for video, frame_scores in zip(videos, all_scores, strict=True):
        labels = video["_cached_frames"][1]
        indices = np.searchsorted(scores, frame_scores)
        count = np.bincount(indices, minlength=len(scores))
        positive = np.bincount(indices, weights=labels, minlength=len(scores))
        occupied = np.flatnonzero(count)
        video["_hist"] = (occupied, count[occupied], positive[occupied])
        video["_hist_size"] = len(scores)


def _pooled_metric(videos: Sequence[dict], kind: str) -> float | None:
    """Compute pooled frame AUC or AP from score counts instead of sorting repeated frames."""
    if not videos:
        return None
    if kind == "anomaly_auc":
        videos = [video for video in videos if np.any(video["_cached_frames"][1])]
        if not videos:
            return None
    size = videos[0]["_hist_size"]
    indices = np.concatenate([video["_hist"][0] for video in videos])
    count = np.bincount(indices, weights=np.concatenate([video["_hist"][1] for video in videos]), minlength=size)
    positive = np.bincount(indices, weights=np.concatenate([video["_hist"][2] for video in videos]), minlength=size)
    n_positive = positive.sum()
    if not n_positive:
        return None
    if kind == "ap":
        occupied = count[::-1] > 0
        cumulative_positive = np.cumsum(positive[::-1])[occupied]
        cumulative_count = np.cumsum(count[::-1])[occupied]
        return float(np.sum((positive[::-1][occupied] / n_positive) *
                            (cumulative_positive / cumulative_count)))
    negative = count - positive
    n_negative = negative.sum()
    if not n_negative:
        return None
    return float(np.sum(positive * (np.cumsum(negative) - negative / 2)) / (n_positive * n_negative))


def paired_bootstrap(runs: dict[str, Sequence[dict]], metric: Metric, n: int = 1000, seed: int = 0,
                     level: float = 0.95) -> dict:
    """Point values, intervals and paired differences of metric over the runs.

    Returns:
      {"runs": {name: {"point", "low", "high", "draws"}},
       "pairs": {"<b> - <a>": {"diff", "low", "high", "draws"}}}
      point = metric on all videos; diff = point of b - point of a; draws = number of draws used.
      One pair for each a before b in the order of runs. low, high are None when no draw is usable.

    Raises ValueError when the runs differ in video count or in "events" at any index, or when runs is empty.
    """
    _check_runs(runs)
    draws = _draws(next(iter(runs.values())), n, seed)
    points = {name: metric(videos) for name, videos in runs.items()}
    values = {name: [metric([videos[i] for i in draw]) for draw in draws] for name, videos in runs.items()}

    results: dict = {"runs": {}, "pairs": {}}
    for name in runs:
        usable = [value for value in values[name] if value is not None]
        results["runs"][name] = {"point": points[name], **_interval(usable, level)}
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            results["pairs"][f"{b} - {a}"] = _pair(points[a], points[b], values[a], values[b], level)
    return results


def _check_runs(runs: dict[str, Sequence[dict]]) -> None:
    """Raise ValueError unless every run has the same video count and the same "events" at each index."""
    if not runs:
        raise ValueError("runs must not be empty")
    reference = next(iter(runs.values()))
    for videos in runs.values():
        if len(videos) != len(reference):
            raise ValueError("runs must contain the same number of videos")
        for video, expected in zip(videos, reference, strict=True):
            if video["events"] != expected["events"]:
                raise ValueError("runs must contain the same videos in the same order")


def _draws(videos: Sequence[dict], n: int, seed: int) -> list[list[int]]:
    """n draws of video indices: anomaly picks, then normal picks (see the module docstring)."""
    anomaly = [i for i, video in enumerate(videos) if video["events"]]
    normal = [i for i, video in enumerate(videos) if not video["events"]]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        anomaly_picks = rng.integers(0, len(anomaly), size=len(anomaly))
        normal_picks = rng.integers(0, len(normal), size=len(normal))
        draws.append([anomaly[i] for i in anomaly_picks] + [normal[i] for i in normal_picks])
    return draws


def _interval(values: list[float], level: float) -> dict:
    """{"low", "high", "draws"} of the usable draw values; low and high are None without values."""
    if not values:
        return {"low": None, "high": None, "draws": 0}
    q = 100 * (1 - level) / 2
    low, high = np.percentile(values, [q, 100 - q]).tolist()
    return {"low": low, "high": high, "draws": len(values)}


def _pair(point_a: float | None, point_b: float | None, values_a: list, values_b: list, level: float) -> dict:
    """b - a: point difference and the interval of the per-draw differences where both values exist."""
    differences = [vb - va for va, vb in zip(values_a, values_b, strict=True) if va is not None and vb is not None]
    diff = None
    if point_a is not None and point_b is not None:
        diff = point_b - point_a
    return {"diff": diff, **_interval(differences, level)}
