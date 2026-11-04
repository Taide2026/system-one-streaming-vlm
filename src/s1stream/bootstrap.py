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

Metric = Callable[[Sequence[dict]], float | None]


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
        for video, expected in zip(videos, reference):
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
    differences = [vb - va for va, vb in zip(values_a, values_b) if va is not None and vb is not None]
    diff = None
    if point_a is not None and point_b is not None:
        diff = point_b - point_a
    return {"diff": diff, **_interval(differences, level)}
