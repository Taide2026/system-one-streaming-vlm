"""Metrics on saved scores and times; no model, no GPU.

Latency (one value L per window, the runner's `seconds`: frame read + preprocessing + backbone + head):
- The first window is warm-up. Every latency field except first_window_ms leaves it out.
- Percentiles use numpy.percentile with its default (linear) method.
- The budget is the window stride: a window must be done before the next one is ready.

Classification (window level): a window is predicted abnormal when its score >= threshold. Positive = abnormal.

AUC (threshold free): ROC-AUC with ties counted as half (the Mann-Whitney U statistic).
- window: one score and one label per window.
- frame: the UCF-Crime convention. Every frame of a window's frame range gets that window's score; a frame is
  abnormal when it lies in an annotated event. Frames that no window covers (the tail shorter than one window)
  are left out. With overlapping windows a frame takes the highest score of the windows covering it.

AP (threshold free): average precision, the area under the precision-recall curve as scikit-learn's
average_precision_score computes it; window and frame as for AUC, on the same frames. XD-Violence's main metric.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def latency(seconds: Sequence[float], frames_per_window: int, budget_s: float) -> dict[str, float | None]:
    """Latency summary of the windows in run order.

    Returns:
      first_window_ms: seconds[0] in ms (None when there are no windows).
      ms_per_frame: sum of L after the first window / (those windows * frames_per_window), in ms.
      p50_ms, p95_ms, p99_ms: percentiles of L after the first window, in ms.
      rtf_p95: p95_ms / budget_ms.
      budget_ms: budget_s in ms (always set).
    Every field except first_window_ms and budget_ms is None when there are fewer than 2 windows.
    """
    values = np.asarray(seconds, dtype=float)
    budget_ms = float(budget_s * 1000)
    first_window_ms = None
    if values.size:
        first_window_ms = float(values[0] * 1000)

    result: dict[str, float | None] = {
        "first_window_ms": first_window_ms,
        "ms_per_frame": None,
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "rtf_p95": None,
        "budget_ms": budget_ms,
    }
    measured = values[1:]
    if measured.size:
        milliseconds = measured * 1000
        result["ms_per_frame"] = float(np.sum(measured) * 1000 / (measured.size * frames_per_window))
        result["p50_ms"] = float(np.percentile(milliseconds, 50))
        result["p95_ms"] = float(np.percentile(milliseconds, 95))
        result["p99_ms"] = float(np.percentile(milliseconds, 99))
        result["rtf_p95"] = float(result["p95_ms"] / budget_ms)
    return result


def classification(scores: Sequence[float], labels: Sequence[int], threshold: float) -> dict[str, float | int | None]:
    """Window classification at one threshold.

    Returns threshold, tp, fp, fn, tn, accuracy = (tp + tn) / all, recall = tp / (tp + fn), fnr = fn / (tp + fn),
    specificity = tn / (tn + fp), balanced_accuracy = (recall + specificity) / 2.
    accuracy is None when there are no windows; recall and fnr are None when no window is abnormal;
    specificity is None when no window is normal; balanced_accuracy is None when recall or specificity is.
    """
    score_values = np.asarray(scores)
    label_values = np.asarray(labels)
    predicted = score_values >= threshold
    actual = label_values == 1

    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    total = int(label_values.size)
    abnormal = tp + fn
    accuracy = float((tp + tn) / total) if total else None
    recall = float(tp / abnormal) if abnormal else None
    fnr = float(fn / abnormal) if abnormal else None
    specificity = float(tn / (tn + fp)) if tn + fp else None
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "recall": recall,
        "fnr": fnr,
        "specificity": specificity,
        "balanced_accuracy": balanced,
    }


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """ROC-AUC = P(score of a random abnormal > score of a random normal) + P(tie) / 2.

    None when every label is the same (or there are none).
    """
    score_values = np.asarray(scores, dtype=float)
    actual = np.asarray(labels) == 1
    positives = int(np.count_nonzero(actual))
    negatives = int(actual.size - positives)
    if not positives or not negatives:
        return None
    order = np.argsort(score_values, kind="mergesort")
    ordered = score_values[order]
    ranks = np.empty(ordered.size, dtype=float)
    start = 0
    while start < ordered.size:  # average rank (1-based) within each run of equal scores
        end = start
        while end < ordered.size and ordered[end] == ordered[start]:
            end += 1
        ranks[start:end] = (start + end + 1) / 2
        start = end
    rank_sum = float(ranks[actual[order]].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def ap(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Average precision: over the distinct scores from the highest, the sum of (recall gained at that score) x
    (precision at that score), as scikit-learn's average_precision_score. None when no label is abnormal."""
    score_values = np.asarray(scores, dtype=float)
    actual = np.asarray(labels) == 1
    positives = int(np.count_nonzero(actual))
    if not positives:
        return None
    order = np.argsort(-score_values, kind="mergesort")
    ordered, hits = score_values[order], actual[order]
    true_positives = np.cumsum(hits)
    last = np.r_[np.nonzero(np.diff(ordered))[0], ordered.size - 1]  # last index of each run of equal scores
    precision = true_positives[last] / (last + 1)
    recall = true_positives[last] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def frame_scores(videos: Sequence[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    """The scores and labels of every covered frame of every video (see the module docstring), or None without
    videos.

    Each video: {"n_frames": int, "events": [[start_frame, end_frame], ...] (inclusive),
                 "windows": [(first_frame, last_frame, score), ...] (inclusive frame ranges)}.
    """
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for video in videos:
        frame_scores = np.full(video["n_frames"], -np.inf)
        for first, last, score in video["windows"]:
            span = slice(max(first, 0), min(last, video["n_frames"] - 1) + 1)
            frame_scores[span] = np.maximum(frame_scores[span], score)
        frame_labels = np.zeros(video["n_frames"], dtype=int)
        for start, end in video["events"]:
            frame_labels[max(start, 0):min(end, video["n_frames"] - 1) + 1] = 1
        covered = np.isfinite(frame_scores)
        scores.append(frame_scores[covered])
        labels.append(frame_labels[covered])
    if not scores:
        return None
    return np.concatenate(scores), np.concatenate(labels)


def frame_auc(videos: Sequence[dict]) -> float | None:
    """Frame-level ROC-AUC over every video (frame_scores)."""
    frames = frame_scores(videos)
    return None if frames is None else auc(*frames)


def frame_ap(videos: Sequence[dict]) -> float | None:
    """Frame-level average precision over every video, on the frames of frame_auc."""
    frames = frame_scores(videos)
    return None if frames is None else ap(*frames)
