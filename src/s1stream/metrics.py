"""Metrics on saved scores and times; no model, no GPU.

Latency (one value L per window, the runner's `seconds`: frame read + preprocessing + backbone + head):
- The first window is warm-up. Every latency field except first_window_ms leaves it out.
- Percentiles use numpy.percentile with its default (linear) method.
- The budget is the window stride: a window must be done before the next one is ready.

Classification (window level): a window is predicted abnormal when its score >= threshold. Positive = abnormal.
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

