"""Metrics on saved scores and times; no model, no GPU.

Latency (one value L per window, the runner's `seconds`: frame read + preprocessing + backbone + head):
- The first window is warm-up. Every latency field except first_window_ms leaves it out.
- Percentiles use numpy.percentile with its default (linear) method.
- The budget is the window stride: a window must be done before the next one is ready.

Classification (window level): a window is predicted abnormal when its score >= threshold. Positive = abnormal.
"""
from __future__ import annotations

from collections.abc import Sequence


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
    raise NotImplementedError


def classification(scores: Sequence[float], labels: Sequence[int], threshold: float) -> dict[str, float | int | None]:
    """Window classification at one threshold.

    Returns threshold, tp, fp, fn, tn, accuracy = (tp + tn) / all, recall = tp / (tp + fn), fnr = fn / (tp + fn).
    accuracy is None when there are no windows; recall and fnr are None when no window is abnormal.
    """
    raise NotImplementedError
