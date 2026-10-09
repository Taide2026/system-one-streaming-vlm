"""Metrics of dev doc section 12. Pure functions on saved scores; no model, no GPU.

Alarm rules (decision 8):
- An alarm is a window whose decision is True. Its time is the window end.
- An event is detected when an alarm window overlaps that event by at least 1 frame.
  The delay is (end_s of the first such window) - (event start frame / fps).
- A false alarm is an alarm on a window with label 0.
- Negative time is the summed length (end_s - start_s) of the label-0 windows.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from s1stream.types import VideoItem, Window


@dataclass(frozen=True)
class VideoAlarms:
    """Alarm result of one video for one setting of a decision rule.

    delays_s: one entry per event, None when the event was missed.
    """
    video: str
    delays_s: tuple[float | None, ...]
    false_alarms: int
    negative_s: float


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve. Ties count one half. Raise ValueError when labels hold one class only."""
    values = np.asarray(scores, dtype=float)
    targets = np.asarray(labels, dtype=int)
    positives = targets == 1
    negatives = targets == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both label classes")
    ranks = np.empty(len(values), dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2
        index = end
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def frame_auc(videos: Sequence[tuple[VideoItem, Sequence[Window], Sequence[float]]]) -> float:
    """Frame-level AUC over all videos.

    Each frame inside a window gets that window's score; with overlapping windows, the latest window wins.
    A frame is positive when it lies inside an event range (inclusive). Frames outside every window are skipped.
    A window covers frames round(start_s * fps) to round(end_s * fps) - 1.
    """
    all_scores = []
    all_labels = []
    for item, windows, scores in videos:
        frame_scores: dict[int, float] = {}
        for window, score in zip(windows, scores):
            first = round(window.start_s * item.fps)
            last = round(window.end_s * item.fps) - 1
            for frame in range(first, last + 1):
                frame_scores[frame] = float(score)
        for frame, score in frame_scores.items():
            all_scores.append(score)
            all_labels.append(int(any(start <= frame <= end for start, end in item.events)))
    return auroc(all_scores, all_labels)


def ece(scores: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]. A score of 1.0 goes into the last bin.

    ECE = sum over non-empty bins of (bin size / n) * |mean score - mean label|.
    """
    values = np.asarray(scores, dtype=float)
    targets = np.asarray(labels, dtype=float)
    if len(values) == 0:
        return 0.0
    total = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (values >= lower) & (values < upper)
        if index == bins - 1:
            mask |= values == 1.0
        if mask.any():
            total += mask.sum() / len(values) * abs(values[mask].mean() - targets[mask].mean())
    return float(total)


def brier(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mean of (score - label) ** 2."""
    return float(np.mean((np.asarray(scores, dtype=float) - np.asarray(labels, dtype=float)) ** 2))


def video_alarms(item: VideoItem, windows: Sequence[Window], alarms: Sequence[bool]) -> VideoAlarms:
    """Apply the alarm rules of this module to one video. alarms has one bool per window, in window order."""
    if len(windows) != len(alarms):
        raise ValueError("one alarm value is required per window")
    delays = []
    for event_start, event_end in item.events:
        detection = None
        for window, alarm in zip(windows, alarms):
            first = round(window.start_s * item.fps)
            last = round(window.end_s * item.fps) - 1
            if alarm and first <= event_end and last >= event_start:
                detection = window.end_s - event_start / item.fps
                break
        delays.append(detection)
    false_alarms = sum(bool(alarm) and window.label == 0 for window, alarm in zip(windows, alarms))
    negative_s = sum(window.end_s - window.start_s for window in windows if window.label == 0)
    return VideoAlarms(item.name, tuple(delays), int(false_alarms), float(negative_s))


def summarize_alarms(results: Sequence[VideoAlarms]) -> dict[str, float | None]:
    """Combine videos into: detection_rate, mean_delay_s, median_delay_s, false_alarms_per_hour.

    detection_rate = detected events / all events (None when there are no events).
    mean_delay_s and median_delay_s use detected events only (None when none was detected).
    false_alarms_per_hour = all false alarms / (all negative_s / 3600) (None when negative_s is 0).
    """
    delays = [delay for result in results for delay in result.delays_s if delay is not None]
    events = sum(len(result.delays_s) for result in results)
    detected = len(delays)
    false_alarms = sum(result.false_alarms for result in results)
    negative_s = sum(result.negative_s for result in results)
    return {
        "detection_rate": detected / events if events else None,
        "mean_delay_s": float(np.mean(delays)) if delays else None,
        "median_delay_s": float(np.median(delays)) if delays else None,
        "false_alarms_per_hour": false_alarms / (negative_s / 3600) if negative_s else None,
    }
