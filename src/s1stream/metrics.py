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
    raise NotImplementedError


def frame_auc(videos: Sequence[tuple[VideoItem, Sequence[Window], Sequence[float]]]) -> float:
    """Frame-level AUC over all videos.

    Each frame inside a window gets that window's score; with overlapping windows, the latest window wins.
    A frame is positive when it lies inside an event range (inclusive). Frames outside every window are skipped.
    A window covers frames round(start_s * fps) to round(end_s * fps) - 1.
    """
    raise NotImplementedError


def ece(scores: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]. A score of 1.0 goes into the last bin.

    ECE = sum over non-empty bins of (bin size / n) * |mean score - mean label|.
    """
    raise NotImplementedError


def brier(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mean of (score - label) ** 2."""
    raise NotImplementedError


def video_alarms(item: VideoItem, windows: Sequence[Window], alarms: Sequence[bool]) -> VideoAlarms:
    """Apply the alarm rules of this module to one video. alarms has one bool per window, in window order."""
    raise NotImplementedError


def summarize_alarms(results: Sequence[VideoAlarms]) -> dict[str, float | None]:
    """Combine videos into: detection_rate, mean_delay_s, median_delay_s, false_alarms_per_hour.

    detection_rate = detected events / all events (None when there are no events).
    mean_delay_s and median_delay_s use detected events only (None when none was detected).
    false_alarms_per_hour = all false alarms / (all negative_s / 3600) (None when negative_s is 0).
    """
    raise NotImplementedError
