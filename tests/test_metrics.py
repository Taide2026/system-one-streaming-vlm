"""Unit metrics: section 12 metrics and the threshold rule."""
from pathlib import Path

import pytest

from s1stream.metrics import auroc, brier, ece, frame_auc, summarize_alarms, video_alarms
from s1stream.rules.threshold import Threshold
from s1stream.types import VideoItem, Window

FPS = 30.0


def item(n_frames: int, events=()) -> VideoItem:
    return VideoItem(name="v", path=Path("v.mp4"), category="X", fps=FPS, n_frames=n_frames, events=tuple(events))


def windows(n: int, length: float = 2.0, labels=None) -> list[Window]:
    labels = labels or [0] * n
    return [Window(video="v", index=i, start_s=i * length, end_s=(i + 1) * length,
                   frame_indices=(), label=labels[i]) for i in range(n)]


def test_auroc_separated_and_constant():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == pytest.approx(0.5)


def test_frame_auc_uses_window_scores_and_skips_uncovered_frames():
    # 5 s video (150 frames), event frames 60-119 (2-4 s). Windows 0-2 s and 2-4 s; frames 120-149 are not covered.
    video = item(150, [(60, 119)])
    wins = windows(2, labels=[0, 1])
    # Uncovered frames would be negatives with no score; if they were counted the result would not be 1.0.
    assert frame_auc([(video, wins, [0.1, 0.9])]) == pytest.approx(1.0)
    assert frame_auc([(video, wins, [0.9, 0.1])]) == pytest.approx(0.0)


def test_ece_calibrated_and_hand_computed():
    assert ece([0.25, 0.25, 0.25, 0.25], [1, 0, 0, 0], bins=4) == pytest.approx(0.0)
    # Bin [0.75, 1.0]: 2 scores, mean 0.9, mean label 0.5 -> |0.4| * 2/4. Bin [0, 0.25): 2 scores, mean 0.1, label 0 -> 0.1 * 2/4.
    assert ece([0.9, 0.9, 0.1, 0.1], [1, 0, 0, 0], bins=4) == pytest.approx(0.25)


def test_brier_hand_computed():
    # (0.9-1)^2 + (0.2-0)^2 + (0.6-1)^2 + (0.0-0)^2 = 0.01 + 0.04 + 0.16 + 0 = 0.21; / 4
    assert brier([0.9, 0.2, 0.6, 0.0], [1, 0, 1, 0]) == pytest.approx(0.0525)


def test_detected_event_delay():
    # Event 4-6 s = frames 120-179. Windows 0-2, 2-4, 4-6, 6-8 s. First alarm on window 4-6 s.
    video = item(240, [(120, 179)])
    res = video_alarms(video, windows(4, labels=[0, 0, 1, 0]), [False, False, True, True])
    assert res.delays_s == (pytest.approx(2.0),)


def test_missed_event_has_no_delay():
    video = item(240, [(120, 179)])
    res = video_alarms(video, windows(4, labels=[0, 0, 1, 0]), [False, False, False, False])
    assert res.delays_s == (None,)
    assert summarize_alarms([res])["detection_rate"] == pytest.approx(0.0)
    assert summarize_alarms([res])["mean_delay_s"] is None


def test_false_alarms_per_hour():
    # Normal video, 4 windows of 2 s = 8 s negative time, 2 alarms -> 2 / (8 / 3600) = 900 per hour.
    res = video_alarms(item(240), windows(4), [True, False, True, False])
    assert res.false_alarms == 2
    assert res.negative_s == pytest.approx(8.0)
    assert summarize_alarms([res])["false_alarms_per_hour"] == pytest.approx(900.0)


def test_threshold_rule_one_sequence_per_threshold_and_monotone_detection():
    rule = Threshold(start=0.0, stop=1.0, step=0.25)
    assert rule.settings() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    scores = [0.1, 0.3, 0.6, 0.9]
    seqs = rule.alarms(scores)
    assert len(seqs) == 5
    assert seqs[2] == [False, False, True, True]
    video = item(240, [(120, 179)])
    wins = windows(4, labels=[0, 0, 1, 0])
    rates = [summarize_alarms([video_alarms(video, wins, s)])["detection_rate"] for s in seqs]
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_classification_hand_computed_and_no_positive_predictions():
    from s1stream.metrics import classification
    row = classification([0.9, 0.6, 0.4, 0.1], [1, 0, 1, 0], 0.5)
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 1, 1, 1)
    assert (row["accuracy"], row["precision"], row["recall"], row["f1"]) == pytest.approx((0.5, 0.5, 0.5, 0.5))
    none = classification([0.9, 0.6, 0.4, 0.1], [1, 0, 1, 0], 1.0)
    assert (none["tp"], none["fp"]) == (0, 0)
    assert none["precision"] is None
    assert none["f1"] is None
    assert none["recall"] == pytest.approx(0.0)
