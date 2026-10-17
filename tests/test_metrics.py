"""Unit latency: latency and classification metrics on saved values."""
import pytest

from s1stream.metrics import auc, classification, frame_auc, latency

MEASURED = ("ms_per_frame", "p50_ms", "p95_ms", "p99_ms", "rtf_p95")


def test_latency_percentiles_leave_out_the_first_window():
    # After the warm-up window: [1, 2, 3, 4] s. Linear percentiles: 2.5, 3.85, 3.97 s.
    result = latency([9.0, 1.0, 2.0, 3.0, 4.0], frames_per_window=8, budget_s=2.0)
    assert result["first_window_ms"] == pytest.approx(9000.0)
    assert result["p50_ms"] == pytest.approx(2500.0)
    assert result["p95_ms"] == pytest.approx(3850.0)
    assert result["p99_ms"] == pytest.approx(3970.0)


def test_latency_ms_per_frame():
    # 10 s over 4 windows of 8 frames = 0.3125 s per frame.
    result = latency([9.0, 1.0, 2.0, 3.0, 4.0], frames_per_window=8, budget_s=2.0)
    assert result["ms_per_frame"] == pytest.approx(312.5)


def test_latency_rtf_p95_is_p95_over_budget():
    result = latency([5.0, 1.0, 1.0], frames_per_window=8, budget_s=2.0)
    assert result["budget_ms"] == pytest.approx(2000.0)
    assert result["p95_ms"] == pytest.approx(1000.0)
    assert result["rtf_p95"] == pytest.approx(0.5)


def test_latency_with_one_or_no_window():
    one = latency([1.5], frames_per_window=8, budget_s=2.0)
    assert one["first_window_ms"] == pytest.approx(1500.0)
    assert all(one[key] is None for key in MEASURED)
    none = latency([], frames_per_window=8, budget_s=2.0)
    assert none["first_window_ms"] is None
    assert all(none[key] is None for key in MEASURED)
    assert none["budget_ms"] == pytest.approx(2000.0)


def test_classification_hand_computed():
    # threshold 0.5: predictions [1, 1, 0, 0, 0] vs labels [1, 0, 1, 1, 0] -> tp 1, fp 1, fn 2, tn 1.
    row = classification([0.9, 0.6, 0.4, 0.3, 0.1], [1, 0, 1, 1, 0], 0.5)
    assert row["threshold"] == 0.5
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 1, 2, 1)
    assert row["accuracy"] == pytest.approx(0.4)
    assert row["recall"] == pytest.approx(1 / 3)
    assert row["fnr"] == pytest.approx(2 / 3)
    assert "precision" not in row and "f1" not in row


def test_classification_without_abnormal_windows():
    row = classification([0.9, 0.1], [0, 0], 0.5)
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (0, 1, 0, 1)
    assert row["recall"] is None
    assert row["fnr"] is None


def test_classification_specificity_and_balanced_accuracy():
    # Same case as above: recall 1/3, specificity tn / (tn + fp) = 1/2.
    row = classification([0.9, 0.6, 0.4, 0.3, 0.1], [1, 0, 1, 1, 0], 0.5)
    assert row["specificity"] == pytest.approx(0.5)
    assert row["balanced_accuracy"] == pytest.approx((1 / 3 + 0.5) / 2)
    assert classification([0.9, 0.1], [0, 0], 0.5)["balanced_accuracy"] is None


def test_auc_counts_ties_as_half():
    # Pairs (abnormal, normal): (0.8, 0.2) win, (0.8, 0.5) win, (0.5, 0.2) win, (0.5, 0.5) tie -> 3.5 / 4.
    assert auc([0.8, 0.5, 0.5, 0.2], [1, 1, 0, 0]) == pytest.approx(0.875)
    assert auc([0.1, 0.9], [1, 0]) == pytest.approx(0.0)
    assert auc([0.3, 0.4], [0, 0]) is None


def test_frame_auc_spreads_window_scores_and_drops_uncovered_frames():
    # 10 frames, event 2-5; windows 0-4 (score 0.9) and 5-8 (score 0.1); frame 9 is uncovered.
    # Frames 0-4 score 0.9 (labels 0,0,1,1,1), frames 5-8 score 0.1 (labels 1,0,0,0).
    # Abnormal: 3 at 0.9, 1 at 0.1. Normal: 2 at 0.9, 3 at 0.1.
    # Wins 3*3 = 9, ties 3*2 + 1*3 = 9 -> (9 + 4.5) / 20.
    video = {"n_frames": 10, "events": [[2, 5]], "windows": [(0, 4, 0.9), (5, 8, 0.1)]}
    assert frame_auc([video]) == pytest.approx(13.5 / 20)
