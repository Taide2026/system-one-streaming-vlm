"""Unit in-video-metrics: per-video AUCs, pooled anomaly AUC, and recall at false-alarm budget."""
import pytest

from s1stream.metrics import (
    anomaly_auc,
    macro_auc,
    median_video_auc,
    recall_at_fa_per_hour,
    video_aucs,
    within_auc,
)


def test_video_aucs_is_one_when_event_frames_score_above_other_frames():
    video = {
        "n_frames": 4,
        "events": [[2, 3]],
        "windows": [(0, 0, 0.1), (1, 1, 0.2), (2, 2, 0.8), (3, 3, 0.9)],
    }
    scores = video_aucs([video])
    assert len(scores) == 1
    assert scores[0] == pytest.approx(1.0)


def test_video_aucs_returns_none_for_normal_and_all_event_videos():
    normal = {"n_frames": 3, "events": [], "windows": [(0, 0, 0.1), (1, 1, 0.9), (2, 2, 0.4)]}
    # Frame 3 is uncovered; covered frames 0-2 all lie inside event [0, 2].
    all_in_event = {"n_frames": 4, "events": [[0, 2]], "windows": [(0, 0, 0.2), (1, 1, 0.8), (2, 2, 0.5)]}
    assert video_aucs([normal, all_in_event]) == [None, None]


def test_video_aucs_counts_tied_event_and_normal_frame_as_half():
    video = {"n_frames": 2, "events": [[1, 1]], "windows": [(0, 0, 0.7), (1, 1, 0.7)]}
    scores = video_aucs([video])
    assert len(scores) == 1
    assert scores[0] == pytest.approx(0.5)


def test_macro_auc_averages_defined_video_aucs():
    # Video 1: 2 normal (0.1, 0.2) and 2 abnormal (0.8, 0.9) -> AUC 1.0.
    video_1 = {
        "n_frames": 4,
        "events": [[2, 3]],
        "windows": [(0, 0, 0.1), (1, 1, 0.2), (2, 2, 0.8), (3, 3, 0.9)],
    }
    # Video 2: 10 normal at 0.5, 5 abnormal at 0.9, 5 abnormal at 0.1 -> 50 / 100 = 0.5.
    windows_2 = (
        [(i, i, 0.5) for i in range(10)]
        + [(i, i, 0.9) for i in range(10, 15)]
        + [(i, i, 0.1) for i in range(15, 20)]
    )
    video_2 = {"n_frames": 20, "events": [[10, 19]], "windows": windows_2}
    assert macro_auc([video_1, video_2]) == pytest.approx(0.75)


def test_within_auc_weights_video_aucs_by_frame_pair_counts():
    # Same two videos as above: 2 x 2 = 4 pairs at AUC 1.0, and 10 x 10 = 100 pairs at AUC 0.5.
    video_1 = {
        "n_frames": 4,
        "events": [[2, 3]],
        "windows": [(0, 0, 0.1), (1, 1, 0.2), (2, 2, 0.8), (3, 3, 0.9)],
    }
    windows_2 = (
        [(i, i, 0.5) for i in range(10)]
        + [(i, i, 0.9) for i in range(10, 15)]
        + [(i, i, 0.1) for i in range(15, 20)]
    )
    video_2 = {"n_frames": 20, "events": [[10, 19]], "windows": windows_2}
    assert within_auc([video_1, video_2]) == pytest.approx((4 * 1.0 + 100 * 0.5) / 104)


def test_median_video_auc_returns_median_of_defined_video_aucs():
    # Each video has 10 normal frames at 0.05..0.95 and 1 event frame scoring 0.9, 0.4, or 0.6.
    normal_windows = [(i, i, (i + 0.5) / 10) for i in range(10)]
    video_09 = {"n_frames": 11, "events": [[10, 10]], "windows": normal_windows + [(10, 10, 0.9)]}
    video_04 = {"n_frames": 11, "events": [[10, 10]], "windows": normal_windows + [(10, 10, 0.4)]}
    video_06 = {"n_frames": 11, "events": [[10, 10]], "windows": normal_windows + [(10, 10, 0.6)]}
    assert median_video_auc([video_09, video_04, video_06]) == pytest.approx(0.6)


def test_high_scoring_normal_video_does_not_change_macro_within_or_anomaly_auc():
    # Video 1: 1x1 pair, AUC 1.0. Video 2: 2x2 pairs, AUC 0.75. Pooled abnormal vs normal: 8 / 9.
    video_1 = {"n_frames": 2, "events": [[1, 1]], "windows": [(0, 0, 0.1), (1, 1, 0.9)]}
    video_2 = {
        "n_frames": 4,
        "events": [[2, 3]],
        "windows": [(0, 0, 0.2), (1, 1, 0.6), (2, 2, 0.4), (3, 3, 0.8)],
    }
    high_normal = {"n_frames": 3, "events": [], "windows": [(0, 0, 0.95), (1, 1, 0.98), (2, 2, 0.99)]}
    base = [video_1, video_2]
    with_normal = [video_1, video_2, high_normal]
    base_macro = macro_auc(base)
    base_within = within_auc(base)
    base_anomaly = anomaly_auc(base)
    assert base_macro == pytest.approx(0.875)
    assert base_within == pytest.approx(0.8)
    assert base_anomaly == pytest.approx(8 / 9)
    assert macro_auc(with_normal) == pytest.approx(base_macro)
    assert within_auc(with_normal) == pytest.approx(base_within)
    assert anomaly_auc(with_normal) == pytest.approx(base_anomaly)


def test_shifted_second_anomaly_video_lowers_anomaly_auc_while_macro_auc_stays_one():
    # Both videos have per-video AUC 1.0, but shifted_video's normal frame (0.8) beats low_video's event frame (0.2).
    low_video = {"n_frames": 2, "events": [[1, 1]], "windows": [(0, 0, 0.1), (1, 1, 0.2)]}
    shifted_video = {"n_frames": 2, "events": [[1, 1]], "windows": [(0, 0, 0.8), (1, 1, 0.9)]}
    first_anomaly = anomaly_auc([low_video])
    both_anomaly = anomaly_auc([low_video, shifted_video])
    assert first_anomaly == pytest.approx(1.0)
    assert both_anomaly == pytest.approx(0.75)
    assert both_anomaly < first_anomaly
    assert macro_auc([low_video]) == pytest.approx(1.0)
    assert macro_auc([low_video, shifted_video]) == pytest.approx(1.0)


def test_pooled_and_summary_aucs_return_none_when_no_video_has_both_labels():
    normal = {"n_frames": 2, "events": [], "windows": [(0, 0, 0.2), (1, 1, 0.8)]}
    all_event = {"n_frames": 2, "events": [[0, 1]], "windows": [(0, 0, 0.3), (1, 1, 0.9)]}
    for videos in ([], [normal], [all_event], [normal, all_event]):
        assert macro_auc(videos) is None
        assert median_video_auc(videos) is None
        assert within_auc(videos) is None
        assert anomaly_auc(videos) is None


def test_recall_at_fa_per_hour_uses_second_highest_normal_window_for_one_hour():
    # Normal videos total 3600 frames at 1.0 fps = 1.0 hour -> k = 1 -> threshold is 2nd highest normal score (0.7).
    normal_1 = {
        "n_frames": 1800,
        "events": [],
        "windows": [(0, 599, 0.2), (600, 1199, 0.7), (1200, 1799, 0.4)],
    }
    normal_2 = {
        "n_frames": 1800,
        "events": [],
        "windows": [(0, 599, 0.9), (600, 1199, 0.1), (1200, 1799, 0.5)],
    }
    # Window (0, 599, 0.99) precedes event [900, 2999]; of the 4 event windows, 0.95 and 0.8 exceed 0.7 -> 2 / 4.
    anomaly = {
        "n_frames": 3000,
        "events": [[900, 2999]],
        "windows": [
            (0, 599, 0.99),
            (600, 1199, 0.95),
            (1200, 1799, 0.8),
            (1800, 2399, 0.7),
            (2400, 2999, 0.6),
        ],
    }
    recall = recall_at_fa_per_hour([normal_1, normal_2, anomaly], fps=1.0, fa_per_hour=1.0)
    assert recall == pytest.approx(0.5)


def test_recall_at_fa_per_hour_returns_none_without_normal_videos_or_event_windows():
    normal = {
        "n_frames": 3600,
        "events": [],
        "windows": [(0, 599, 0.2), (600, 1199, 0.8), (1200, 1799, 0.4)],
    }
    anomaly = {
        "n_frames": 1200,
        "events": [[0, 599]],
        "windows": [(0, 599, 0.9), (600, 1199, 0.1)],
    }
    missed_event = {
        "n_frames": 1200,
        "events": [[600, 1199]],
        "windows": [(0, 599, 0.95)],
    }
    assert recall_at_fa_per_hour([anomaly], fps=1.0, fa_per_hour=1.0) is None
    assert recall_at_fa_per_hour([normal], fps=1.0, fa_per_hour=1.0) is None
    assert recall_at_fa_per_hour([normal, missed_event], fps=1.0, fa_per_hour=1.0) is None
