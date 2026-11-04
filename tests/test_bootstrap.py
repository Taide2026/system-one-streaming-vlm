"""Unit bootstrap: paired bootstrap over videos, anomaly and normal videos drawn separately."""
import numpy as np
import pytest

from s1stream.bootstrap import paired_bootstrap


def _video(vid: int, value: float, anomaly: bool) -> dict:
    """A one-window video; "value" is read by the test metrics below."""
    events = [[0, 0]] if anomaly else []
    return {"id": vid, "value": value, "n_frames": 1, "events": events, "windows": [(0, 0, value)]}


def _mean_value(videos) -> float | None:
    return float(np.mean([v["value"] for v in videos])) if videos else None


def _runs():
    a = [_video(0, 0.1, True), _video(1, 0.5, True), _video(2, 0.2, False), _video(3, 0.4, False), _video(4, 0.9, False)]
    a = [dict(v, run="A") for v in a]
    b = [dict(v, value=v["value"] + 0.1, run="B") for v in a]
    return {"A": a, "B": b}


def test_point_is_the_metric_on_all_videos():
    result = paired_bootstrap(_runs(), _mean_value, n=50, seed=1)
    assert result["runs"]["A"]["point"] == pytest.approx(_mean_value(_runs()["A"]))
    assert result["runs"]["B"]["point"] == pytest.approx(_mean_value(_runs()["B"]))


def test_identical_runs_give_zero_difference_and_zero_width_pair_interval():
    runs = {"A": _runs()["A"], "C": _runs()["A"]}
    pair = paired_bootstrap(runs, _mean_value, n=50, seed=1)["pairs"]["C - A"]
    assert pair["diff"] == pytest.approx(0.0)
    assert pair["low"] == pytest.approx(0.0)
    assert pair["high"] == pytest.approx(0.0)


def test_pair_key_and_diff_follow_run_order():
    pairs = paired_bootstrap(_runs(), _mean_value, n=50, seed=1)["pairs"]
    assert list(pairs) == ["B - A"]
    assert pairs["B - A"]["diff"] == pytest.approx(0.1)


def test_every_draw_keeps_the_anomaly_and_normal_counts():
    counts = []

    def record(videos):
        counts.append((sum(1 for v in videos if v["events"]), sum(1 for v in videos if not v["events"])))
        return 0.0

    paired_bootstrap({"A": _runs()["A"]}, record, n=30, seed=2)
    assert len(counts) == 31  # the point value and 30 draws
    assert set(counts) == {(2, 3)}


def test_all_runs_see_the_same_videos_in_each_draw():
    seen = {"A": [], "B": []}
    runs = _runs()

    def record(videos):
        seen[videos[0]["run"]].append([v["id"] for v in videos])
        return 0.0

    paired_bootstrap(runs, record, n=20, seed=3)
    assert seen["A"] == seen["B"]


def test_intervals_follow_the_documented_draw_order_and_percentiles():
    runs = {"A": _runs()["A"]}
    result = paired_bootstrap(runs, _mean_value, n=200, seed=7, level=0.9)["runs"]["A"]
    anomaly = [v for v in runs["A"] if v["events"]]
    normal = [v for v in runs["A"] if not v["events"]]
    rng = np.random.default_rng(7)
    values = []
    for _ in range(200):
        picks_a = rng.integers(0, len(anomaly), size=len(anomaly))
        picks_n = rng.integers(0, len(normal), size=len(normal))
        values.append(_mean_value([anomaly[i] for i in picks_a] + [normal[i] for i in picks_n]))
    assert result["low"] == pytest.approx(np.percentile(values, 5))
    assert result["high"] == pytest.approx(np.percentile(values, 95))
    assert result["draws"] == 200


def test_same_seed_gives_the_same_result():
    first = paired_bootstrap(_runs(), _mean_value, n=40, seed=5)
    second = paired_bootstrap(_runs(), _mean_value, n=40, seed=5)
    assert first == second


def test_draws_with_none_are_left_out():
    def none_without_video_1(videos):
        return 1.0 if any(v["id"] == 1 for v in videos) else None

    result = paired_bootstrap({"A": _runs()["A"]}, none_without_video_1, n=100, seed=4)["runs"]["A"]
    assert 0 < result["draws"] < 100
    assert result["low"] == pytest.approx(1.0)
    assert result["high"] == pytest.approx(1.0)


def test_mismatched_runs_raise_value_error():
    runs = _runs()
    with pytest.raises(ValueError):
        paired_bootstrap({"A": runs["A"], "B": runs["B"][:4]}, _mean_value, n=5)
    changed = [dict(v) for v in runs["B"]]
    changed[2]["events"] = [[0, 0]]
    with pytest.raises(ValueError):
        paired_bootstrap({"A": runs["A"], "B": changed}, _mean_value, n=5)
    with pytest.raises(ValueError):
        paired_bootstrap({}, _mean_value, n=5)
