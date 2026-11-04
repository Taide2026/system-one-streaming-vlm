"""Unit in-video-report: every metric with bootstrap intervals in report.json and encoder_compare.json."""
import json
from pathlib import Path

import numpy as np
import pytest

from s1stream import metrics
from s1stream.bootstrap import VIDEO_METRICS, metric_summary, paired_bootstrap
from s1stream.encoder.compare import BASELINE, run_encoder_compare, select_state
from s1stream.report import frame_videos, report
from test_encoder_compare import make_test_set, write_fits

FUNCTIONS = {
    "frame_auc": metrics.frame_auc,
    "frame_ap": metrics.frame_ap,
    "macro_auc": metrics.macro_auc,
    "median_video_auc": metrics.median_video_auc,
    "within_auc": metrics.within_auc,
    "anomaly_auc": metrics.anomaly_auc,
}


def _video(event: bool, scores: list[float]) -> dict:
    """1-frame windows; an event video has its event on the last 2 frames."""
    n = len(scores)
    events = [[n - 2, n - 1]] if event else []
    return {"n_frames": n, "events": events, "windows": [(i, i, s) for i, s in enumerate(scores)]}


def _runs() -> dict[str, list[dict]]:
    a = [_video(True, [0.1, 0.3, 0.6, 0.9]), _video(True, [0.5, 0.2, 0.4, 0.8]), _video(True, [0.7, 0.1, 0.3, 0.2]),
         _video(False, [0.2, 0.3, 0.1, 0.4]), _video(False, [0.6, 0.1, 0.2, 0.3])]
    b = [_video(True, [0.2, 0.1, 0.8, 0.9]), _video(True, [0.1, 0.2, 0.7, 0.6]), _video(True, [0.4, 0.3, 0.5, 0.6]),
         _video(False, [0.1, 0.2, 0.1, 0.3]), _video(False, [0.2, 0.1, 0.3, 0.2])]
    return {"A": a, "B": b}


def _metric(name: str, fps: float):
    if name == "recall_at_1fa_h":
        return lambda videos: metrics.recall_at_fa_per_hour(videos, fps, 1.0)
    return FUNCTIONS[name]


def test_summary_values_equal_the_metric_functions():
    runs = _runs()
    summary = metric_summary(runs, fps=0.001, n=20, seed=3)
    assert set(summary["runs"]["A"]) == set(VIDEO_METRICS)
    for name in VIDEO_METRICS:
        assert summary["runs"]["A"][name]["value"] == pytest.approx(_metric(name, 0.001)(runs["A"]))


def test_summary_intervals_equal_paired_bootstrap():
    runs = _runs()
    summary = metric_summary(runs, fps=0.001, n=30, seed=4)
    for name in VIDEO_METRICS:
        expected = paired_bootstrap(runs, _metric(name, 0.001), n=30, seed=4)["runs"]["B"]
        assert summary["runs"]["B"][name]["low"] == pytest.approx(expected["low"])
        assert summary["runs"]["B"][name]["high"] == pytest.approx(expected["high"])


def test_summary_differences_against_baseline_equal_paired_bootstrap_pairs():
    runs = _runs()
    summary = metric_summary(runs, fps=0.001, baseline="A", n=30, seed=5)
    assert list(summary["differences"]) == ["B"]
    for name in VIDEO_METRICS:
        pair = paired_bootstrap(runs, _metric(name, 0.001), n=30, seed=5)["pairs"]["B - A"]
        got = summary["differences"]["B"][name]
        assert got["value"] == pytest.approx(pair["diff"])
        assert got["low"] == pytest.approx(pair["low"])
        assert got["high"] == pytest.approx(pair["high"])


def test_summary_without_baseline_has_no_differences():
    assert metric_summary(_runs(), fps=0.001, n=10, seed=6)["differences"] == {}


def _write_run(run_dir: Path, fps_values: list[float]) -> None:
    """A runner output folder with 4 videos of 3 windows (2 s each); videos 0 and 1 have an event in window 1."""
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("window: {frames: 8, stride_s: 2.0}\ndecision_rules: []\n")
    (run_dir / "run.json").write_text(json.dumps({"peak_gpu_gb": 1.0, "load_s": 1.0}))
    manifest, rows = [], []
    for v, fps in enumerate(fps_values):
        name = f"v{v}.mp4"
        frames_per_window = round(2 * fps)
        events = [[frames_per_window, 2 * frames_per_window - 1]] if v < 2 else []
        manifest.append({"name": name, "fps": fps, "n_frames": 3 * frames_per_window, "events": events})
        for w in range(3):
            score = 0.1 * (v + 1) + (0.5 if (v < 2 and w == 1) else 0.0) + 0.05 * w
            rows.append({"video": name, "window": w, "start_s": 2.0 * w, "end_s": 2.0 * w + 2.0,
                         "label": int(v < 2 and w == 1), "score": score, "seconds": 0.1})
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_report_puts_every_metric_in_one_metrics_block(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, [10.0, 10.0, 10.0, 10.0])
    result = report(run_dir)
    assert set(result["metrics"]) == set(VIDEO_METRICS)
    for entry in result["metrics"].values():
        assert set(entry) == {"value", "low", "high"}
    for old_key in ("auc", "ap", "in_video"):
        assert old_key not in result
    rows = [json.loads(line) for line in (run_dir / "scores.jsonl").read_text().splitlines()]
    manifest = {video["name"]: video for video in json.loads((run_dir / "manifest.json").read_text())}
    assert result["metrics"]["macro_auc"]["value"] == pytest.approx(metrics.macro_auc(frame_videos(rows, manifest)))
    assert result["metrics"]["frame_auc"]["value"] == pytest.approx(metrics.frame_auc(frame_videos(rows, manifest)))
    assert json.loads((run_dir / "report.json").read_text())["metrics"] == result["metrics"]


def test_report_raises_value_error_when_fps_differ(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, [10.0, 10.0, 5.0, 10.0])
    with pytest.raises(ValueError):
        report(run_dir)


def test_encoder_compare_entry_has_state_metrics_difference_and_curve(tmp_path):
    items, meta = make_test_set()
    run_dir = write_fits(tmp_path, meta)
    payload = json.loads(run_encoder_compare(Path("unused.yaml"), run_dir, items=items, log=lambda line: None)
                         .read_text())
    entry = payload["results"]["dinov3-vitl/patch_mean/holmes"]
    baseline = payload["results"][f"{BASELINE[0]}/{BASELINE[1]}/holmes"]
    assert set(entry) == {"n_states", "state", "baseline_state", "metrics", "difference", "curve"}
    assert set(entry["metrics"]) == set(VIDEO_METRICS)
    assert set(entry["difference"]) == set(VIDEO_METRICS)
    fit = json.loads((run_dir / "fit-dinov3-vitl" / "fit.json").read_text())
    state = select_state(fit, "holmes", "patch_mean")
    assert entry["state"] == state
    assert entry["baseline_state"] == baseline["state"]
    scores = np.load(run_dir / "fit-dinov3-vitl" / "test_scores.npy")[1, 0, state]
    rows = [dict(row, score=float(s)) for row, s in zip(meta, scores)]
    manifest = {name: {"fps": item.fps, "n_frames": item.n_frames, "events": [list(e) for e in item.events]}
                for name, item in items.items()}
    videos = frame_videos(rows, manifest)
    for name, function in FUNCTIONS.items():
        assert entry["metrics"][name]["value"] == pytest.approx(function(videos))
        assert entry["difference"][name]["value"] == pytest.approx(
            entry["metrics"][name]["value"] - baseline["metrics"][name]["value"])
        assert baseline["difference"][name] == {"value": 0.0, "low": 0.0, "high": 0.0}


def test_encoder_compare_overwrites_an_existing_encoder_compare_json(tmp_path):
    items, meta = make_test_set()
    run_dir = write_fits(tmp_path, meta)
    (run_dir / "encoder_compare.json").write_text("{}")
    out = run_encoder_compare(Path("unused.yaml"), run_dir, items=items, log=lambda line: None)
    assert "results" in json.loads(out.read_text())
