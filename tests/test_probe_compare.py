"""Unit probe-compare: bootstrap, layer summary, CKA, outputs (finetune-layer-probing decisions 9, 11, 12).
CPU, synthetic scores."""
import json
from pathlib import Path

import numpy as np

from s1stream import metrics
from s1stream.probe.compare import bootstrap, frame_metric, layer_summary, linear_cka, run_compare
from s1stream.probe.fit import KINDS, METHODS
from s1stream.types import VideoItem
from s1stream.windows import window_frame_range

STATES = 3


def make_test_set(n_anomaly: int = 3, n_normal: int = 3, windows: int = 4):
    """Videos at 10 fps with 2 s windows; anomaly videos have an event over frames 20-39 (window 1)."""
    items, meta = {}, []
    for v in range(n_anomaly + n_normal):
        anomaly = v < n_anomaly
        name = f"v{v}.mp4"
        events = ((20, 39),) if anomaly else ()
        items[name] = VideoItem(name=name, path=Path(name), category="Abuse" if anomaly else "Normal", fps=10.0,
                                n_frames=windows * 20, events=events)
        for w in range(windows):
            meta.append({"i": len(meta), "video": name, "window": w, "start_s": 2.0 * w, "end_s": 2.0 * w + 2.0,
                         "label": int(anomaly and w == 1), "p_yes": 0.5})
    return items, meta


def scores_array(meta: list[dict], seed: int, signal: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.array([row["label"] for row in meta], dtype=float)
    return (rng.normal(size=(len(METHODS), len(KINDS), STATES, len(meta))) + signal * labels).astype(np.float32)


def test_linear_cka_identity_and_invariance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 20))
    rotation, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    assert abs(linear_cka(x, x) - 1.0) < 1e-9
    assert abs(linear_cka(x, 3.0 * x @ rotation) - 1.0) < 1e-9


def test_linear_cka_independent_is_small():
    rng = np.random.default_rng(1)
    assert linear_cka(rng.normal(size=(1000, 50)), rng.normal(size=(1000, 50))) < 0.1


def test_layer_summary_onset_and_peak():
    assert layer_summary([0.5, 0.6, 0.9, 1.0, 0.95]) == {"onset": 3, "peak": 3, "peak_value": 1.0}


def test_bootstrap_resamples_whole_videos():
    items, meta = make_test_set(n_anomaly=1, n_normal=0)
    scores = {"B": scores_array(meta, 0)}
    p_yes = {"B": np.linspace(0, 1, len(meta))}
    result = bootstrap(scores, p_yes, meta, items, n=50, seed=3)
    for metric in ("frame_auc", "window_auc"):
        entry = result["metrics"][f"B/mil/last/{metric}"]
        assert np.allclose(entry["low"], entry["high"])
    items, meta = make_test_set()
    scores = {"B": scores_array(meta, 0)}
    p_yes = {"B": np.linspace(0, 1, len(meta))}
    assert bootstrap(scores, p_yes, meta, items, n=50, seed=3) == bootstrap(scores, p_yes, meta, items, n=50, seed=3)


def test_paired_difference_interval():
    items, meta = make_test_set(n_anomaly=6, n_normal=6)
    base = scores_array(meta, 0, signal=0.0)
    labels = np.array([row["label"] for row in meta], dtype=np.float32)
    better = base + 10.0 * labels
    p_yes = {name: np.full(len(meta), 0.5) for name in ("A", "B", "C")}
    result = bootstrap({"A": base, "B": base.copy(), "C": better}, p_yes, meta, items, n=200, seed=4)
    same = result["differences"]["A-B/mil/last/frame_auc"]
    assert all(low <= 0.0 <= high for low, high in zip(same["low"], same["high"]))
    gain = result["differences"]["C-B/mil/last/frame_auc"]
    assert all(low > 0.0 for low in gain["low"])


def test_bootstrap_point_equals_frame_metric():
    items, meta = make_test_set(n_anomaly=4, n_normal=4)
    scores = {"B": scores_array(meta, 5, signal=0.7)}
    result = bootstrap(scores, {"B": np.linspace(0, 1, len(meta))}, meta, items, n=20, seed=5)
    for metric in ("frame_auc", "frame_ap", "window_auc"):
        for state in range(STATES):
            expected = frame_metric(scores["B"][0, 0, state], meta, items, metric)
            assert abs(result["metrics"][f"B/mil/last/{metric}"]["value"][state] - expected) < 1e-9
    videos = [{"n_frames": items[name].n_frames, "events": [list(e) for e in items[name].events],
               "windows": [(*window_frame_range(r["start_s"], r["end_s"], 10.0), float(scores["B"][0, 0, 0][r["i"]]))
                           for r in meta if r["video"] == name]} for name in items]
    assert abs(frame_metric(scores["B"][0, 0, 0], meta, items, "frame_auc") - metrics.frame_auc(videos)) < 1e-12


def test_run_compare_writes_outputs(tmp_path):
    items, meta = make_test_set()
    run = tmp_path / "run"
    rng = np.random.default_rng(6)
    for seed, name in enumerate(("A", "B", "C")):
        features = tmp_path / "features" / name / "test"
        features.mkdir(parents=True)
        for kind in KINDS:
            np.save(features / f"{kind}.npy", rng.normal(size=(len(meta), STATES, 8)).astype(np.float16))
        fit = run / f"fit-{name}"
        fit.mkdir(parents=True)
        np.save(fit / "test_scores.npy", scores_array(meta, seed))
        (fit / "test_meta.jsonl").write_text("".join(json.dumps(row) + "\n" for row in meta))
        (fit / "fit.json").write_text(json.dumps({"methods": list(METHODS), "kinds": list(KINDS), "n_states": STATES,
                                                  "features": str(features.parent), "choices": {}}))
    out = run_compare(Path("unused.yaml"), run, items=items, log=lambda line: None)
    assert out == run / "compare.json"
    result = json.loads(out.read_text())
    assert result["conditions"] == ["A", "B", "C"] and len(result["cka"]["C-B/last"]) == STATES
    for method in METHODS:
        for kind in KINDS:
            assert (run / f"{method}_{kind}.png").stat().st_size > 0
    assert (run / "cka.png").stat().st_size > 0
