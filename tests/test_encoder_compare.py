"""Unit encoder-compare: state selection, paired bootstrap differences against baseline, head fits, CLI."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from s1stream import cli
from s1stream.encoder.compare import BASELINE, N_BOOT, SEED, run_encoder_compare, select_state
from s1stream.probe.fit import METHODS
from s1stream.types import VideoItem

FITS: dict[str, tuple[tuple[str, ...], int]] = {
    "qwen35-vit": (("patch_mean", "patch_max"), 3),
    "dinov3-vitl": (("patch_mean", "cls_mean"), 3),
    "siglip2-so400m-head": (("pooler_mean",), 1),
}


def make_test_set(n_anomaly: int = 3, n_normal: int = 3, windows: int = 4) -> tuple[dict[str, VideoItem], list[dict]]:
    """Videos at 10 fps with 2 s windows; anomaly videos have an event over frames 20-39 (window 1)."""
    items: dict[str, VideoItem] = {}
    meta: list[dict] = []
    for v in range(n_anomaly + n_normal):
        anomaly = v < n_anomaly
        name = f"v{v}.mp4"
        category = "Abuse"
        events: tuple[tuple[int, int], ...] = ((20, 39),)
        if not anomaly:
            category = "Normal"
            events = ()
        items[name] = VideoItem(name=name, path=Path(name), category=category, fps=10.0,
                                n_frames=windows * 20, events=events)
        for w in range(windows):
            meta.append({"i": len(meta), "video": name, "window": w, "start_s": 2.0 * w, "end_s": 2.0 * w + 2.0,
                         "label": int(anomaly and w == 1), "p_yes": None})
    return items, meta


def write_fits(tmp_path: Path, meta: list[dict], metas: dict[str, list[dict]] | None = None) -> Path:
    """Synthetic encoder fit folders under tmp_path/run; metas overrides test_meta.jsonl rows per fit."""
    run = tmp_path / "run"
    labels = np.array([row["label"] for row in meta], dtype=float)
    for seed, (fit_name, (kinds, n_states)) in enumerate(FITS.items()):
        rng = np.random.default_rng(seed)
        scores = np.empty((len(METHODS), len(kinds), n_states, len(meta)), dtype=np.float32)
        choices: dict[str, dict] = {}
        for method_i, method in enumerate(METHODS):
            for kind_i, kind in enumerate(kinds):
                for state in range(n_states):
                    signal = 0.5 + 0.4 * ((state + seed + kind_i) % n_states)
                    scores[method_i, kind_i, state] = (rng.normal(size=len(meta)) + signal * labels).astype(np.float32)
                    val = float(((state + seed + kind_i + method_i) % n_states) + 1) / 10.0
                    choices[f"{method}/{kind}/{state}"] = {"setting": {}, "validation": val}
        folder = run / f"fit-{fit_name}"
        folder.mkdir(parents=True)
        np.save(folder / "test_scores.npy", scores)
        rows = (metas or {}).get(fit_name, meta)
        (folder / "test_meta.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        fit_data = {"methods": list(METHODS), "kinds": list(kinds), "n_states": n_states,
                    "features": "x", "val_videos": [], "choices": choices}
        (folder / "fit.json").write_text(json.dumps(fit_data))
    return run


def test_select_state_direction_and_tie_breaking():
    choices = {
        "mil/patch_mean/0": {"setting": {}, "validation": 0.45},
        "mil/patch_mean/1": {"setting": {}, "validation": 0.25},
        "mil/patch_mean/2": {"setting": {}, "validation": 0.35},
        "mil/patch_mean/3": {"setting": {}, "validation": 0.50},
        "holmes/patch_mean/0": {"setting": {}, "validation": 0.60},
        "holmes/patch_mean/1": {"setting": {}, "validation": 0.85},
        "holmes/patch_mean/2": {"setting": {}, "validation": 0.92},
        "holmes/patch_mean/3": {"setting": {}, "validation": 0.80},
        "knn/patch_mean/0": {"setting": {}, "validation": 0.70},
        "knn/patch_mean/1": {"setting": {}, "validation": 0.65},
        "knn/patch_mean/2": {"setting": {}, "validation": 0.80},
        "knn/patch_mean/3": {"setting": {}, "validation": 0.95},
        "mil/patch_max/0": {"setting": {}, "validation": 0.50},
        "mil/patch_max/1": {"setting": {}, "validation": 0.20},
        "mil/patch_max/2": {"setting": {}, "validation": 0.20},
        "mil/patch_max/3": {"setting": {}, "validation": 0.40},
        "holmes/patch_max/0": {"setting": {}, "validation": 0.70},
        "holmes/patch_max/1": {"setting": {}, "validation": 0.90},
        "holmes/patch_max/2": {"setting": {}, "validation": 0.85},
        "holmes/patch_max/3": {"setting": {}, "validation": 0.90},
        "knn/patch_max/0": {"setting": {}, "validation": 0.88},
        "knn/patch_max/1": {"setting": {}, "validation": 0.75},
        "knn/patch_max/2": {"setting": {}, "validation": 0.88},
        "knn/patch_max/3": {"setting": {}, "validation": 0.60},
    }
    fit = {"methods": list(METHODS), "kinds": ["patch_mean", "patch_max"], "n_states": 4,
           "features": "x", "val_videos": [], "choices": choices}
    assert select_state(fit, "mil", "patch_mean") == 1
    assert select_state(fit, "holmes", "patch_mean") == 2
    assert select_state(fit, "knn", "patch_mean") == 3
    assert select_state(fit, "mil", "patch_max") == 1
    assert select_state(fit, "holmes", "patch_max") == 1
    assert select_state(fit, "knn", "patch_max") == 0


def test_run_encoder_compare_reports_selected_last_and_peak(tmp_path):
    items, meta = make_test_set()
    run_dir = write_fits(tmp_path, meta)
    out = run_encoder_compare(Path("unused.yaml"), run_dir, items=items, log=lambda line: None)
    assert out == run_dir / "encoder_compare.json"
    payload = json.loads(out.read_text())
    assert set(payload["fits"]) == set(FITS)
    assert payload["methods"] == list(METHODS)
    assert payload["n_boot"] == N_BOOT and payload["seed"] == SEED
    assert payload["baseline"] == {"fit": BASELINE[0], "kind": BASELINE[1]}
    results = payload["results"]
    for fit_name, (kinds, n_states) in FITS.items():
        fit_json = json.loads((run_dir / f"fit-{fit_name}" / "fit.json").read_text())
        for kind in kinds:
            for method in METHODS:
                entry = results[f"{fit_name}/{kind}/{method}"]
                assert entry["n_states"] == n_states
                curve = entry["curve"]
                assert len(curve["value"]) == n_states
                assert len(curve["low"]) == n_states
                assert len(curve["high"]) == n_states
                for key in ("selected", "last", "peak"):
                    point = entry[key]
                    assert set(point) == {"state", "value", "low", "high"}
                    state = point["state"]
                    assert 0 <= state < n_states
                    assert point["value"] == pytest.approx(curve["value"][state])
                    assert point["low"] == pytest.approx(curve["low"][state])
                    assert point["high"] == pytest.approx(curve["high"][state])
                    assert point["low"] <= point["value"] <= point["high"]
                assert entry["selected"]["state"] == select_state(fit_json, method, kind)
                assert entry["last"]["state"] == n_states - 1
                assert entry["peak"]["state"] == int(np.argmax(curve["value"]))
    for method in METHODS:
        assert (run_dir / f"{method}.png").stat().st_size > 0


def test_difference_against_baseline_matches_selected_states(tmp_path):
    items, meta = make_test_set(n_anomaly=5, n_normal=5)
    run_dir = write_fits(tmp_path, meta)
    shutil.copytree(run_dir / "fit-qwen35-vit", run_dir / "fit-copy-vit")
    out = run_encoder_compare(Path("unused.yaml"), run_dir, items=items, log=lambda line: None)
    results = json.loads(out.read_text())["results"]
    for method in METHODS:
        base_key = f"qwen35-vit/patch_mean/{method}"
        copy_key = f"copy-vit/patch_mean/{method}"
        copy_diff = results[copy_key]["difference"]
        assert copy_diff["state"] == results[copy_key]["selected"]["state"]
        assert copy_diff["baseline_state"] == results[base_key]["selected"]["state"]
        assert (copy_diff["value"], copy_diff["low"], copy_diff["high"]) == (0.0, 0.0, 0.0)
        other_key = f"dinov3-vitl/patch_mean/{method}"
        diff = results[other_key]["difference"]
        assert diff["state"] == results[other_key]["selected"]["state"]
        assert diff["baseline_state"] == results[base_key]["selected"]["state"]
        expected = results[other_key]["selected"]["value"] - results[base_key]["selected"]["value"]
        assert diff["value"] == pytest.approx(expected)
        assert diff["low"] <= diff["value"] <= diff["high"]


def test_head_folder_with_one_state_reports_state_zero(tmp_path):
    items, meta = make_test_set()
    run_dir = write_fits(tmp_path, meta)
    out = run_encoder_compare(Path("unused.yaml"), run_dir, items=items, log=lambda line: None)
    results = json.loads(out.read_text())["results"]
    for method in METHODS:
        entry = results[f"siglip2-so400m-head/pooler_mean/{method}"]
        assert entry["n_states"] == 1
        for key in ("selected", "last", "peak"):
            assert entry[key]["state"] == 0
            assert entry[key]["value"] == pytest.approx(entry["curve"]["value"][0])


def test_missing_baseline_fit_raises_value_error(tmp_path):
    items, meta = make_test_set()
    run_dir = write_fits(tmp_path, meta)
    shutil.rmtree(run_dir / "fit-qwen35-vit")
    with pytest.raises(ValueError):
        run_encoder_compare(Path("unused.yaml"), run_dir, items=items, log=lambda line: None)


def test_mismatched_test_meta_raises_value_error(tmp_path):
    items, meta = make_test_set()
    moved = []
    for row in meta:
        updated = dict(row)
        if row["i"] == 5:
            updated["video"] = "v0.mp4"
        moved.append(updated)
    run_dir = write_fits(tmp_path, meta, {"dinov3-vitl": moved})
    with pytest.raises(ValueError):
        run_encoder_compare(Path("unused.yaml"), run_dir, items=items, log=lambda line: None)


def test_existing_encoder_compare_json_raises_file_exists_error(tmp_path):
    items, meta = make_test_set()
    run_dir = write_fits(tmp_path, meta)
    (run_dir / "encoder_compare.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run_encoder_compare(Path("unused.yaml"), run_dir, items=items, log=lambda line: None)


def test_cli_encoder_compare(tmp_path, monkeypatch, capsys):
    calls = []

    def fake(config_path, run_dir):
        calls.append((config_path, run_dir))
        return tmp_path / "x.json"

    monkeypatch.setattr("s1stream.encoder.compare.run_encoder_compare", fake)
    code = cli.main(["encoder-compare", "c.yaml", str(tmp_path)])
    assert code == 0
    assert calls == [(Path("c.yaml"), tmp_path)]
    assert capsys.readouterr().out.strip() == str(tmp_path / "x.json")
