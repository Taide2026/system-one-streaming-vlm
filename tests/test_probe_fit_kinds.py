"""Unit probe-fit-kinds: token kinds from features/run.json in probe-fit. CPU, synthetic data."""
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from s1stream.probe.fit import KINDS, METHODS, SETS, fit_condition, read_kinds

STATES, HIDDEN = 2, 6


def write_set(folder: Path, rows: list[dict], features: np.ndarray, kinds: Sequence[str] = KINDS) -> None:
    folder.mkdir(parents=True)
    for i, kind in enumerate(kinds):
        np.save(folder / f"{kind}.npy", (features / (i + 1)).astype(np.float16))
    (folder / "meta.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_features(root: Path, kinds: Sequence[str] = KINDS, seed: int = 0) -> Path:
    """Two anomaly categories (6 videos each) and 12 normal videos, 4 windows per video. The first window of every
    anomaly video is abnormal (holmes 1) and has feature 0 raised by 3. Test: 4 anomaly + 4 normal videos."""
    rng = np.random.default_rng(seed)

    def build(videos: list[tuple[str, str]], test: bool) -> tuple[list[dict], np.ndarray]:
        rows, features = [], []
        for name, category in videos:
            for window in range(4):
                abnormal = int(category != "Normal" and window == 0)
                vector = rng.normal(size=(STATES, HIDDEN))
                vector[:, 0] += 3.0 * abnormal
                features.append(vector)
                label = abnormal if test else int(category != "Normal")
                holmes = None if test else abnormal
                rows.append({"i": len(rows), "video": name, "category": category, "window": window,
                             "start_s": 2.0 * window, "end_s": 2.0 * window + 2.0, "label": label,
                             "holmes": holmes, "p_yes": 0.5, "seconds": 0.1})
        return rows, np.stack(features)

    anomaly = [(f"{c}{i:02d}.mp4", c) for c in ("Abuse", "Arson") for i in range(6)]
    normal = [(f"Normal{i:02d}.mp4", "Normal") for i in range(12)]
    test = [(f"TAbuse{i}.mp4", "Abuse") for i in range(4)] + [(f"TNormal{i}.mp4", "Normal") for i in range(4)]
    for name, videos, is_test in (("train_anomaly", anomaly, False), ("train_normal", normal, False),
                                  ("test", test, True)):
        write_set(root / name, *build(videos, is_test), kinds=kinds)
    return root


def test_run_json_kinds_list(tmp_path):
    kinds = ["patch_mean", "patch_max", "cls_mean"]
    features = make_features(tmp_path / "f", kinds=kinds)
    (features / "run.json").write_text(json.dumps({"kinds": kinds}))
    fit_condition(features, tmp_path / "o", log=lambda line: None)
    scores = np.load(tmp_path / "o" / "test_scores.npy")
    assert scores.shape == (3, 3, STATES, 32) and scores.dtype == np.float32
    fit = json.loads((tmp_path / "o" / "fit.json").read_text())
    assert fit["kinds"] == kinds
    assert all(f"mil/{kind}/0" in fit["choices"] for kind in kinds)
    assert read_kinds(features) == tuple(kinds)


def test_missing_run_json_or_kinds_key(tmp_path):
    f1 = make_features(tmp_path / "f1")
    fit_condition(f1, tmp_path / "o1", log=lambda line: None)
    fit1 = json.loads((tmp_path / "o1" / "fit.json").read_text())
    assert fit1["kinds"] == list(KINDS)
    assert np.load(tmp_path / "o1" / "test_scores.npy").shape == (len(METHODS), len(KINDS), STATES, 32)
    assert read_kinds(f1) == KINDS

    f2 = make_features(tmp_path / "f2")
    (f2 / "run.json").write_text(json.dumps({"name": "x"}))
    fit_condition(f2, tmp_path / "o2", log=lambda line: None)
    fit2 = json.loads((tmp_path / "o2" / "fit.json").read_text())
    assert fit2["kinds"] == list(KINDS)
    assert np.load(tmp_path / "o2" / "test_scores.npy").shape == (len(METHODS), len(KINDS), STATES, 32)
    assert read_kinds(f2) == KINDS


def test_dict_kinds_uses_default_kinds(tmp_path):
    features = make_features(tmp_path / "f")
    (features / "run.json").write_text(json.dumps({"kinds": {"visual": "token mean", "last": "token max"}}))
    fit_condition(features, tmp_path / "o", log=lambda line: None)
    fit = json.loads((tmp_path / "o" / "fit.json").read_text())
    assert fit["kinds"] == list(KINDS)
    assert np.load(tmp_path / "o" / "test_scores.npy").shape == (len(METHODS), len(KINDS), STATES, 32)
    assert read_kinds(features) == KINDS


def test_mismatched_state_count_raises(tmp_path):
    features = make_features(tmp_path / "f", kinds=("a",))
    for name in SETS:
        a = np.load(features / name / "a.npy")
        np.save(features / name / "b.npy", np.zeros((len(a), 3, HIDDEN), dtype=np.float16))
    (features / "run.json").write_text(json.dumps({"kinds": ["a", "b"]}))
    with pytest.raises(ValueError, match="b"):
        fit_condition(features, tmp_path / "o", log=lambda line: None)
