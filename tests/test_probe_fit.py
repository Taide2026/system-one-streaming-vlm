"""Unit probe-fit: MIL, Holmes head and kNN per state (finetune-layer-probing decision 10). CPU, synthetic data."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from s1stream.probe.fit import KINDS, METHODS, bags, fit_condition, fit_mil, topk_pool

STATES, HIDDEN = 2, 6


def write_set(folder: Path, rows: list[dict], features: np.ndarray) -> None:
    folder.mkdir(parents=True)
    np.save(folder / "last.npy", features.astype(np.float16))
    np.save(folder / "visual.npy", (features * 0.5).astype(np.float16))
    (folder / "meta.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_features(root: Path, seed: int = 0, test_shift: float = 0.0) -> Path:
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
                if test:
                    vector += test_shift
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
        write_set(root / name, *build(videos, is_test))
    return root


def test_topk_pool_caps_k():
    index, mask, order = bags(["a", "a", "a", "b"])
    scores = torch.tensor([1.0, 3.0, 2.0, 5.0])
    assert order == ["a", "b"]
    assert torch.allclose(topk_pool(scores, index, mask, "3"), torch.tensor([2.0, 5.0]))
    assert torch.allclose(topk_pool(scores, index, mask, "1"), torch.tensor([3.0, 5.0]))


def test_fit_mil_finds_positive_instance():
    rng = np.random.default_rng(1)
    videos, x, instance = [], [], []
    for v in range(40):
        positive = v < 20
        for w in range(6):
            vector = rng.normal(size=8)
            hit = positive and w == 2
            vector[3] += 4.0 * hit
            videos.append(f"v{v}")
            x.append(vector)
            instance.append(int(hit))
    index, mask, order = bags(videos)
    video_y = torch.tensor([1.0 if int(name[1:]) < 20 else 0.0 for name in order])
    head = fit_mil(torch.tensor(np.stack(x), dtype=torch.float32), video_y, index, mask, "1", 1e-3)
    with torch.no_grad():
        scores = head(torch.tensor(np.stack(x), dtype=torch.float32)).squeeze(1).numpy()
    assert roc_auc_score(instance, scores) > 0.95


def test_choices_ignore_test_features(tmp_path):
    fit_condition(make_features(tmp_path / "f1"), tmp_path / "o1", log=lambda line: None)
    fit_condition(make_features(tmp_path / "f2", test_shift=5.0), tmp_path / "o2", log=lambda line: None)
    first = json.loads((tmp_path / "o1" / "fit.json").read_text())["choices"]
    second = json.loads((tmp_path / "o2" / "fit.json").read_text())["choices"]
    assert first == second


def test_knn_uses_only_train_normal(tmp_path):
    features = make_features(tmp_path / "f")
    fit_condition(features, tmp_path / "o1", log=lambda line: None)
    for kind in KINDS:
        array = np.load(features / "train_anomaly" / f"{kind}.npy")
        array[0] += 100
        np.save(features / "train_anomaly" / f"{kind}.npy", array)
    fit_condition(features, tmp_path / "o2", log=lambda line: None)
    knn = METHODS.index("knn")
    first, second = np.load(tmp_path / "o1" / "test_scores.npy"), np.load(tmp_path / "o2" / "test_scores.npy")
    assert np.array_equal(first[knn], second[knn])


def test_outputs_shape_and_fields(tmp_path):
    fit_condition(make_features(tmp_path / "f"), tmp_path / "o", log=lambda line: None)
    scores = np.load(tmp_path / "o" / "test_scores.npy")
    assert scores.shape == (len(METHODS), len(KINDS), STATES, 32) and scores.dtype == np.float32
    fit = json.loads((tmp_path / "o" / "fit.json").read_text())
    assert fit["methods"] == list(METHODS) and fit["kinds"] == list(KINDS) and fit["n_states"] == STATES
    for method in METHODS:
        for kind in KINDS:
            for state in range(STATES):
                choice = fit["choices"][f"{method}/{kind}/{state}"]
                assert "setting" in choice and isinstance(choice["validation"], float)
    assert fit["choices"]["mil/last/0"]["setting"]["k"] in ("1", "3", "frac8")
    assert len((tmp_path / "o" / "test_meta.jsonl").read_text().splitlines()) == 32
