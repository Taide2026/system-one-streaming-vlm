"""Unit probe-fit-device: device argument, convergence record and run-2 optimizer setting of probe-fit.
CPU, synthetic data."""
import json

import numpy as np
import torch

from s1stream.probe.fit import KINDS, METHODS, Fitted, bags, fit_condition, fit_mil, fit_window, knn_scores
from test_probe_fit import STATES, make_features


def separable(seed: int = 3, n_videos: int = 40, windows: int = 6, hidden: int = 16):
    """Videos of 6 windows; in the first half window 2 is abnormal and has feature 3 raised by 4. Features 0-7
    share one common factor, so the standardized inputs are strongly correlated."""
    rng = np.random.default_rng(seed)
    videos, x, y = [], [], []
    for v in range(n_videos):
        for w in range(windows):
            vector = rng.normal(size=hidden)
            vector[:8] += 3.0 * rng.normal()
            hit = v < n_videos // 2 and w == 2
            vector[3] += 4.0 * hit
            videos.append(f"v{v}")
            x.append(vector)
            y.append(float(hit))
    return videos, torch.tensor(np.stack(x), dtype=torch.float32), torch.tensor(y)


def test_fit_condition_cpu_device_keeps_outputs(tmp_path):
    fit_condition(make_features(tmp_path / "f"), tmp_path / "o", log=lambda line: None, device="cpu")
    scores = np.load(tmp_path / "o" / "test_scores.npy")
    assert scores.shape == (len(METHODS), len(KINDS), STATES, 32) and scores.dtype == np.float32
    assert np.isfinite(scores).all()
    fit = json.loads((tmp_path / "o" / "fit.json").read_text())
    assert fit["methods"] == list(METHODS) and fit["kinds"] == list(KINDS) and fit["n_states"] == STATES


def test_choices_record_losses(tmp_path):
    fit_condition(make_features(tmp_path / "f"), tmp_path / "o", log=lambda line: None, device="cpu")
    choices = json.loads((tmp_path / "o" / "fit.json").read_text())["choices"]
    for method in ("mil", "holmes"):
        for kind in KINDS:
            for state in range(STATES):
                choice = choices[f"{method}/{kind}/{state}"]
                assert isinstance(choice["final_loss"], float) and isinstance(choice["min_loss"], float)
                assert choice["min_loss"] <= choice["final_loss"]
    assert "final_loss" not in choices["knn/last/0"]


def test_heads_converge_without_oscillation():
    videos, x, y = separable()
    window = fit_window(x, y, 1e-3)
    assert isinstance(window, Fitted) and window.head.weight.device.type == "cpu"
    assert window.final_loss - window.min_loss <= 0.01 * window.min_loss
    index, mask, order = bags(videos)
    video_y = torch.tensor([1.0 if int(name[1:]) < 20 else 0.0 for name in order])
    mil = fit_mil(x, video_y, index, mask, "1", 1e-3)
    assert mil.final_loss - mil.min_loss <= 0.01 * mil.min_loss


def test_knn_scores_do_not_depend_on_heads(tmp_path):
    features = make_features(tmp_path / "f")
    fit_condition(features, tmp_path / "o", log=lambda line: None, device="cpu")
    scores = np.load(tmp_path / "o" / "test_scores.npy")
    for kind_i, kind in enumerate(KINDS):
        normal = np.load(features / "train_normal" / f"{kind}.npy").astype(np.float32)
        test = np.load(features / "test" / f"{kind}.npy").astype(np.float32)
        for state in range(STATES):
            expected = knn_scores(normal[:, state], test[:, state]).astype(np.float32)
            assert np.array_equal(scores[METHODS.index("knn"), kind_i, state], expected)
