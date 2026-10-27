"""Unit probe-extract: probe config, window plan, resumable extraction, run.json (finetune-layer-probing decisions
3, 5, 6, 8). CPU only; a fake run_window replaces the model."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from s1stream.config import ConfigError, WindowSpec
from s1stream.probe.extract import (extract, limit_sets, load_probe_config, plan_sets, run_extract,
                                    write_run_json)
from s1stream.probe.model import WindowResult
from s1stream.types import VideoItem

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "probe_ucf_short_v1_f8.yaml"
ASSETS = ("ucf-crime", "holmes-vau-ucf-train", "d3-flash", "qwen3.5-9b")
WINDOW = WindowSpec(length_s=2.0, stride_s=2.0, frames=8)
STATES, HIDDEN = 3, 4


def resolver(name: str) -> Path:
    return Path("/assets") / name


def video(name: str, category: str, seconds: float, events=()) -> VideoItem:
    return VideoItem(name=name, path=Path(name), category=category, fps=10.0, n_frames=int(seconds * 10),
                     events=tuple(events))


def rows(n_videos: int = 3, seconds: float = 6.0):
    """test rows: n_videos anomaly test videos of 3 windows each."""
    items = [video(f"v{i}.mp4", "Abuse", seconds, [(25, 44)]) for i in range(n_videos)]
    return plan_sets([], items, WINDOW)["test"]


class Runner:
    """Fake run_window: deterministic features from the row index; counts calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, row) -> WindowResult:
        self.calls.append((row.item.name, row.window.index))
        value = float(len(self.calls))
        return WindowResult(last=torch.full((STATES, HIDDEN), value), visual=torch.full((STATES, HIDDEN), -value),
                            p_yes=0.5)


def test_load_probe_config_resolves_assets_and_checks_attention(tmp_path):
    config = load_probe_config(CONFIG, resolver)
    assert config.name == "probe_ucf_short_v1_f8"
    assert config.train["path"] == Path("/assets/ucf-crime") and config.test["path"] == Path("/assets/ucf-crime")
    assert config.runtime_path == Path("/assets/d3-flash")
    assert config.conditions["A"].weights_path == Path("/assets/qwen3.5-9b")
    assert config.conditions["A"].attention == "causal" and config.conditions["B"].attention == "noncausal"
    assert config.conditions["C"].weights is None and config.conditions["C"].weights_path is None
    assert config.question.positive == "Yes" and config.window == WINDOW
    data = yaml.safe_load(CONFIG.read_text())
    data["conditions"]["B"]["attention"] = "bidirectional"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError) as err:
        load_probe_config(bad, resolver)
    assert "B" in str(err.value)


def test_plan_sets_test_labels():
    items = [video("a.mp4", "Abuse", 6.0, [(25, 44)]), video("n.mp4", "Normal", 4.0)]
    test = plan_sets([], items, WINDOW)["test"]
    assert [(r.item.name, r.window.index, r.label, r.holmes) for r in test] == [
        ("a.mp4", 0, 0, None), ("a.mp4", 1, 1, None), ("a.mp4", 2, 1, None), ("n.mp4", 0, 0, None),
        ("n.mp4", 1, 0, None)]


def test_plan_sets_train_holmes_labels():
    items = [video("a.mp4", "Abuse", 6.0, [(25, 44)]), video("n.mp4", "Normal", 4.0, [(0, 39)])]
    sets = plan_sets(items, [], WINDOW)
    assert [(r.label, r.holmes) for r in sets["train_anomaly"]] == [(1, 0), (1, 1), (1, 1)]
    assert [(r.label, r.holmes) for r in sets["train_normal"]] == [(0, 0), (0, 0)]
    assert sets["test"] == []


def test_extract_writes_arrays_and_meta(tmp_path):
    plan = rows()
    extract(plan, Runner(), tmp_path / "test", part_size=4, log=lambda line: None)
    last = np.load(tmp_path / "test" / "last.npy")
    visual = np.load(tmp_path / "test" / "visual.npy")
    assert last.shape == visual.shape == (9, STATES, HIDDEN) and last.dtype == np.float16
    assert last[8, 0, 0] == 9.0 and visual[8, 0, 0] == -9.0
    meta = [json.loads(line) for line in (tmp_path / "test" / "meta.jsonl").read_text().splitlines()]
    assert len(meta) == 9
    assert {"i", "video", "category", "window", "start_s", "end_s", "label", "holmes", "p_yes", "seconds"} <= set(meta[0])
    assert (meta[4]["i"], meta[4]["video"], meta[4]["window"]) == (4, "v1.mp4", 1)
    assert not (tmp_path / "test" / "parts").exists()


def test_extract_resumes_unfinished_parts(tmp_path):
    plan = rows()
    out = tmp_path / "test"

    class Crash(Runner):
        def __call__(self, row):
            if len(self.calls) == 6:
                raise KeyboardInterrupt
            return super().__call__(row)

    with pytest.raises(KeyboardInterrupt):
        extract(plan, Crash(), out, part_size=4, log=lambda line: None)
    runner = Runner()
    extract(plan, runner, out, part_size=4, log=lambda line: None)
    assert runner.calls == [(r.item.name, r.window.index) for r in plan[4:]]
    assert np.load(out / "last.npy").shape[0] == 9


def test_extract_rejects_changed_plan(tmp_path):
    plan = rows()
    out = tmp_path / "test"

    class Crash(Runner):
        def __call__(self, row):
            if len(self.calls) == 5:
                raise KeyboardInterrupt
            return super().__call__(row)

    with pytest.raises(KeyboardInterrupt):
        extract(plan, Crash(), out, part_size=4, log=lambda line: None)
    with pytest.raises(RuntimeError):
        extract(rows(seconds=8.0), Runner(), out, part_size=4, log=lambda line: None)


def test_limit_sets_keeps_first_rows():
    sets = {"train_anomaly": rows(), "train_normal": rows(1), "test": rows(2)}
    limited = limit_sets(sets, 2)
    assert {name: len(value) for name, value in limited.items()} == {"train_anomaly": 2, "train_normal": 2, "test": 2}
    assert limited["test"] == sets["test"][:2]
    assert limit_sets(sets, None) == sets


def test_write_run_json_records_provenance(tmp_path):
    config = load_probe_config(CONFIG, resolver)
    write_run_json(tmp_path, config, "A", {"peak_gpu_gb": 1.5})
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["condition"] == "A" and run["peak_gpu_gb"] == 1.5
    assert len(run["git_commit"]) == 40 and isinstance(run["git_dirty"], bool)
    assert set(ASSETS) <= set(run["assets"])
    assert run["assets"]["qwen3.5-9b"]["revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert "config.json" in run["assets"]["qwen3.5-9b"]["sha256"]


def test_run_extract_never_overwrites_finished_run(tmp_path, monkeypatch):
    for name in ASSETS:
        (tmp_path / "assets" / name).mkdir(parents=True)
    monkeypatch.setenv("S1_ASSETS", str(tmp_path / "assets"))
    monkeypatch.setenv("S1_FEATURES", str(tmp_path / "features"))
    out = tmp_path / "features" / "probe_ucf_short_v1_f8" / "C" / "r1"
    out.mkdir(parents=True)
    (out / "run.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run_extract(CONFIG, "C", "r1")
