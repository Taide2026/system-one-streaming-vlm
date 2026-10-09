"""Unit runner: run and report with fake plugins. No model, no video."""
import json
from pathlib import Path

import pytest
import yaml

from s1stream import registry
from s1stream.config import load_config
from s1stream.registry import ANSWER_HEADS, BACKBONES, DATASETS, INPUT_FORMS
from s1stream.report import report
from s1stream.runner import run
from s1stream.types import BackboneOutput, ModelInput, VideoItem


@DATASETS.register("fake_dataset")
class FakeDataset:
    def items(self):
        # 6 s anomaly video with an event at 2-4 s (frames 60-119), and a 4 s normal video.
        return [VideoItem("a.mp4", Path("a.mp4"), "Abuse", 30.0, 180, ((60, 119),)),
                VideoItem("n.mp4", Path("n.mp4"), "Normal", 30.0, 120, ())]


@INPUT_FORMS.register("fake_input")
class FakeInput:
    def build(self, item, window, question):
        return ModelInput(text=str(window.label))  # passes the label through, so scores separate perfectly


@BACKBONES.register("fake_backbone")
class FakeBackbone:
    def forward(self, inp, question):
        return BackboneOutput(hidden=int(inp.text))


@ANSWER_HEADS.register("fake_head")
class FakeHead:
    def probabilities(self, out, question):
        yes = 0.9 if out.hidden else 0.1
        return {"Yes": yes, "No": 1 - yes}


CONFIG = {
    "name": "fake",
    "dataset": {"name": "fake_dataset"},
    "window": {"length_s": 2.0, "stride_s": 2.0, "frames": 8},
    "question": {"state": "s", "text": "q", "options": ["Yes", "No"], "positive": "Yes"},
    "input_form": {"name": "fake_input"},
    "backbone": {"name": "fake_backbone"},
    "answer_head": {"name": "fake_head"},
    "decision_rules": [{"name": "threshold", "start": 0.0, "stop": 1.0, "step": 0.5}],
}


@pytest.fixture
def config(tmp_path):
    import s1stream.rules.threshold  # noqa: F401  registers "threshold"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    return load_config(path)


def lines(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "scores.jsonl").read_text().splitlines()]


def test_run_writes_one_line_per_window(tmp_path, config):
    out = run(config, tmp_path / "run")
    rows = lines(out)
    assert len(rows) == 5  # 3 windows + 2 windows
    assert [(r["video"], r["window"], r["start_s"], r["end_s"], r["label"]) for r in rows[:3]] == [
        ("a.mp4", 0, 0.0, 2.0, 0), ("a.mp4", 1, 2.0, 4.0, 1), ("a.mp4", 2, 4.0, 6.0, 0)]
    assert rows[1]["score"] == pytest.approx(0.9)
    assert all(r["seconds"] >= 0 for r in rows)


def test_run_writes_config_manifest_and_run_data(tmp_path, config):
    out = run(config, tmp_path / "run")
    assert yaml.safe_load((out / "config.yaml").read_text()) == CONFIG
    manifest = json.loads((out / "manifest.json").read_text())
    assert [m["name"] for m in manifest] == ["a.mp4", "n.mp4"]
    assert manifest[0]["events"] == [[60, 119]]
    data = json.loads((out / "run.json").read_text())
    for key in ("free_memory_gb", "peak_gpu_gb", "load_s", "versions", "git_commit", "git_dirty", "assets"):
        assert key in data
    assert "torch" in data["versions"]


def test_run_refuses_existing_folder(tmp_path, config):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        run(config, tmp_path / "run")


def test_run_limit_windows(tmp_path, config):
    out = run(config, tmp_path / "run", limit_windows=3)
    assert len(lines(out)) == 3


def test_report_computes_metrics_without_a_model(tmp_path, config, monkeypatch):
    out = run(config, tmp_path / "run")

    def no_model(spec):
        raise AssertionError("report must not build a backbone or a head")

    monkeypatch.setattr(registry.BACKBONES, "build", no_model)
    monkeypatch.setattr(registry.ANSWER_HEADS, "build", no_model)
    result = report(out)
    assert json.loads((out / "report.json").read_text()) == result
    assert result["windows"] == 5
    assert result["window_auroc"] == pytest.approx(1.0)
    assert result["frame_auc"] == pytest.approx(1.0)
    for key in ("ece", "brier", "seconds_per_window", "real_time_factor", "peak_gpu_gb", "load_s"):
        assert key in result
    rows = result["rules"]["threshold"]
    assert [r["setting"] for r in rows] == [0.0, 0.5, 1.0]
    assert rows[1]["detection_rate"] == pytest.approx(1.0)
    assert rows[1]["mean_delay_s"] == pytest.approx(2.0)  # event starts at 2 s, first alarm window ends at 4 s
    assert rows[1]["false_alarms_per_hour"] == pytest.approx(0.0)
