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


@DATASETS.register("fake_rooted_dataset")
class FakeRootedDataset:
    def __init__(self, path):
        self.root = Path(path)

    def items(self):
        return [VideoItem("a.mp4", self.root / "videos" / "a.mp4", "Abuse", 30.0, 60, ())]


def test_manifest_paths_are_relative_to_the_dataset_folder(tmp_path):
    import s1stream.rules.threshold  # noqa: F401
    data = dict(CONFIG, dataset={"name": "fake_rooted_dataset", "path": str(tmp_path / "assets" / "ucf")})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    out = run(load_config(path), tmp_path / "run")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest[0]["path"] == "videos/a.mp4"


REPORT_KEYS = {"windows", "videos", "frames_per_window", "peak_gpu_gb", "load_s", "latency", "classification",
               "decision", "auc", "ap"}


def edit_config(run_dir: Path, **changes) -> None:
    data = yaml.safe_load((run_dir / "config.yaml").read_text())
    for key, value in changes.items():
        if key == "decision_rules":
            data[key] = value
        else:
            data["window"][key] = value
    (run_dir / "config.yaml").write_text(yaml.safe_dump(data))


def test_report_has_only_the_new_keys(tmp_path, config):
    result = report(run(config, tmp_path / "run"))
    assert set(result) == REPORT_KEYS
    assert result["windows"] == 5
    assert result["videos"] == 2
    assert result["frames_per_window"] == 8
    assert set(result["latency"]) == {"first_window_ms", "ms_per_frame", "p50_ms", "p95_ms", "p99_ms",
                                      "rtf_p95", "budget_ms"}


def test_report_budget_and_frames_come_from_the_config(tmp_path, config):
    out = run(config, tmp_path / "run")
    base = report(out)["latency"]
    edit_config(out, stride_s=1.0, frames=4)
    changed = report(out)["latency"]
    assert changed["budget_ms"] == pytest.approx(1000.0)
    assert changed["rtf_p95"] == pytest.approx(2 * base["rtf_p95"])
    assert changed["ms_per_frame"] == pytest.approx(2 * base["ms_per_frame"])


def test_report_classification_per_threshold_or_default(tmp_path, config):
    out = run(config, tmp_path / "run")
    rows = report(out)["classification"]
    assert [r["threshold"] for r in rows] == [0.0, 0.5, 1.0]
    assert rows[1]["accuracy"] == pytest.approx(1.0)
    assert (rows[1]["recall"], rows[1]["fnr"]) == pytest.approx((1.0, 0.0))
    edit_config(out, decision_rules=[])
    rows = report(out)["classification"]
    assert [r["threshold"] for r in rows] == [0.5]


def test_report_computes_metrics_without_a_model(tmp_path, config, monkeypatch):
    out = run(config, tmp_path / "run")

    def no_model(spec):
        raise AssertionError("report must not build a backbone or a head")

    monkeypatch.setattr(registry.BACKBONES, "build", no_model)
    monkeypatch.setattr(registry.ANSWER_HEADS, "build", no_model)
    result = report(out)
    assert json.loads((out / "report.json").read_text()) == result
    assert result["latency"]["p50_ms"] > 0


def test_report_decision_and_auc_on_perfectly_separated_scores(tmp_path, config):
    # The fake head scores abnormal windows 0.9 and normal ones 0.1, and the event (frames 60-119) is exactly window 1.
    result = report(run(config, tmp_path / "run"))
    assert result["decision"]["threshold"] == 0.5
    assert result["decision"]["balanced_accuracy"] == pytest.approx(1.0)
    assert result["auc"] == pytest.approx({"frame": 1.0, "window": 1.0})
    assert result["ap"] == pytest.approx({"frame": 1.0, "window": 1.0})


def test_run_writes_the_answer_text_of_a_decoding_backbone(tmp_path, config, monkeypatch):
    assert "text" not in lines(run(config, tmp_path / "plain"))[0]
    monkeypatch.setattr(FakeBackbone, "forward", lambda self, inp, question: BackboneOutput(hidden=int(inp.text),
                                                                                           text="Yes."))
    assert lines(run(config, tmp_path / "decoding"))[0]["text"] == "Yes."
