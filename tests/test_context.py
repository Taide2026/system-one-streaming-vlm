"""Unit context: the context section and the runner's strategies, with fake plugins. No model, no video."""
import json
from pathlib import Path

import pytest
import yaml

from s1stream.config import ConfigError, load_config
from s1stream.context import CachePolicy, ContextError, ContextSpec, parse_context, policy, text_state, with_past_frames
from s1stream.registry import ANSWER_HEADS, BACKBONES, DATASETS, INPUT_FORMS
from s1stream.runner import run
from s1stream.types import BackboneOutput, ModelInput, Question, VideoItem
from s1stream.windows import plan_windows

QUESTION = Question(state="Camera, 2 seconds.", text="q", options=("Yes", "No"), positive="Yes")


@DATASETS.register("ctx_dataset")
class CtxDataset:
    def items(self):
        # 8 s anomaly video, event at 3-5 s (frames 90-149): windows 1 and 2 are abnormal; a 4 s normal video.
        return [VideoItem("a.mp4", Path("a.mp4"), "Abuse", 30.0, 240, ((90, 149),)),
                VideoItem("n.mp4", Path("n.mp4"), "Normal", 30.0, 120, ())]


@INPUT_FORMS.register("ctx_input")
class CtxInput:
    def build(self, item, window, question):
        # One fake image per frame index, and the window's label as text.
        return ModelInput(images=list(window.frame_indices), text=f"{window.label}|{question.state}")


@BACKBONES.register("ctx_backbone")
class CtxBackbone:
    calls: list = []
    videos: int = 0

    def forward(self, inp, question):
        CtxBackbone.calls.append(("forward", len(inp.images), question.state))
        return BackboneOutput(logits=[2.0, 0.0] if inp.text.startswith("1") else [0.0, 2.0])

    def start_video(self):
        CtxBackbone.videos += 1

    def forward_stream(self, inp, question, policy, window):
        CtxBackbone.calls.append(("stream", len(inp.images), policy, window))
        return BackboneOutput(logits=[2.0, 0.0] if inp.text.startswith("1") else [0.0, 2.0])


@BACKBONES.register("ctx_stateless")
class CtxStateless:
    def forward(self, inp, question):
        return BackboneOutput(logits=[0.0, 0.0])


@ANSWER_HEADS.register("ctx_head")
class CtxHead:
    def probabilities(self, out, question):
        import torch
        values = torch.tensor(out.logits).softmax(-1).tolist()
        return dict(zip(question.options, values))


def config(tmp_path, context=None, backbone="ctx_backbone"):
    import s1stream.rules.threshold  # noqa: F401
    data = {
        "name": "ctx",
        "dataset": {"name": "ctx_dataset"},
        "window": {"length_s": 2.0, "stride_s": 2.0, "frames": 8},
        "question": {"state": QUESTION.state, "text": "q", "options": ["Yes", "No"], "positive": "Yes"},
        "input_form": {"name": "ctx_input"},
        "backbone": {"name": backbone},
        "answer_head": {"name": "ctx_head"},
        "decision_rules": [{"name": "threshold", "start": 0.0, "stop": 1.0, "step": 0.5}],
    }
    if context is not None:
        data["context"] = context
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path)


def rows(run_dir):
    return [json.loads(line) for line in (run_dir / "scores.jsonl").read_text().splitlines()]


@pytest.fixture(autouse=True)
def reset_backbone():
    CtxBackbone.calls = []
    CtxBackbone.videos = 0


# ---------- the section ----------

def test_no_section_is_s0():
    assert parse_context(None, 2.0) == ContextSpec()


def test_parse_kv_and_policy():
    spec = parse_context({"strategy": "kv_sinks", "past_s": 4.0, "sinks_s": 2.0}, 2.0)
    assert spec.keep_windows(2.0) == 2
    assert policy(spec, 2.0) == CachePolicy(keep_windows=2, sink_windows=1, budget_tokens=None)
    plain = parse_context({"strategy": "kv", "past_s": 4.0, "sinks_s": 2.0}, 2.0)
    assert policy(plain, 2.0).sink_windows == 0  # sinks only for kv_sinks


@pytest.mark.parametrize("data", [
    {"strategy": "sliding"},
    {"strategy": "kv", "past_s": 3.0},
    {"strategy": "recompute"},
    {"strategy": "kv_sinks", "past_s": 2.0},
    {"strategy": "kv_compress", "past_s": 2.0},
    {"strategy": "text"},
])
def test_parse_rejects_bad_sections(data):
    with pytest.raises(ContextError):
        parse_context(data, 2.0)


def test_load_config_turns_a_context_error_into_a_config_error(tmp_path):
    with pytest.raises(ConfigError):
        config(tmp_path, {"strategy": "kv", "past_s": 3.0})


def test_with_past_frames_prepends_the_past_windows():
    windows = plan_windows(VideoItem("v", Path("v"), "X", 30.0, 240), 2.0, 2.0, 8)
    seen = with_past_frames(windows, 2, 1)
    assert seen.index == 2 and seen.label == windows[2].label
    assert seen.frame_indices == windows[1].frame_indices + windows[2].frame_indices
    assert with_past_frames(windows, 0, 2).frame_indices == windows[0].frame_indices


def test_text_state_lists_the_last_scores_oldest_first():
    assert text_state(QUESTION, [], 4) == QUESTION
    q = text_state(QUESTION, [0.1, 0.26, 0.2], 2)
    assert q.state == "Camera, 2 seconds. Abnormal probability of the previous windows, oldest first: 0.26, 0.20."
    assert q.text == QUESTION.text and q.options == QUESTION.options


# ---------- the runner ----------

def test_s0_reads_the_newest_window_only(tmp_path):
    run(config(tmp_path), tmp_path / "run")
    assert all(call[1] == 8 for call in CtxBackbone.calls)
    assert CtxBackbone.videos == 0


def test_recompute_reads_the_past_windows_too_and_scores_the_newest(tmp_path):
    out = run(config(tmp_path, {"strategy": "recompute", "past_s": 4.0}), tmp_path / "run")
    frames = [call[1] for call in CtxBackbone.calls]
    assert frames == [8, 16, 24, 24, 8, 16]  # video a: 4 windows, video n: 2 windows
    lines = rows(out)
    assert [row["frames_read"] for row in lines] == frames
    assert [row["label"] for row in lines[:4]] == [0, 1, 1, 0]
    assert lines[1]["score"] > 0.5  # the label of the newest window decides


def test_kv_streams_per_video(tmp_path):
    run(config(tmp_path, {"strategy": "kv", "past_s": 2.0}), tmp_path / "run")
    assert CtxBackbone.videos == 2
    assert [(call[0], call[1], call[3]) for call in CtxBackbone.calls] == \
        [("stream", 8, 0), ("stream", 8, 1), ("stream", 8, 2), ("stream", 8, 3), ("stream", 8, 0), ("stream", 8, 1)]
    assert CtxBackbone.calls[0][2] == CachePolicy(keep_windows=1)


def test_kv_needs_a_streaming_backbone(tmp_path):
    with pytest.raises(ConfigError):
        run(config(tmp_path, {"strategy": "kv", "past_s": 2.0}, backbone="ctx_stateless"), tmp_path / "run")


def test_text_puts_the_past_scores_into_the_state(tmp_path):
    out = run(config(tmp_path, {"strategy": "text", "history": 2}), tmp_path / "run")
    states = [call[2] for call in CtxBackbone.calls]
    assert states[0] == QUESTION.state  # no past yet
    scores = [row["score"] for row in rows(out)]
    assert states[1].endswith(f"oldest first: {scores[0]:.2f}.")
    assert states[3].endswith(f"oldest first: {scores[1]:.2f}, {scores[2]:.2f}.")
    assert states[4] == QUESTION.state  # a new video starts without a past
