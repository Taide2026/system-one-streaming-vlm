"""Unit d3: the d3 backbone with a fake System One model on the CPU. Never loads d3."""
import pytest
import torch
from PIL import Image

from s1stream.backbones.d3 import D3Backbone, build_questions, load_release_code, runtime_device
from s1stream.heads.option_logits import OptionLogits
from s1stream.types import ModelInput, Question

QUESTION = Question(state="Camera, 2 seconds.", text="Abnormal?", options=("Yes", "No"), positive="Yes")


def images(n: int = 8) -> list:
    return [Image.new("RGB", (32, 24), (i * 20, 0, 0)) for i in range(n)]


class FakeD3:
    def __init__(self, answer):
        self.answer = answer
        self.kwargs = None

    def system_one(self, **kwargs):
        self.kwargs = kwargs
        return {"model": "fake", "answers": {"answer": self.answer}, "usage": {"input_tokens": 1, "output_tokens": 0}}


def test_questions_are_one_choice_with_bare_options():
    assert build_questions(QUESTION) == {
        "answer": {"type": "choice", "instructions": "Abnormal?", "criteria": {"Yes": None, "No": None}}}


def test_forward_sends_state_frames_and_question_and_keeps_option_order():
    # The runtime may list probabilities in its own order; forward follows question.options.
    fake = FakeD3({"type": "choice", "choice": "No", "probabilities": {"No": 0.8, "Yes": 0.2}, "confidence": 0.8})
    imgs = images(16)
    out = D3Backbone.from_parts(fake).forward(ModelInput(images=imgs), QUESTION)
    assert fake.kwargs["state"] == "Camera, 2 seconds."
    assert fake.kwargs["images"] == imgs
    assert fake.kwargs["questions"] == build_questions(QUESTION)
    probs = OptionLogits().probabilities(out, QUESTION)
    assert probs["Yes"] == pytest.approx(0.2)
    assert probs["No"] == pytest.approx(0.8)


def test_forward_survives_a_zero_probability():
    fake = FakeD3({"type": "choice", "choice": "No", "probabilities": {"Yes": 0.0, "No": 1.0}, "confidence": 1.0})
    out = D3Backbone.from_parts(fake).forward(ModelInput(images=images(1)), QUESTION)
    assert torch.isfinite(out.logits).all()
    assert OptionLogits().probabilities(out, QUESTION)["Yes"] == pytest.approx(0.0, abs=1e-12)


def test_forward_raises_on_a_runtime_error():
    fake = FakeD3({"type": "choice", "error": "max_length_exceeded", "message": "too long"})
    with pytest.raises(RuntimeError, match="max_length_exceeded"):
        D3Backbone.from_parts(fake).forward(ModelInput(images=images(1)), QUESTION)


def test_missing_release_code_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="d3_runtime.py"):
        load_release_code(tmp_path)


def test_release_code_imports_its_sibling_modules(tmp_path):
    (tmp_path / "d3_format.py").write_text("VALUE = 7\n")
    (tmp_path / "d3_runtime.py").write_text("from d3_format import VALUE\nclass D3:\n    pass\n")
    assert load_release_code(tmp_path).VALUE == 7


def test_runtime_device_has_an_index():
    assert runtime_device("cuda") == "cuda:0"
    assert runtime_device("cuda:1") == "cuda:1"
    assert runtime_device("cpu") == "cpu"
