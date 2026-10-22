"""Unit startlux: the startlux backbone with a fake Decider on the CPU. Never loads StartLux."""
from types import SimpleNamespace

import pytest
from PIL import Image

from s1stream.backbones.d3 import build_questions
from s1stream.backbones.startlux import StartLuxBackbone
from s1stream.heads.option_logits import OptionLogits
from s1stream.registry import BACKBONES
from s1stream.types import ModelInput, Question

QUESTION = Question(state="Camera, 2 seconds.", text="Which?", options=("A", "B", "None"), positive=("A", "B"))


class FakeDecider:
    def __init__(self, answer):
        self.answer, self.call = answer, None

    async def decide(self, state, questions, controls=None, images=None):
        self.call = (state, questions, images)
        return SimpleNamespace(answers={"answer": self.answer})


def test_startlux_is_registered():
    import s1stream.plugins  # noqa: F401

    assert BACKBONES._classes["startlux"] is StartLuxBackbone


def test_forward_sends_the_d3_request_with_the_frames_and_keeps_option_order():
    fake = FakeDecider({"type": "choice", "choice": "A", "probabilities": {"None": 0.3, "B": 0.1, "A": 0.6}})
    imgs = [Image.new("RGB", (32, 24)) for _ in range(8)]
    out = StartLuxBackbone.from_parts(fake).forward(ModelInput(images=imgs), QUESTION)
    assert fake.call == ("Camera, 2 seconds.", build_questions(QUESTION), imgs)
    probs = OptionLogits().probabilities(out, QUESTION)
    assert QUESTION.score(probs) == pytest.approx(0.7)


def test_forward_raises_on_an_error_answer():
    fake = FakeDecider({"error": "bad_request", "message": "too many images"})
    with pytest.raises(RuntimeError, match="too many images"):
        StartLuxBackbone.from_parts(fake).forward(ModelInput(images=[]), QUESTION)
