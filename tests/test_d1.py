"""Unit d1: the d1 backbone with a fake System One model on the CPU. Never loads d1."""
import pytest
from PIL import Image

from s1stream.backbones.d1 import D1Backbone
from s1stream.backbones.d3 import build_questions
from s1stream.heads.option_logits import OptionLogits
from s1stream.registry import BACKBONES
from s1stream.types import ModelInput, Question

QUESTION = Question(state="Camera, 2 seconds.", text="Which?", options=("None", "A", "B"), positive=("A", "B"))


class FakeD1:
    def __init__(self):
        self.kwargs = None

    def system_one(self, state, questions, images=None):
        self.kwargs = {"state": state, "questions": questions, "images": images}
        return {"answers": {"answer": {"type": "choice", "choice": "A", "confidence": 0.6,
                                       "probabilities": {"A": 0.6, "B": 0.1, "None": 0.3}}},
                "usage": {"input_tokens": 1, "output_tokens": 0}}


def test_d1_is_registered():
    import s1stream.plugins  # noqa: F401

    assert BACKBONES._classes["d1"] is D1Backbone


def test_forward_sends_the_d3_request_and_reads_option_probabilities():
    fake = FakeD1()
    imgs = [Image.new("RGB", (32, 24)) for _ in range(8)]
    out = D1Backbone.from_parts(fake).forward(ModelInput(images=imgs), QUESTION)
    assert fake.kwargs == {"state": "Camera, 2 seconds.", "questions": build_questions(QUESTION), "images": imgs}
    probs = OptionLogits().probabilities(out, QUESTION)
    assert QUESTION.score(probs) == pytest.approx(0.7)
