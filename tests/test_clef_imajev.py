"""Unit clef/imajev: clef_flash and imajev backbones and the option_logits head, with tiny fakes on the CPU.
Never loads Clef-Flash, Qwen3.5 or the imajev adapter."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from s1stream.heads.option_logits import OptionLogits
from s1stream.types import BackboneOutput, ModelInput, Question

QUESTION = Question(state="Camera, 2 seconds.", text="Abnormal?", options=("Yes", "No"), positive="Yes")


def images(n: int = 8) -> list:
    return [Image.new("RGB", (32, 24), (i * 20, 0, 0)) for i in range(n)]


# ---------- option_logits ----------

def test_option_logits_softmax_in_option_order():
    probs = OptionLogits().probabilities(BackboneOutput(logits=torch.tensor([2.0, 0.0])), QUESTION)
    assert list(probs) == ["Yes", "No"]
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["Yes"] == pytest.approx(torch.tensor([2.0, 0.0]).softmax(-1)[0].item())


def test_option_logits_rejects_missing_or_wrong_count():
    with pytest.raises(ValueError):
        OptionLogits().probabilities(BackboneOutput(), QUESTION)
    with pytest.raises(ValueError):
        OptionLogits().probabilities(BackboneOutput(logits=torch.zeros(3)), QUESTION)
