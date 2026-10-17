"""Unit clef/imajev: clef_flash and imajev backbones and the option_logits head, with tiny fakes on the CPU.
Never loads Clef-Flash, Qwen3.5 or the imajev adapter."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from s1stream.backbones.clef_flash import ClefFlash, build_record, load_release_code
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


# ---------- clef_flash ----------

@dataclass
class FakeEncodedQuestion:
    option_ids: tuple


class FakeClefCode:
    """Stands in for joint_schema_model.py: options sorted by id, as the release does for choice fields."""

    def __init__(self):
        self.record = None

    def encode_record(self, tokenizer, record, processor=None):
        self.record = record
        options = sorted(record["questions"]["answer"]["criteria"])
        return SimpleNamespace(questions=(FakeEncodedQuestion(tuple(options)),))

    def collate_records(self, records, pad_token_id, device):
        return {"records": records}


class FakeClefModel:
    def __init__(self, logits):
        self.logits = logits
        self.grad_enabled = None

    def __call__(self, batch):
        self.grad_enabled = torch.is_grad_enabled()
        return [[self.logits]]


def test_clef_record_has_frames_state_and_one_choice_question():
    imgs = images()
    record = build_record(QUESTION, imgs)
    assert record["state"] == QUESTION.state
    assert record["images"] == imgs
    assert record["questions"] == {"answer": {"type": "choice", "instructions": "Abnormal?",
                                              "criteria": {"Yes": None, "No": None}}}


def test_clef_forward_returns_logits_in_question_order_without_grad():
    code = FakeClefCode()
    model = FakeClefModel(torch.tensor([0.25, 1.5]))  # release order: No, Yes
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    out = ClefFlash.from_parts(model, processor, code).forward(ModelInput(images=images()), QUESTION)
    assert out.logits.tolist() == [1.5, 0.25]  # Yes, No
    assert model.grad_enabled is False
    assert len(code.record["images"]) == 8


def test_clef_missing_release_code_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        load_release_code(tmp_path)
    assert "joint_schema_model.py" in str(err.value)
