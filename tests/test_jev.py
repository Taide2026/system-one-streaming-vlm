"""Unit jev: jev_omni backbone and option_prob head, with a tiny fake model on the CPU. Never loads Jev-Omni."""
import pytest
import torch
from PIL import Image

from s1stream.backbones.jev_omni import JevOmni, build_prompt, load_parts
from s1stream.heads.option_prob import Head256, OptionProb
from s1stream.types import BackboneOutput, ModelInput, Question

HIDDEN = 4
QUESTION = Question(state="Camera, 2 seconds.", text="Abnormal?", options=("Yes", "No"), positive="Yes")


class FakeDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(HIDDEN, HIDDEN)])
        self.grad_enabled = None

    def forward(self, x):
        self.grad_enabled = torch.is_grad_enabled()
        return (self.layers[0](x),)


class FakeModel(torch.nn.Module):
    """model.language_model is the decoder, like Gemma4UnifiedForConditionalGeneration.model.language_model."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, HIDDEN)
        self.model = torch.nn.Module()
        self.model.language_model = FakeDecoder()

    def forward(self, input_ids, use_cache=False):
        return self.model.language_model(self.embed(input_ids))


class FakeProcessor:
    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def backbone() -> tuple[JevOmni, FakeModel, FakeProcessor]:
    torch.manual_seed(0)
    model, processor = FakeModel(), FakeProcessor()
    return JevOmni.from_parts(model, processor, model.model.language_model, device="cpu"), model, processor


def images(n: int = 8) -> list:
    return [Image.new("RGB", (32, 24), (i * 20, 0, 0)) for i in range(n)]


def test_option_prob_sums_to_one():
    torch.manual_seed(0)
    head = OptionProb.from_module(Head256(HIDDEN))
    probs = head.probabilities(BackboneOutput(hidden=torch.randn(1, HIDDEN)), QUESTION)
    assert list(probs) == ["Yes", "No"]
    assert sum(probs.values()) == pytest.approx(1.0)


def test_forward_runs_without_grad_and_returns_no_grad_fn():
    bb, model, _ = backbone()
    out = bb.forward(ModelInput(images=images()), QUESTION)
    assert model.model.language_model.grad_enabled is False
    assert out.hidden.shape == (1, HIDDEN)
    assert out.hidden.grad_fn is None
    assert out.hidden.requires_grad is False


def test_prompt_matches_jev_omni_format():
    assert build_prompt(QUESTION) == (
        "Camera, 2 seconds.\n\n---\n\nQUESTION: Abnormal?\n\nOPTIONS:\n1. Yes\n2. No\n\n"
        "Reply with only the number of the correct option (1-2).\nOutput a single number and nothing else.")


def test_option_prob_rejects_one_option():
    head = OptionProb.from_module(Head256(HIDDEN))
    one = Question(state="s", text="q", options=("Yes",), positive="Yes")
    with pytest.raises(ValueError):
        head.probabilities(BackboneOutput(hidden=torch.zeros(1, HIDDEN)), one)


def test_backbone_passes_eight_images_in_order():
    bb, _, processor = backbone()
    imgs = images()
    bb.forward(ModelInput(images=imgs), QUESTION)
    content = processor.messages[0]["content"]
    assert [c["image"] for c in content if c["type"] == "image"] == imgs
    assert content[-1] == {"type": "text", "text": build_prompt(QUESTION)}
    assert processor.kwargs["enable_thinking"] is False


def test_load_parts_missing_config_raises_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tmp_path / "head.pt").touch()
    with pytest.raises(FileNotFoundError) as err:
        load_parts(tmp_path, "cpu")
    assert "config.json" in str(err.value)
