"""Unit probe-model: conditions, weight swap, hidden-state tap, P(Yes) (finetune-layer-probing decisions 7, 9).

CPU only. A fake runtime mimics the parts of d3_runtime.D3 that the probe touches: backbone.language_model.layers
and .norm, processor (returns input_ids), encoder, fast and system_one.
"""
import pytest
import torch

from s1stream.probe.model import (ConditionError, HiddenTap, ProbeModel, rename_base_weights, set_attention,
                                  swap_weights, yes_probability)
from s1stream.types import Question

IMAGE = 7  # image token id of the fake processor
HIDDEN, LAYERS = 4, 3
QUESTION = Question(state="s", text="q", options=("Yes", "No"), positive="Yes")


class Layer(torch.nn.Module):
    """Adds a fixed vector; returns a tuple like some transformers versions. Records whether grad was enabled."""

    def __init__(self, shift: float) -> None:
        super().__init__()
        self.shift = torch.nn.Parameter(torch.full((HIDDEN,), shift))
        self.grad_seen: list[bool] = []

    def forward(self, hidden):
        self.grad_seen.append(torch.is_grad_enabled())
        return (hidden + self.shift,)


class LanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(10, HIDDEN)
        self.layers = torch.nn.ModuleList(Layer(float(i + 1)) for i in range(LAYERS))
        self.norm = torch.nn.LayerNorm(HIDDEN)


class Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = LanguageModel()

    def forward(self, input_ids):
        hidden = self.language_model.embed(input_ids)
        for layer in self.language_model.layers:
            hidden = layer(hidden)[0]
        return self.language_model.norm(hidden)


class Processor:
    """Callable like a transformers processor; ids is the fixed prompt; tokenizer is a plain attribute."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.tokenizer = "tok"

    def __call__(self, text, images, padding=True, return_tensors="pt"):
        return {"input_ids": torch.tensor([self.ids])}


class Encoder:
    def __init__(self, language_model) -> None:
        self.language_model = language_model

    def __call__(self, embeds):
        hidden = embeds
        for layer in self.language_model.layers:
            hidden = layer(hidden)[0]
        return self.language_model.norm(hidden)


class FakeRuntime:
    def __init__(self, ids=(1, IMAGE, IMAGE, 2, 3)) -> None:
        self.backbone = Backbone()
        self.processor = Processor(list(ids))
        self.encoder = None
        self.fast = None

    def system_one(self, state, images, questions):
        inputs = self.processor(text=["p"], images=images)
        if self.encoder is not None:
            self.encoder(self.backbone.language_model.embed(inputs["input_ids"]))
        else:
            self.backbone(inputs["input_ids"])
        return {"answers": {"answer": {"probabilities": {"Yes": 0.25, "No": 0.75}}}}


def manual_states(runtime: FakeRuntime) -> torch.Tensor:
    """(n_states, L, hidden) computed without hooks."""
    with torch.no_grad():
        hidden = runtime.backbone.language_model.embed(torch.tensor([runtime.processor.ids]))[0]
        states = []
        for layer in runtime.backbone.language_model.layers:
            hidden = layer(hidden)[0]
            states.append(hidden)
        states.append(runtime.backbone.language_model.norm(hidden))
    return torch.stack(states)


def test_rename_base_weights():
    tensor = torch.zeros(1)
    state = {"model.language_model.norm.weight": tensor, "model.visual.blocks.0.attn.proj.bias": tensor,
             "lm_head.weight": tensor, "mtp.fc.weight": tensor, "mtp.layers.0.mlp.up_proj.weight": tensor}
    assert set(rename_base_weights(state)) == {"language_model.norm.weight", "visual.blocks.0.attn.proj.bias"}


def test_swap_weights_names_missing_and_unexpected():
    backbone = Backbone()
    state = {name: value.clone() for name, value in backbone.state_dict().items()}
    del state["language_model.norm.bias"]
    state["language_model.extra.weight"] = torch.zeros(1)
    with pytest.raises(ConditionError) as err:
        swap_weights(backbone, state)
    assert "language_model.norm.bias" in str(err.value) and "language_model.extra.weight" in str(err.value)


def test_swap_weights_copies_values():
    backbone, source = Backbone(), Backbone()
    for parameter in source.parameters():
        torch.nn.init.normal_(parameter)
    swap_weights(backbone, {name: value.clone() for name, value in source.state_dict().items()})
    for name, value in source.state_dict().items():
        assert torch.equal(backbone.state_dict()[name], value)


def test_set_attention_modes():
    runtime = FakeRuntime()
    set_attention(runtime, "noncausal", Encoder)
    assert isinstance(runtime.encoder, Encoder) and runtime.encoder.language_model is runtime.backbone.language_model
    set_attention(runtime, "causal", Encoder)
    assert runtime.encoder is None


def test_set_attention_rejects_fast_path():
    runtime = FakeRuntime()
    runtime.fast = object()
    with pytest.raises(ConditionError):
        set_attention(runtime, "noncausal", Encoder)


def test_tap_noncausal_last_and_visual():
    runtime = FakeRuntime()
    set_attention(runtime, "noncausal", Encoder)
    tap = HiddenTap(runtime, IMAGE)
    tap.reset()
    runtime.system_one("s", [], {})
    last, visual = tap.collect()
    states = manual_states(runtime)
    assert last.shape == visual.shape == (LAYERS + 1, HIDDEN) and last.dtype == torch.float32
    assert torch.allclose(last, states[:, -1])
    assert torch.allclose(visual, states[:, 1:3].mean(dim=1))


def test_tap_causal_takes_ids_from_processor():
    runtime = FakeRuntime()
    set_attention(runtime, "causal", Encoder)
    tap = HiddenTap(runtime, IMAGE)
    tap.reset()
    runtime.system_one("s", [], {})
    last, visual = tap.collect()
    states = manual_states(runtime)
    assert last.shape == (LAYERS + 1, HIDDEN)
    assert torch.allclose(visual, states[:, 1:3].mean(dim=1))
    assert runtime.processor.tokenizer == "tok"  # the wrapper forwards other attributes
    tap.close()
    assert isinstance(runtime.processor, Processor)


def test_tap_collect_errors():
    runtime = FakeRuntime(ids=(1, 2, 3))
    tap = HiddenTap(runtime, IMAGE)
    tap.reset()
    runtime.system_one("s", [], {})
    with pytest.raises(RuntimeError):
        tap.collect()  # no image token
    tap.reset()
    with pytest.raises(RuntimeError):
        tap.collect()  # no forward since reset: wrong state count


def test_run_window_has_no_grad():
    runtime = FakeRuntime()
    set_attention(runtime, "noncausal", Encoder)
    model = ProbeModel(condition="C", runtime=runtime, tap=HiddenTap(runtime, IMAGE), lm_head=None,
                       yes_id=None, no_id=None)
    result = model.run_window([], QUESTION)
    seen = [flag for layer in runtime.backbone.language_model.layers for flag in layer.grad_seen]
    assert seen and not any(seen)
    assert result.last.grad_fn is None and result.visual.grad_fn is None
    assert result.p_yes == pytest.approx(0.25)


def test_yes_probability_two_way_softmax():
    head = torch.nn.Linear(HIDDEN, 5, bias=False)
    hidden = torch.randn(HIDDEN)
    logits = head(hidden).detach()
    expected = torch.softmax(torch.stack([logits[3], logits[1]]), dim=0)[0].item()
    assert yes_probability(head, hidden, yes_id=3, no_id=1) == pytest.approx(expected)
