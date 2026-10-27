"""Probe conditions A / B / C on the d3-flash runtime, and the hidden-state tap (finetune-layer-probing decisions 7, 9).

All three conditions load the d3-flash release runtime (d3_runtime.D3), so the processor, the prompt rendering and the
input tokens are the same. A condition changes only two things:
  weights:   the d3-flash backbone (C), or the qwen3.5-9b tensors loaded into the same backbone (A, B)
  attention: causal (A: the stock forward, runtime.encoder = None) or noncausal (B, C: d3_runtime.PromptEncoder,
             the 8 full-attention layers see every prompt position)
The runtime picks its path in image_logits: runtime.encoder set -> PromptEncoder; None -> self.backbone(**inputs).
Both paths run runtime.backbone.language_model.layers in order and then its norm, so the tap hooks those modules.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from s1stream.types import Question

BASE_DROPPED = ("lm_head.", "mtp.")  # qwen3.5-9b tensors that are not part of the d3 backbone
BASE_PREFIX = "model."


class ConditionError(Exception):
    """A condition cannot be built. The message names the keys or the setting."""


@dataclass
class WindowResult:
    """One tapped window. last, visual: float32 CPU tensors (n_states, hidden), no grad_fn."""
    last: torch.Tensor
    visual: torch.Tensor
    p_yes: float


def rename_base_weights(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """qwen3.5-9b checkpoint keys -> d3 backbone keys: drop keys that start with BASE_DROPPED, strip a leading
    BASE_PREFIX from the rest ("model.language_model.norm.weight" -> "language_model.norm.weight")."""
    return {key.removeprefix(BASE_PREFIX): value for key, value in state.items()
            if not key.startswith(BASE_DROPPED)}


def swap_weights(backbone: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load state into backbone with load_state_dict(strict=True), keeping each parameter's device and dtype.

    Raise ConditionError that lists every missing and every unexpected key (sorted) when the key sets differ.
    """
    expected = set(backbone.state_dict())
    actual = set(state)
    if expected != actual:
        raise ConditionError(f"missing keys: {sorted(expected - actual)}; unexpected keys: {sorted(actual - expected)}")
    backbone.load_state_dict(state, strict=True)


def set_attention(runtime: Any, mode: Literal["causal", "noncausal"],
                  encoder_factory: Callable[[Any], Any]) -> None:
    """causal: runtime.encoder = None. noncausal: runtime.encoder = encoder_factory(runtime.backbone.language_model).

    Raise ConditionError when runtime.fast is not None (the fused fast path skips the hooked modules) or when mode
    is neither "causal" nor "noncausal".
    """
    if runtime.fast is not None:
        raise ConditionError("fast path skips decoder layers")
    if mode == "causal":
        runtime.encoder = None
    elif mode == "noncausal":
        runtime.encoder = encoder_factory(runtime.backbone.language_model)
    else:
        raise ConditionError(f"unknown attention mode: {mode}")


class _TappedProcessor:
    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.input_ids: torch.Tensor | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.processor(*args, **kwargs)
        self.input_ids = result["input_ids"].detach()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.processor, name)


class HiddenTap:
    """Forward hooks on every decoder layer of runtime.backbone.language_model.layers and on its norm
    (n_states = n_layers + 1), and a wrapper around runtime.processor that keeps the input_ids of its last call.

    The wrapper is callable like the processor and forwards every other attribute to it, so the runtime keeps working.
    One tapped forward must be batch 1. The hooks store detached outputs; a layer may return a tensor or a tuple
    whose first item is the hidden state.
    """

    def __init__(self, runtime: Any, image_token_id: int) -> None:
        self.runtime = runtime
        self.image_token_id = image_token_id
        self.processor = runtime.processor
        self.wrapped_processor = _TappedProcessor(self.processor)
        runtime.processor = self.wrapped_processor
        modules = list(runtime.backbone.language_model.layers) + [runtime.backbone.language_model.norm]
        self.n_states = len(modules)
        self.states: list[torch.Tensor] = []
        self.hooks = [module.register_forward_hook(self._capture) for module in modules]

    def _capture(self, module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self.states.append(hidden.detach())

    def reset(self) -> None:
        """Forget the states and input_ids of the previous forward."""
        self.states.clear()
        self.wrapped_processor.input_ids = None

    def collect(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(last, visual) of the one forward since reset(): last = every state at the last position, visual = every
        state averaged over the positions where input_ids == image_token_id. float32 CPU, shape (n_states, hidden).

        Raise RuntimeError when the state count is not n_states, when input_ids is missing, when the batch is not 1,
        or when the prompt has no image token.
        """
        ids = self.wrapped_processor.input_ids
        if len(self.states) != self.n_states:
            raise RuntimeError(f"expected {self.n_states} hidden states, got {len(self.states)}")
        if ids is None:
            raise RuntimeError("processor did not return input_ids")
        if ids.ndim != 2 or ids.shape[0] != 1 or any(state.ndim != 3 or state.shape[0] != 1 for state in self.states):
            raise RuntimeError("tapped forward must have batch 1")
        positions = (ids[0] == self.image_token_id).nonzero().flatten()
        if positions.numel() == 0:
            raise RuntimeError("prompt has no image token")
        last = torch.stack([state[0, -1].float().cpu() for state in self.states])
        visual = torch.stack([state[0, positions.to(state.device)].float().mean(dim=0).cpu()
                              for state in self.states])
        return last, visual

    def close(self) -> None:
        """Remove the hooks and put the original processor back."""
        for hook in self.hooks:
            hook.remove()
        self.runtime.processor = self.processor


def yes_probability(lm_head: torch.nn.Linear, hidden: torch.Tensor, yes_id: int, no_id: int) -> float:
    """Softmax over the two LM-head logits of yes_id and no_id for hidden (shape (hidden,)): the share of yes_id."""
    with torch.inference_mode():
        logits = lm_head(hidden.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype))
        return torch.softmax(torch.stack((logits[yes_id], logits[no_id])).float(), dim=0)[0].item()


@dataclass
class ProbeModel:
    """A loaded condition. lm_head and the token ids are set for base weights (A, B) and None for C."""
    condition: str
    runtime: Any
    tap: HiddenTap
    lm_head: torch.nn.Linear | None
    yes_id: int | None
    no_id: int | None

    def run_window(self, images: list[Any], question: Question) -> WindowResult:
        """One window under torch.inference_mode(): tap.reset(), runtime.system_one(state=question.state,
        images=images, questions=s1stream.backbones.d3.build_questions(question)), tap.collect().

        p_yes: lm_head is None -> the runtime's probability of question.positive (a str); else
        yes_probability(lm_head, last[-1], yes_id, no_id). Raise RuntimeError when the runtime answers with an error.
        """
        from s1stream.backbones.d3 import QUESTION_ID, build_questions

        with torch.inference_mode():
            self.tap.reset()
            result = self.runtime.system_one(state=question.state, images=images,
                                             questions=build_questions(question))
            answer = result["answers"][QUESTION_ID]
            if "error" in answer:
                raise RuntimeError(f"d3 could not answer: {answer['error']}: {answer.get('message', '')}")
            last, visual = self.tap.collect()
            if self.lm_head is None:
                p_yes = float(answer["probabilities"][question.positive])
            else:
                if self.yes_id is None or self.no_id is None:
                    raise RuntimeError("base LM head needs Yes and No token ids")
                p_yes = yes_probability(self.lm_head, last[-1], self.yes_id, self.no_id)
            return WindowResult(last=last, visual=visual, p_yes=p_yes)


def load_condition(name: str, runtime_path: Path, weights_path: Path | None,
                   attention: Literal["causal", "noncausal"], device: str = "cuda") -> ProbeModel:
    """Load the d3 runtime from runtime_path (s1stream.backbones.d3.load_release_code, D3.from_pretrained with
    local_files_only=True). When weights_path is set: read every *.safetensors file there, rename_base_weights,
    swap_weights(runtime.backbone, ...), and keep lm_head.weight as an LM head on the runtime device; yes_id and no_id
    are the first token ids of "Yes" and "No" from runtime.tokenizer (add_special_tokens=False). Then
    set_attention(runtime, attention, code.PromptEncoder) and HiddenTap(runtime, runtime.backbone.config.image_token_id).
    Not unit-tested (needs the checkpoints); the manager checks it on the GPU (decision 13).
    """
    from safetensors.torch import load_file

    from s1stream.backbones.d3 import load_release_code, runtime_device

    code = load_release_code(runtime_path)
    runtime = code.D3.from_pretrained(runtime_path, device=runtime_device(device), local_files_only=True)
    lm_head = None
    yes_id = no_id = None
    if weights_path is not None:
        state: dict[str, torch.Tensor] = {}
        head_weight = None
        for shard in sorted(weights_path.glob("*.safetensors")):
            tensors = load_file(str(shard), device="cpu")
            if "lm_head.weight" in tensors:
                head_weight = tensors.pop("lm_head.weight")
            state.update(rename_base_weights(tensors))
        swap_weights(runtime.backbone, state)
        if head_weight is None:
            raise ConditionError("missing lm_head.weight")
        vocab, hidden = head_weight.shape
        lm_head = torch.nn.Linear(hidden, vocab, bias=False, device=runtime.device, dtype=torch.bfloat16)
        lm_head.load_state_dict({"weight": head_weight})
        lm_head.requires_grad_(False)
        yes_id = runtime.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        no_id = runtime.tokenizer("No", add_special_tokens=False)["input_ids"][0]
    set_attention(runtime, attention, code.PromptEncoder)
    tap = HiddenTap(runtime, runtime.backbone.config.image_token_id)
    return ProbeModel(name, runtime, tap, lm_head, yes_id, no_id)
