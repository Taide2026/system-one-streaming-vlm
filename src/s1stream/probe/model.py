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
    raise NotImplementedError


def swap_weights(backbone: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load state into backbone with load_state_dict(strict=True), keeping each parameter's device and dtype.

    Raise ConditionError that lists every missing and every unexpected key (sorted) when the key sets differ.
    """
    raise NotImplementedError


def set_attention(runtime: Any, mode: Literal["causal", "noncausal"],
                  encoder_factory: Callable[[Any], Any]) -> None:
    """causal: runtime.encoder = None. noncausal: runtime.encoder = encoder_factory(runtime.backbone.language_model).

    Raise ConditionError when runtime.fast is not None (the fused fast path skips the hooked modules) or when mode
    is neither "causal" nor "noncausal".
    """
    raise NotImplementedError


class HiddenTap:
    """Forward hooks on every decoder layer of runtime.backbone.language_model.layers and on its norm
    (n_states = n_layers + 1), and a wrapper around runtime.processor that keeps the input_ids of its last call.

    The wrapper is callable like the processor and forwards every other attribute to it, so the runtime keeps working.
    One tapped forward must be batch 1. The hooks store detached outputs; a layer may return a tensor or a tuple
    whose first item is the hidden state.
    """

    def __init__(self, runtime: Any, image_token_id: int) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Forget the states and input_ids of the previous forward."""
        raise NotImplementedError

    def collect(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(last, visual) of the one forward since reset(): last = every state at the last position, visual = every
        state averaged over the positions where input_ids == image_token_id. float32 CPU, shape (n_states, hidden).

        Raise RuntimeError when the state count is not n_states, when input_ids is missing, when the batch is not 1,
        or when the prompt has no image token.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Remove the hooks and put the original processor back."""
        raise NotImplementedError


def yes_probability(lm_head: torch.nn.Linear, hidden: torch.Tensor, yes_id: int, no_id: int) -> float:
    """Softmax over the two LM-head logits of yes_id and no_id for hidden (shape (hidden,)): the share of yes_id."""
    raise NotImplementedError


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
        raise NotImplementedError


def load_condition(name: str, runtime_path: Path, weights_path: Path | None,
                   attention: Literal["causal", "noncausal"], device: str = "cuda") -> ProbeModel:
    """Load the d3 runtime from runtime_path (s1stream.backbones.d3.load_release_code, D3.from_pretrained with
    local_files_only=True). When weights_path is set: read every *.safetensors file there, rename_base_weights,
    swap_weights(runtime.backbone, ...), and keep lm_head.weight as an LM head on the runtime device; yes_id and no_id
    are the first token ids of "Yes" and "No" from runtime.tokenizer (add_special_tokens=False). Then
    set_attention(runtime, attention, code.PromptEncoder) and HiddenTap(runtime, runtime.backbone.config.image_token_id).
    Not unit-tested (needs the checkpoints); the manager checks it on the GPU (decision 13).
    """
    raise NotImplementedError
