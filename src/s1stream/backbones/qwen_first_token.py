"""Backbone "qwen_first_token": a Qwen3.5 VLM with no decision training (Qwen/Qwen3.5-9B, the base of d3-flash and
clef-flash), read zero-shot as a plain VLM: one user turn with the frames, then

  <state>
  <question text>
  Options: <option>, <option>, ...
  Answer with one option only.

in the non-thinking chat template (it ends with an empty think block, as the imajev backbone renders Qwen3.5), and the
next-token logits at the answer position, one per option: its first token. forward returns them in question.options
order (option_logits); the options must start with distinct tokens. The backbone reads local files only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from s1stream.registry import BACKBONES
from s1stream.types import BackboneOutput, ModelInput, Question

ANSWER_TAIL = "</think>\n\n"


def build_prompt(question: Question) -> str:
    """The text after the frames: state, question, the options in order, and the answer instruction."""
    return (f"{question.state}\n{question.text}\nOptions: {', '.join(question.options)}\n"
            "Answer with one option only.")


def first_tokens(tokenizer: Any, options: tuple[str, ...]) -> list[int]:
    """The first token id of each option; raise ValueError when two options share it."""
    ids = [tokenizer.encode(option, add_special_tokens=False)[0] for option in options]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Options {options} do not start with distinct tokens")
    return ids


@BACKBONES.register("qwen_first_token")
class QwenFirstToken:
    """Config: {name: qwen_first_token, asset: qwen3.5-9b, memory_cap_gb: 40}. load_config adds path."""

    def __init__(self, path: str | Path, memory_cap_gb: float = 40.0, device: str = "cuda",
                 asset: str | None = None) -> None:
        """On cuda, cap the GPU memory as jev_omni does; load the model in bfloat16 and its processor."""
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if device.startswith("cuda"):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, memory_cap_gb * 2**30 / total_memory))
        model = Qwen3_5ForConditionalGeneration.from_pretrained(Path(path), local_files_only=True,
                                                                dtype=torch.bfloat16).to(device).eval()
        processor = AutoProcessor.from_pretrained(Path(path), local_files_only=True)
        self._init(model, processor, device)

    def _init(self, model: Any, processor: Any, device: str) -> None:
        self.model, self.processor, self.device = model, processor, device
        self._ids: dict[tuple[str, ...], list[int]] = {}

    @classmethod
    def from_parts(cls, model: Any, processor: Any, device: str = "cpu") -> "QwenFirstToken":
        """Build without loading files (used by tests and by __init__)."""
        instance = cls.__new__(cls)
        instance._init(model, processor, device)
        return instance

    def forward(self, inp: ModelInput, question: Question) -> BackboneOutput:
        """1. rendered = chat template of [frames..., build_prompt(question)], non-thinking; it must end with
        ANSWER_TAIL. 2. logits = model(**processor(rendered, frames), logits_to_keep=1).logits[0, -1].
        3. Return the logits of the options' first tokens, float32 on the CPU."""
        if question.options not in self._ids:
            self._ids[question.options] = first_tokens(self.processor.tokenizer, question.options)
        messages = [{"role": "user", "content": [{"type": "image"}] * len(inp.images)
                     + [{"type": "text", "text": build_prompt(question)}]}]
        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                      enable_thinking=False)
        if not rendered.endswith(ANSWER_TAIL):
            raise ValueError("Unexpected Qwen non-thinking template boundary")
        inputs = self.processor(text=[rendered], images=list(inp.images) or None, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1]
        return BackboneOutput(logits=logits[self._ids[question.options]].detach().float().cpu())
