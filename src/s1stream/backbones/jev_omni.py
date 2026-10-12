"""Backbone "jev_omni" (V2): Jev-Omni (Gemma 4 12B) forward pass, reimplemented from jev_omni.py at revision 5addda8.

The backbone reads only the local asset folder and never downloads. It returns the last hidden state of the
text decoder at the last prompt position; the answer head (heads/option_prob.py) turns it into probabilities.
Smoke run 2026-10-09: 1.42 s per window of 8 frames, 22.6 GB peak GPU memory, 2228 tokens.
"""
from __future__ import annotations

import contextlib
import inspect
from pathlib import Path
from typing import Any

import torch

from s1stream.registry import BACKBONES
from s1stream.types import BackboneOutput, ModelInput, Question

DECODER_PATHS = ("model.language_model", "language_model.model", "model.text_model", "model")


def build_prompt(question: Question) -> str:
    """The Jev-Omni prompt, character for character as in jev_omni.py `_prompt`:

    f"{state}\\n\\n---\\n\\nQUESTION: {text}\\n\\nOPTIONS:\\n1. {opt1}\\n2. {opt2}\\n\\n"
    f"Reply with only the number of the correct option (1-{n}).\\nOutput a single number and nothing else."
    """
    choices = "\n".join(f"{i + 1}. {option}" for i, option in enumerate(question.options))
    return (f"{question.state}\n\n---\n\nQUESTION: {question.text}\n\nOPTIONS:\n{choices}\n\n"
            f"Reply with only the number of the correct option (1-{len(question.options)}).\n"
            "Output a single number and nothing else.")


def find_decoder(model: Any) -> Any:
    """Return the first attribute path of DECODER_PATHS that exists on model and has a `layers` attribute.

    Raise RuntimeError when none matches.
    """
    for path in DECODER_PATHS:
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and hasattr(node, "layers"):
            return node
    raise RuntimeError("Could not locate the Gemma text backbone")


def load_parts(path: Path, device: str) -> tuple[Any, Any, Any]:
    """Load (model, processor, decoder) from the local folder path, in bfloat16, with local_files_only=True.

    Raise FileNotFoundError naming the file when path / "config.json" does not exist, before any transformers call.
    The model class is getattr(transformers, config.architectures[0]); the model is put in eval mode on device.
    """
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    import transformers
    config = transformers.AutoConfig.from_pretrained(path, local_files_only=True)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        path, dtype=torch.bfloat16, device_map=device, local_files_only=True).eval()
    processor = transformers.AutoProcessor.from_pretrained(path, local_files_only=True)
    return model, processor, find_decoder(model)


@BACKBONES.register("jev_omni")
class JevOmni:
    """Config: {name: jev_omni, asset: jev-omni, memory_cap_gb: 40}. load_config adds path."""

    def __init__(self, path: str | Path, memory_cap_gb: float = 40.0, device: str = "cuda",
                 asset: str | None = None) -> None:
        """On cuda, set torch.cuda.set_per_process_memory_fraction(memory_cap_gb GiB / total GPU memory) first,
        then load_parts(Path(path), device) and attach the hook (see from_parts)."""
        if device.startswith("cuda"):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            fraction = min(1.0, memory_cap_gb * 2**30 / total_memory)
            torch.cuda.set_per_process_memory_fraction(fraction)
        self.model, self.processor, decoder = load_parts(Path(path), device)
        self.device = device
        self._register_hook(decoder)

    def _register_hook(self, decoder: Any) -> None:
        self._capture: dict[str, torch.Tensor] = {}

        def capture(_module: Any, _args: Any, output: Any) -> None:
            hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
            self._capture["hidden"] = hidden[:, -1].float()

        decoder.register_forward_hook(capture)

    @classmethod
    def from_parts(cls, model: Any, processor: Any, decoder: Any, device: str = "cpu") -> "JevOmni":
        """Build without loading files (used by tests and by __init__).

        Registers a forward hook on decoder that stores
        (output.last_hidden_state if present else output[0])[:, -1] as float32.
        """
        instance = cls.__new__(cls)
        instance.model = model
        instance.processor = processor
        instance.device = device
        instance._register_hook(decoder)
        return instance

    def forward(self, inp: ModelInput, question: Question) -> BackboneOutput:
        """One forward pass, as jev_omni.py `predict` up to the hidden state.

        1. content = one {"type": "image", "image": img} per image of inp, in order, then
           {"type": "text", "text": build_prompt(question)}.
        2. inputs = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
           tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False).
        3. Move every tensor to the device; cast floating tensors to the model dtype.
        4. Under torch.inference_mode() (and torch.autocast("cuda", dtype=torch.bfloat16) on cuda only), call
           model(**inputs, use_cache=False), adding logits_to_keep=1 when model.forward accepts it.
        5. Return BackboneOutput(hidden=<hooked hidden>.detach().cpu()); raise RuntimeError when the hook saw nothing.
        """
        content = [{"type": "image", "image": image} for image in inp.images]
        content.append({"type": "text", "text": build_prompt(question)})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        dtype = next(self.model.parameters()).dtype
        inputs = {key: value.to(self.device, dtype=dtype) if torch.is_floating_point(value)
                  else value.to(self.device) for key, value in inputs.items()}
        extra = {}
        if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
            extra["logits_to_keep"] = 1
        autocast = contextlib.nullcontext()
        if self.device.startswith("cuda"):
            autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        self._capture.clear()
        with torch.inference_mode(), autocast:
            self.model(**inputs, use_cache=False, **extra)
        if "hidden" not in self._capture:
            raise RuntimeError("Decoder hook did not capture a hidden state")
        return BackboneOutput(hidden=self._capture["hidden"].detach().cpu())
