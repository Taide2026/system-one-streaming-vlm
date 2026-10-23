"""Backbone "h2o_lightning": H2O-Lightning-4B, its image weights (h2oai/h2o-lightning-4b, branch image-preview), read in
process with transformers as its shim reads it through vLLM (h2o_lightning_shim.py in the asset, Apache-2.0).

The prompt, from serve_config.json: the system turn, then one user turn with the images ahead of

  record: <state>
  question: <question text>
  options:
  A) <option>: <option>
  ...

(an option without a description is its own description), then the assistant turn pre-filled with "Answer:" after the
empty think block (thinking off). Each option's score is the logit of " <label>" at the next position, read from the
float32 head (the release serves its head in float32), divided by the image temperature; option_logits' softmax then
gives the shim's probabilities (softmax of the log-probabilities / T is the same distribution).

The model reads at most max_images images (serve_config.json: 4), fewer than a window's frames: the backbone takes
max_images frames evenly spread over the window, its first and last frame included (8 frames -> 0, 2, 5, 7).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from s1stream.registry import BACKBONES
from s1stream.types import BackboneOutput, ModelInput, Question

CHATML = {"system": "<|im_start|>system\n{content}<|im_end|>\n",
          "user": "<|im_start|>user\n{content}<|im_end|>\n",
          "generation": "<|im_start|>assistant\n<think>\n\n</think>\n\n"}
VISION_SLOT = "<|vision_start|><|image_pad|><|vision_end|>"


def spread(n_frames: int, k: int) -> list[int]:
    """k frame indices evenly spread over n_frames, the first and the last included (all of them when k >= n_frames)."""
    if k >= n_frames:
        return list(range(n_frames))
    if k == 1:
        return [0]
    return [round(i * (n_frames - 1) / (k - 1)) for i in range(k)]


def build_prompt(serve: dict, question: Question, n_images: int) -> str:
    """The shim's image prompt (Contract.image_prompt) as text, one VISION_SLOT per image."""
    p = serve["prompt"]
    labels = serve["labels"][:len(question.options)]
    lines = "\n".join(f"{label}) {option}: {option}" for label, option in zip(labels, question.options))
    user = f"record: {question.state}\nquestion: {question.text}\noptions:\n{lines}"
    out = CHATML["system"].replace("{content}", p["system"].strip()) if p.get("system") else ""
    out += CHATML["user"].replace("{content}", VISION_SLOT * n_images + user.strip())
    return out + CHATML["generation"] + p.get("prefill", "Answer:")


def label_ids(tokenizer: Any, prompt: str, labels: list[str]) -> list[int]:
    """Each label's token at the answer slot, as the shim checks it: appending " <label>" to the prompt adds exactly
    one token. Raise ValueError otherwise."""
    base = tokenizer.encode(prompt, add_special_tokens=False)
    ids = []
    for label in labels:
        full = tokenizer.encode(prompt + " " + label, add_special_tokens=False)
        if len(full) != len(base) + 1 or full[:len(base)] != base:
            raise ValueError(f"label {label!r} is not one token at the answer slot")
        ids.append(full[-1])
    return ids


@BACKBONES.register("h2o_lightning")
class H2OLightning:
    """Config: {name: h2o_lightning, asset: h2o-lightning-4b-image, memory_cap_gb: 40}. load_config adds path."""

    def __init__(self, path: str | Path, memory_cap_gb: float = 40.0, device: str = "cuda",
                 asset: str | None = None) -> None:
        """On cuda, cap the GPU memory as jev_omni does; load the model in bfloat16, its processor and
        serve_config.json."""
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if device.startswith("cuda"):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, memory_cap_gb * 2**30 / total_memory))
        model = Qwen3_5ForConditionalGeneration.from_pretrained(Path(path), local_files_only=True,
                                                                dtype=torch.bfloat16).to(device).eval()
        processor = AutoProcessor.from_pretrained(Path(path), local_files_only=True)
        serve = json.loads((Path(path) / "serve_config.json").read_text())
        self._init(model, processor, serve, device)

    def _init(self, model: Any, processor: Any, serve: dict, device: str) -> None:
        self.model, self.processor, self.serve, self.device = model, processor, serve, device
        self.max_images = int(serve.get("image", {}).get("max_images", 4))
        self.temperature = float(serve.get("image", {}).get("temperature", serve["temperature"]))
        self._ids: dict[tuple[str, ...], list[int]] = {}

    @classmethod
    def from_parts(cls, model: Any, processor: Any, serve: dict, device: str = "cpu") -> "H2OLightning":
        """Build without loading files (used by tests and by __init__)."""
        instance = cls.__new__(cls)
        instance._init(model, processor, serve, device)
        return instance

    def forward(self, inp: ModelInput, question: Question) -> BackboneOutput:
        """1. frames = max_images frames spread over the window; prompt = build_prompt(serve, question, len(frames)).
        2. hidden = model.model(**processor(prompt, frames)).last_hidden_state[0, -1] in float32.
        3. Return lm_head rows of the labels' tokens @ hidden / image temperature, in question.options order."""
        frames = [inp.images[i] for i in spread(len(inp.images), self.max_images)]
        prompt = build_prompt(self.serve, question, len(frames))
        if question.options not in self._ids:
            text = build_prompt(self.serve, question, 0)
            self._ids[question.options] = label_ids(self.processor.tokenizer, text,
                                                    self.serve["labels"][:len(question.options)])
        inputs = self.processor(text=[prompt], images=frames or None, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = self.model.model(**inputs).last_hidden_state[0, -1].float()
            rows = self.model.lm_head.weight[self._ids[question.options]].float()
            logits = rows @ hidden / self.temperature
        return BackboneOutput(logits=logits.detach().float().cpu())
