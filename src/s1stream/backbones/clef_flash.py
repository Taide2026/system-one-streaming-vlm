"""Backbone "clef_flash" (V2): Clef-Flash (Qwen3.5 9B + joint schema head) through the release's own code.

The release ships joint_schema_model.py (revision 8b2e5fd); the backbone imports it from the asset folder and
never downloads. One window is one record:
  {"state": question.state, "images": <frames>,
   "questions": {"answer": {"type": "choice", "instructions": question.text, "criteria": {option: None, ...}}}}
The model scores every option in one forward pass. The release sorts choice options by id; forward returns the
logits in question.options order (head: option_logits).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from s1stream.registry import BACKBONES
from s1stream.types import BackboneOutput, ModelInput, Question

QUESTION_ID = "answer"
RELEASE_CODE = "joint_schema_model.py"


def build_record(question: Question, images: list[Any]) -> dict[str, Any]:
    """The Clef record of one window: the frames as images and the question as one choice field."""
    return {
        "state": question.state,
        "images": list(images),
        "questions": {QUESTION_ID: {"type": "choice", "instructions": question.text,
                                    "criteria": {option: None for option in question.options}}},
    }


def load_release_code(path: Path) -> ModuleType:
    """Import path / joint_schema_model.py as a module. Raise FileNotFoundError naming the file when it is missing."""
    file = path / RELEASE_CODE
    if not file.is_file():
        raise FileNotFoundError(file)
    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", file)
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve string annotations through sys.modules, so register before running it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@BACKBONES.register("clef_flash")
class ClefFlash:
    """Config: {name: clef_flash, asset: clef-flash, memory_cap_gb: 40}. load_config adds path."""

    def __init__(self, path: str | Path, memory_cap_gb: float = 40.0, device: str = "cuda",
                 asset: str | None = None) -> None:
        """On cuda, cap the GPU memory as jev_omni does; then load_release_model(path, device, local_files_only)."""
        if device.startswith("cuda"):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, memory_cap_gb * 2**30 / total_memory))
        code = load_release_code(Path(path))
        model, processor = code.load_release_model(Path(path), device=device, local_files_only=True)
        self._init(model, processor, code, device)

    def _init(self, model: Any, processor: Any, code: ModuleType, device: str) -> None:
        self.model, self.processor, self.code, self.device = model, processor, code, device

    @classmethod
    def from_parts(cls, model: Any, processor: Any, code: ModuleType, device: str = "cpu") -> "ClefFlash":
        """Build without loading files (used by tests and by __init__)."""
        instance = cls.__new__(cls)
        instance._init(model, processor, code, device)
        return instance

    def forward(self, inp: ModelInput, question: Question) -> BackboneOutput:
        """encode_record + collate_records + one model call, as the release's `systemone`.

        Return BackboneOutput(logits=<float32 CPU tensor, one logit per option in question.options order>).
        """
        tokenizer = self.processor.tokenizer
        encoded = self.code.encode_record(tokenizer, build_record(question, inp.images), processor=self.processor)
        batch = self.code.collate_records([encoded], tokenizer.pad_token_id, torch.device(self.device))
        with torch.inference_mode():
            logits = self.model(batch)[0][0].float().cpu()
        by_option = dict(zip(encoded.questions[0].option_ids, logits.tolist()))
        return BackboneOutput(logits=torch.tensor([by_option[option] for option in question.options]))
