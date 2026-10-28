"""Backbone "d3" (V2): Decision 3.0 (vllm-sr/d3-nano and its siblings) through the release's own runtime.

Every size ships the same d3_runtime.py (revisions in assets.yaml); the backbone imports it from the asset folder
and never downloads. One window is one System One request, with the window as clef_flash sends it:
  system_one(state=question.state, images=<frames>,
             questions={"answer": {"type": "choice", "instructions": question.text,
                                   "criteria": {option: None, ...}}})
The runtime returns a probability for every option without generating text; forward returns their logs in
question.options order, so option_logits gives the same probabilities back. The prompt is read with the
checkpoint's attention mode (d3-nano: noncausal full attention over the prompt).

adapter=<run dir> (s1stream.train, trainee d3): the peft adapter is merged into runtime.backbone, readout.safetensors
replaces the code read-out and calibration.json's temperature replaces the release's.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from s1stream.registry import BACKBONES
from s1stream.types import BackboneOutput, ModelInput, Question

QUESTION_ID = "answer"
RELEASE_CODE = "d3_runtime.py"


def build_questions(question: Question) -> dict[str, Any]:
    """The System One questions of one window: the question as one choice field, options without descriptions
    (option k is answer code k, in question.options order)."""
    return {QUESTION_ID: {"type": "choice", "instructions": question.text,
                          "criteria": {option: None for option in question.options}}}


def runtime_device(device: str) -> str:
    """The device as the runtime takes it: it calls torch.cuda.set_device, which needs an index ("cuda" -> "cuda:0")."""
    return "cuda:0" if device == "cuda" else device


def load_release_code(path: Path) -> ModuleType:
    """Import path / d3_runtime.py as a module. Raise FileNotFoundError naming the file when it is missing.

    The runtime imports d3_format (and d3_fast when present) as top-level modules, so path goes on sys.path.
    """
    file = path / RELEASE_CODE
    if not file.is_file():
        raise FileNotFoundError(file)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("d3_runtime", file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_adapter(runtime: Any, run_dir: Path) -> None:
    """Merge run_dir/adapter into runtime.backbone, load run_dir/readout.safetensors, set the temperature."""
    from peft import PeftModel
    from safetensors.torch import load_file

    if (run_dir / "adapter").is_dir():  # a linear probe (lora r 0) ships the read-out only
        PeftModel.from_pretrained(runtime.backbone, run_dir / "adapter").merge_and_unload()
    weight = load_file(run_dir / "readout.safetensors")["weight"]
    runtime.readout = weight.to(runtime.readout.device, runtime.readout.dtype)
    calibration = run_dir / "calibration.json"
    if calibration.is_file():
        runtime.temperature = float(json.loads(calibration.read_text()).get("temperature", 1.0))
    runtime.backbone.eval().requires_grad_(False)
    # d3_fast caches weights of its own; after a merge it must not be used.
    runtime.fast = None


@BACKBONES.register("d3")
class D3Backbone:
    """Config: {name: d3, asset: d3-nano | d3-flash | ..., memory_cap_gb: 40, adapter: <run dir> (optional)}.
    load_config adds path."""

    def __init__(self, path: str | Path, memory_cap_gb: float = 40.0, device: str = "cuda",
                 asset: str | None = None, adapter: str | Path | None = None) -> None:
        """On cuda, cap the GPU memory; then D3.from_pretrained(path, local_files_only), which also checks the files
        against the release's MODEL_MANIFEST.json."""
        if device.startswith("cuda"):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, memory_cap_gb * 2**30 / total_memory))
        code = load_release_code(Path(path))
        model = code.D3.from_pretrained(Path(path), device=runtime_device(device), local_files_only=True)
        self._init(model, code)
        if adapter is not None:
            load_adapter(model, Path(adapter))

    def _init(self, model: Any, code: ModuleType | None = None) -> None:
        self.model, self.code = model, code

    @classmethod
    def from_parts(cls, model: Any) -> "D3Backbone":
        """Build without loading files (used by tests and by __init__)."""
        instance = cls.__new__(cls)
        instance._init(model)
        return instance

    def forward(self, inp: ModelInput, question: Question) -> BackboneOutput:
        """One system_one call with the frames as images.

        Return BackboneOutput(logits=<float32 CPU tensor, log p(option) in question.options order>).
        Raise RuntimeError with the runtime's message when it answers with an error.
        """
        result = self.model.system_one(state=question.state, images=list(inp.images),
                                       questions=build_questions(question))
        answer = result["answers"][QUESTION_ID]
        if "error" in answer:
            raise RuntimeError(f"d3 could not answer: {answer['error']}: {answer.get('message', '')}")
        probabilities = answer["probabilities"]
        return BackboneOutput(logits=torch.tensor(
            [math.log(max(probabilities[option], 1e-30)) for option in question.options], dtype=torch.float32))
