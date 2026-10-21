"""Backbone "d1": Liquid AI d1 (LiquidAI/d1-3B, LFM2.5-VL) through the release's own code (modeling_d1.py).

The release loads with AutoModel (trust_remote_code, files from the asset folder only) and answers System One
requests as Decision 3.0's runtime does, so one window is the same request as in the d3 backbone:
  system_one(state=question.state, images=<frames>,
             questions={"answer": {"type": "choice", "instructions": question.text,
                                   "criteria": {option: None, ...}}})
and forward is D3Backbone.forward: the logs of the option probabilities in question.options order (option_logits).
"""
from __future__ import annotations

from pathlib import Path

import torch

from s1stream.backbones.d3 import D3Backbone
from s1stream.registry import BACKBONES


@BACKBONES.register("d1")
class D1Backbone(D3Backbone):
    """Config: {name: d1, asset: d1-3b, memory_cap_gb: 40}. load_config adds path."""

    def __init__(self, path: str | Path, memory_cap_gb: float = 40.0, device: str = "cuda",
                 asset: str | None = None) -> None:
        """On cuda, cap the GPU memory as jev_omni does; then AutoModel.from_pretrained(path, trust_remote_code,
        local_files_only), bfloat16 on cuda, float32 on the CPU."""
        from transformers import AutoModel

        cuda = device.startswith("cuda")
        if cuda:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, memory_cap_gb * 2**30 / total_memory))
        model = AutoModel.from_pretrained(Path(path), trust_remote_code=True, local_files_only=True,
                                          dtype=torch.bfloat16 if cuda else torch.float32)
        self._init(model.to(device).eval())
