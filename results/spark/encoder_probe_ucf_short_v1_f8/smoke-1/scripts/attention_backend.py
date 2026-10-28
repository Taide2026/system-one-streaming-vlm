"""Why the decision 9 check failed: compare qwen35-vit features with vit-1 under 3 attention settings.

A: PyTorch default SDPA backend choice (what encoder-extract smoke-1 used)
B: the d3 runtime setting: sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH]) (no cuDNN), as vit-1
C: float32 weights and inputs, default backend (how far bfloat16 itself is from vit-1)
The first 5 test windows. Prints the max abs difference / max abs value per kind, as in check_equal.py.
"""
import contextlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import s1stream.plugins  # noqa: F401
from s1stream.encoder.adapters import load_encoder
from s1stream.encoder.extract import load_encoder_config
from s1stream.probe.extract import limit_sets, plan_sets
from s1stream.registry import DATASETS, INPUT_FORMS

config = load_encoder_config(Path("configs/encoder_probe_ucf_short_v1_f8.yaml"))
rows = limit_sets(plan_sets(DATASETS.build(config.train).items(), DATASETS.build(config.test).items(),
                            config.window), 5)["test"]
form = INPUT_FORMS.build(config.input_form)
frames = [form.build(row.item, row.window, None).images for row in rows]
old = Path(os.environ["S1_FEATURES"]) / "probe_ucf_short_v1_f8" / "V" / "vit-1" / "test"
ref = {"patch_mean": np.load(old / "visual.npy", mmap_mode="r")[:5].astype(np.float32),
       "patch_max": np.load(old / "last.npy", mmap_mode="r")[:5].astype(np.float32)}
torch.cuda.set_per_process_memory_fraction(24 / (torch.cuda.get_device_properties(0).total_memory / 2**30))


def run(adapter, context):
    with context:
        outs = [adapter.encode(f) for f in frames]
    return {kind: np.stack([o.kinds[kind].numpy() for o in outs]) for kind in ref}


bf16 = load_encoder("qwen35-vit", config.encoder_paths["qwen35-vit"], "cuda")
results = {"A default": run(bf16, contextlib.nullcontext()),
           "B no cuDNN": run(bf16, sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                                                 SDPBackend.MATH]))}
bf16.model.float()
results["C float32"] = run(bf16, contextlib.nullcontext())
for name, got in results.items():
    for kind in ref:
        diff = float(np.abs(got[kind] - ref[kind]).max())
        print(f"{name:12s} {kind}: max abs diff {diff:.4g}, ratio {diff / float(np.abs(ref[kind]).max()):.2e}")
for kind in ref:
    diff = float(np.abs(results["A default"][kind] - results["B no cuDNN"][kind]).max())
    print(f"A vs B {kind}: max abs diff {diff:.4g}")
def per_state(a, b):
    return np.abs(a - b).max(axis=(0, 2)) / np.abs(b).max(axis=(0, 2))


def cosine(a, b):
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    return float((np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).min())


for kind in ref:
    print(f"per-state ratio {kind}, ours bf16 vs vit-1:  ", np.array2string(per_state(results["A default"][kind], ref[kind]), precision=4))
    print(f"per-state ratio {kind}, ours bf16 vs ours fp32:", np.array2string(per_state(results["A default"][kind], results["C float32"][kind]), precision=4))
    print(f"min cosine {kind}: bf16 vs vit-1 {cosine(results['A default'][kind], ref[kind]):.6f}, "
          f"bf16 vs fp32 {cosine(results['A default'][kind], results['C float32'][kind]):.6f}, "
          f"fp32 vs vit-1 {cosine(results['C float32'][kind], ref[kind]):.6f}")
print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
